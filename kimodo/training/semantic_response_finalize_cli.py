# SPDX-License-Identifier: Apache-2.0
"""Normalize count-repair provenance and fail-safe expert-rejected captions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .file_permissions import publish_file
from .llm_api_augmentation_cli import (
    BATCH_JUDGE_PROMPT,
    _canonical_sha256,
    _chat_payload,
    _receipt_summary,
)
from .qwen_augmentation_cli import _source_preserving_fallback
from .semantic_count_repair_cli import (
    COUNT_REPAIR_PROMPT,
    _load_rows,
    _load_targets,
    _missing_required_facts,
)
from .timeline_multi_cli import _sha256_file, description_word_limit, validate_description

API_ID_FIELDS = (
    "api_generation_response_id",
    "api_judge_response_id",
    "api_repair_generation_response_id",
    "api_repair_judge_response_id",
)


def _canonical_object_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_expert_rejections(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = str(row["request_id"])
            if request_id in rows or row.get("verdict") not in {"major", "minor"}:
                raise ValueError(f"invalid expert rejection at {path}:{line_number}")
            if row.get("resolution") != "deterministic_source_preserving_template":
                raise ValueError(f"unsupported expert resolution at {path}:{line_number}")
            if not str(row.get("reason", "")).strip():
                raise ValueError(f"expert rejection lacks evidence at {path}:{line_number}")
            rows[request_id] = row
    if not rows:
        raise ValueError("expert second-review ledger is empty")
    return rows


def _receipt_index(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get("request_kind", "")).startswith(
                "semantic_count_remediation_"
            ):
                continue
            digest = str(row.get("batch_sha256", ""))
            if not digest or digest in rows:
                raise ValueError(f"duplicate or empty receipt batch hash at line {line_number}")
            rows[digest] = row
    return rows


def _reconstruct_count_response_ids(
    *,
    request_rows: list[dict],
    requests_by_id: dict[str, dict],
    source_by_id: dict[str, dict],
    repaired_by_id: dict[str, dict],
    target_specs: dict[str, dict],
    receipts: dict[str, dict],
    model: str,
    judge_model: str,
    max_completion_tokens: int,
    judge_max_completion_tokens: int,
    temperature: float,
    original_batch_size: int,
) -> dict[str, dict[str, str]]:
    order = [
        str(row["request_id"])
        for row in request_rows
        if str(row["request_id"]) in target_specs
    ]
    intervals = []
    for block_start in range(0, len(order), original_batch_size):
        block = order[block_start : block_start + original_batch_size]
        for left in range(len(block)):
            for right in range(left + 1, len(block) + 1):
                request_ids = block[left:right]
                generation_items = [
                    {
                        "request_id": request_id,
                        "source_texts": requests_by_id[request_id]["source_texts"],
                        "prior_candidate": source_by_id[request_id]["description"],
                        "required_count_facts": target_specs[request_id][
                            "required_count_facts"
                        ],
                        "max_words": description_word_limit(
                            requests_by_id[request_id]["source_texts"]
                        ),
                    }
                    for request_id in request_ids
                ]
                generation_payload = _chat_payload(
                    model,
                    COUNT_REPAIR_PROMPT,
                    generation_items,
                    max_completion_tokens,
                    temperature,
                )
                generation_hash = _canonical_sha256(
                    {
                        "kind": "semantic_count_remediation_generation",
                        "request_ids": request_ids,
                        "payload": generation_payload,
                    }
                )
                judge_items = [
                    {
                        "request_id": request_id,
                        "source_texts": requests_by_id[request_id]["source_texts"],
                        "candidate": repaired_by_id[request_id]["description"],
                    }
                    for request_id in request_ids
                ]
                judge_payload = _chat_payload(
                    judge_model,
                    BATCH_JUDGE_PROMPT,
                    judge_items,
                    judge_max_completion_tokens,
                    0,
                )
                judge_hash = _canonical_sha256(
                    {
                        "kind": "semantic_count_remediation_judge",
                        "request_ids": request_ids,
                        "payload": judge_payload,
                    }
                )
                if generation_hash in receipts and judge_hash in receipts:
                    intervals.append(
                        {
                            "left": block_start + left,
                            "right": block_start + right,
                            "request_ids": request_ids,
                            "generation_response_id": receipts[generation_hash].get(
                                "response_id"
                            ),
                            "judge_response_id": receipts[judge_hash].get("response_id"),
                        }
                    )

    solutions = []

    def cover(position: int, selected: list[dict]) -> None:
        if position == len(order):
            solutions.append(list(selected))
            return
        for interval in intervals:
            if interval["left"] == position:
                cover(interval["right"], [*selected, interval])

    cover(0, [])
    if len(solutions) != 1:
        raise ValueError(
            f"count-repair receipts do not yield one exact target partition: {len(solutions)}"
        )
    result = {}
    for interval in solutions[0]:
        if not interval["generation_response_id"] or not interval["judge_response_id"]:
            raise ValueError("count-repair receipt lacks a response id")
        for request_id in interval["request_ids"]:
            result[request_id] = {
                "generation_response_id": interval["generation_response_id"],
                "judge_response_id": interval["judge_response_id"],
            }
    if set(result) != set(target_specs):
        raise ValueError("reconstructed response-id coverage differs from expert targets")
    return result


def finalize(args) -> dict:
    requests_path = Path(args.requests).expanduser().resolve()
    source_path = Path(args.source_responses).expanduser().resolve()
    repaired_path = Path(args.repaired_responses).expanduser().resolve()
    targets_path = Path(args.targets).expanduser().resolve()
    second_review_path = Path(args.second_review).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output_metadata = output.with_suffix(output.suffix + ".metadata.json")
    output_receipts = output.with_suffix(output.suffix + ".api-receipts.jsonl")
    if any(path.exists() for path in (output, output_metadata, output_receipts)):
        raise FileExistsError("refusing to overwrite a finalized semantic response output")

    request_rows, requests_by_id = _load_rows(requests_path)
    source_rows, source_by_id = _load_rows(source_path)
    repaired_rows, repaired_by_id = _load_rows(repaired_path)
    target_specs, targets_sha256 = _load_targets(targets_path)
    expert_rejections = _load_expert_rejections(second_review_path)
    if not set(expert_rejections) <= set(target_specs):
        raise ValueError("second-review ledger contains a non-target request")
    if set(requests_by_id) != set(source_by_id) or set(source_by_id) != set(repaired_by_id):
        raise ValueError("request/response coverage differs during semantic finalization")
    changed = {
        request_id
        for request_id in source_by_id
        if source_by_id[request_id] != repaired_by_id[request_id]
    }
    if changed != set(target_specs):
        raise ValueError("count repair changed records outside its expert target set")

    repaired_metadata_path = repaired_path.with_suffix(repaired_path.suffix + ".metadata.json")
    repaired_metadata = json.loads(repaired_metadata_path.read_text(encoding="utf-8"))
    if repaired_metadata.get("output", {}).get("sha256") != _sha256_file(repaired_path):
        raise ValueError("count-repaired responses disagree with metadata")
    receipts_path = Path(str(repaired_metadata["api_receipts"]["path"])).expanduser()
    if not receipts_path.is_absolute():
        receipts_path = repaired_path.parent / receipts_path
    if repaired_metadata["api_receipts"]["sha256"] != _sha256_file(receipts_path):
        raise ValueError("count-repair receipt ledger is corrupted")
    receipt_rows = _receipt_index(receipts_path)
    response_ids = _reconstruct_count_response_ids(
        request_rows=request_rows,
        requests_by_id=requests_by_id,
        source_by_id=source_by_id,
        repaired_by_id=repaired_by_id,
        target_specs=target_specs,
        receipts=receipt_rows,
        model=args.model,
        judge_model=args.judge_model,
        max_completion_tokens=args.max_completion_tokens,
        judge_max_completion_tokens=args.judge_max_completion_tokens,
        temperature=args.temperature,
        original_batch_size=args.original_batch_size,
    )

    finalized = {}
    for old_row in repaired_rows:
        request_id = str(old_row["request_id"])
        row = dict(old_row)
        if request_id in target_specs:
            prior_api_ids = {
                field: source_by_id[request_id].get(field) for field in API_ID_FIELDS
            }
            row["api_generation_response_id"] = response_ids[request_id][
                "generation_response_id"
            ]
            row["api_judge_response_id"] = response_ids[request_id]["judge_response_id"]
            row["api_repair_generation_response_id"] = None
            row["api_repair_judge_response_id"] = None
            row["semantic_count_remediation"] = {
                **row.get("semantic_count_remediation", {}),
                "required_count_facts": target_specs[request_id][
                    "required_count_facts"
                ],
                "expert_verdict": target_specs[request_id]["verdict"],
                "generation_response_id": response_ids[request_id][
                    "generation_response_id"
                ],
                "judge_response_id": response_ids[request_id]["judge_response_id"],
                "prior_api_response_ids": prior_api_ids,
            }
        if request_id in expert_rejections:
            prior_description = str(row["description"])
            prior_current_api_ids = {field: row.get(field) for field in API_ID_FIELDS}
            description = _source_preserving_fallback(
                requests_by_id[request_id]["source_texts"]
            )
            validate_description(requests_by_id[request_id]["source_texts"], description)
            missing = _missing_required_facts(
                target_specs[request_id]["required_count_facts"], description
            )
            if missing:
                raise ValueError(
                    f"expert fallback dropped required facts for {request_id}: {missing}"
                )
            for field in API_ID_FIELDS:
                row[field] = None
            row.update(
                {
                    "description": description,
                    "fallback": "deterministic_source_preserving_template",
                    "fallback_reason": "independent second review rejected semantic rewrite",
                    "deterministic_source_preservation": True,
                    "semantic_judge": None,
                    "repair_attempted": True,
                    "repair_succeeded": False,
                    "expert_semantic_remediation": {
                        "prior_description_sha256": hashlib.sha256(
                            prior_description.encode("utf-8")
                        ).hexdigest(),
                        "prior_current_api_response_ids": prior_current_api_ids,
                        "second_review": expert_rejections[request_id],
                        "method": "deterministic_source_preserving_template",
                    },
                }
            )
        finalized[request_id] = row

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = None
    temporary_receipts = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as sink:
            temporary_output = Path(sink.name)
            for source_row in source_rows:
                request_id = str(source_row["request_id"])
                row = finalized[request_id]
                if request_id not in target_specs and row != source_row:
                    raise AssertionError(f"non-target response changed: {request_id}")
                sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        receipt_descriptor, receipt_name = tempfile.mkstemp(
            prefix=output_receipts.name + ".", suffix=".tmp", dir=output.parent
        )
        os.close(receipt_descriptor)
        temporary_receipts = Path(receipt_name)
        shutil.copyfile(receipts_path, temporary_receipts)
        publish_file(temporary_receipts)
        os.replace(temporary_receipts, output_receipts)
        temporary_receipts = None
        publish_file(temporary_output)
        os.replace(temporary_output, output)
        temporary_output = None

        producer_identity = {
            "kind": "composite_targeted_semantic_remediation",
            "base_generation_producer_identity_sha256": repaired_metadata.get(
                "producer_identity_sha256"
            ),
            "duplicate_remediation": repaired_metadata.get("remediation"),
            "count_remediation": {
                "provider": "openai_compatible",
                "base_url": repaired_metadata.get("base_url"),
                "model": args.model,
                "judge_model": args.judge_model,
                "revision": repaired_metadata.get("revision"),
                "generation_prompt_sha256": hashlib.sha256(
                    COUNT_REPAIR_PROMPT.encode("utf-8")
                ).hexdigest(),
                "judge_prompt_sha256": hashlib.sha256(
                    BATCH_JUDGE_PROMPT.encode("utf-8")
                ).hexdigest(),
                "targets_sha256": targets_sha256,
                "temperature": args.temperature,
                "source_responses_sha256": _sha256_file(source_path),
                "count_repaired_responses_sha256": _sha256_file(repaired_path),
            },
            "expert_fallback": {
                "second_review_sha256": _sha256_file(second_review_path),
                "rows": len(expert_rejections),
                "method": "deterministic_source_preserving_template",
            },
        }
        metadata = dict(repaired_metadata)
        quality = {
            "invalid_published": sum(
                bool(row.get("error")) or not bool(row.get("description"))
                for row in finalized.values()
            ),
            "repairs_attempted": sum(
                bool(row.get("repair_attempted")) for row in finalized.values()
            ),
            "repairs_succeeded": sum(
                bool(row.get("repair_succeeded")) for row in finalized.values()
            ),
            "semantic_fallbacks": sum(
                row.get("fallback") == "deterministic_source_preserving_template"
                for row in finalized.values()
            ),
        }
        metadata.update(
            {
                "schema_version": 4,
                "generator": "kimodo.training.semantic_response_finalize_cli",
                "producer_identity": producer_identity,
                "producer_identity_sha256": _canonical_object_sha256(
                    producer_identity
                ),
                "quality": quality,
                "api_receipts": {
                    "path": output_receipts.name,
                    "sha256": _sha256_file(output_receipts),
                    **_receipt_summary(output_receipts),
                },
                "output": {
                    "path": output.name,
                    "sha256": _sha256_file(output),
                    "entries": len(source_rows),
                },
                "semantic_finalization": {
                    "source_responses_sha256": _sha256_file(source_path),
                    "count_repaired_responses_sha256": _sha256_file(repaired_path),
                    "count_repaired_metadata_sha256": _sha256_file(
                        repaired_metadata_path
                    ),
                    "targets_sha256": targets_sha256,
                    "second_review_sha256": _sha256_file(second_review_path),
                    "reconstructed_count_response_id_rows": len(target_specs),
                    "api_bound_rows": len(target_specs) - len(expert_rejections),
                    "expert_fallback_rows": len(expert_rejections),
                    "remaining_expert_required_count_facts": 0,
                    "requests_sha256": _sha256_file(requests_path),
                },
                "requests": {
                    "path": os.path.relpath(requests_path, output.parent),
                    "sha256": _sha256_file(requests_path),
                },
            }
        )
        expert_targets = metadata.get("semantic_count_remediation", {}).get(
            "expert_targets"
        )
        if isinstance(expert_targets, dict):
            expert_targets["path"] = os.path.relpath(targets_path, output.parent)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as sink:
            temporary_metadata = Path(sink.name)
            json.dump(metadata, sink, indent=2, sort_keys=True)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        publish_file(temporary_metadata)
        os.replace(temporary_metadata, output_metadata)
        return metadata["semantic_finalization"]
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        if temporary_receipts is not None:
            temporary_receipts.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--source-responses", required=True)
    parser.add_argument("--repaired-responses", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--second-review", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="mimo-v2.5-pro")
    parser.add_argument("--judge-model", default="mimo-v2.5-pro")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--judge-max-completion-tokens", type=int, default=1024)
    parser.add_argument("--original-batch-size", type=int, default=8)
    return parser


def main() -> None:
    result = finalize(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
