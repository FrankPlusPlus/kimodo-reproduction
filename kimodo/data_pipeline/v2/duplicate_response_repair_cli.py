# SPDX-License-Identifier: Apache-2.0
"""Targeted, API-audited repair of exact duplicate V2 response descriptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kimodo.common.file_permissions import publish_file

from .llm_api_augmentation_cli import (
    BATCH_JUDGE_PROMPT,
    _ApiClient,
    _chat_payload,
    _content,
    _normalize_base_url,
    _process_resilient,
    _receipt_summary,
    _strict_items,
)
from .timeline_multi_cli import _sha256_file, description_word_limit, validate_description

WORDS = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
DUPLICATE_REPAIR_PROMPT = """Rewrite every duplicate motion caption into a semantically exact but
lexically distinct description. Preserve every ordered source action, chronology, direction, body
part, object, interaction, and genuine repetition. Do not invent anything. The prior candidate is
forbidden: the new description must not normalize to the same words, and should retain source detail
that safely distinguishes this timeline. Obey max_words. Return strict JSON only:
{"items":[{"request_id":"unchanged id","description":"new description"}]}.
Return every input request_id exactly once and no additional request_id."""


def _normalized(value: str) -> str:
    return " ".join(WORDS.findall(value.lower()))


def _load_rows(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    by_id = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = str(row["request_id"])
            if request_id in by_id:
                raise ValueError(f"duplicate request id at {path}:{line_number}")
            rows.append(row)
            by_id[request_id] = row
    return rows, by_id


def _plan_weights(path: Path, request_ids: set[str]) -> Counter:
    weights = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = str(row.get("llm_request_id", row.get("qwen_request_id", "")))
            if request_id not in request_ids:
                raise ValueError(f"plan line {line_number} has an unknown request id")
            weights[request_id] += 1
    if set(weights) != request_ids:
        raise ValueError("timeline plan does not cover every request")
    return weights


def _repair_batch(rows: list[dict], old_by_id: dict[str, dict], client: _ApiClient, args) -> list[dict]:
    if not rows:
        return []
    request_ids = [str(row["request_id"]) for row in rows]
    items = [
        {
            "request_id": row["request_id"],
            "source_texts": row["source_texts"],
            "prior_duplicate": old_by_id[str(row["request_id"])]["description"],
            "max_words": description_word_limit(row["source_texts"]),
        }
        for row in rows
    ]
    try:
        generation = client.post(
            _chat_payload(
                args.model,
                DUPLICATE_REPAIR_PROMPT,
                items,
                args.max_completion_tokens,
                args.temperature,
            ),
            request_kind="duplicate_remediation_generation",
            request_ids=request_ids,
        )
        candidates = _strict_items(_content(generation), request_ids, {"description": str})
        valid_rows = []
        valid_candidates = []
        fallback_rows = []
        for row, candidate in zip(rows, candidates, strict=True):
            description = " ".join(candidate["description"].split())
            old_description = str(old_by_id[str(row["request_id"])]["description"])
            try:
                validate_description(row["source_texts"], description)
                if _normalized(description) == _normalized(old_description):
                    raise ValueError("remediation repeated its forbidden duplicate")
            except ValueError:
                fallback_rows.append(row)
            else:
                candidate["description"] = description
                valid_rows.append(row)
                valid_candidates.append(candidate)

        accepted = {}
        if valid_rows:
            judge_ids = [str(row["request_id"]) for row in valid_rows]
            judge_items = [
                {
                    "request_id": row["request_id"],
                    "source_texts": row["source_texts"],
                    "candidate": candidate["description"],
                }
                for row, candidate in zip(valid_rows, valid_candidates, strict=True)
            ]
            judged = client.post(
                _chat_payload(
                    args.judge_model,
                    BATCH_JUDGE_PROMPT,
                    judge_items,
                    args.judge_max_completion_tokens,
                    0,
                ),
                request_kind="duplicate_remediation_judge",
                request_ids=judge_ids,
            )
            verdicts = _strict_items(
                _content(judged), judge_ids, {"accepted": bool, "reason": str}
            )
            for row, candidate, verdict in zip(
                valid_rows, valid_candidates, verdicts, strict=True
            ):
                if verdict["accepted"]:
                    request_id = str(row["request_id"])
                    record = dict(old_by_id[request_id])
                    for key in (
                        "fallback",
                        "fallback_reason",
                        "deterministic_source_preservation",
                        "initial_rejected_candidate",
                        "initial_semantic_judge",
                        "repair_error",
                    ):
                        record.pop(key, None)
                    record.update(
                        {
                            "description": candidate["description"],
                            "semantic_judge": verdict,
                            "repair_attempted": True,
                            "repair_succeeded": True,
                            "duplicate_remediation": {
                                "prior_description_sha256": hashlib.sha256(
                                    str(old_by_id[request_id]["description"]).encode("utf-8")
                                ).hexdigest(),
                                "method": "targeted_api_rewrite_and_semantic_rejudge",
                            },
                        }
                    )
                    accepted[request_id] = record
                else:
                    fallback_rows.append(row)
        if fallback_rows:
            # Reuse the full resilient generation/repair path for rare failed remediations.
            for record in _process_resilient(fallback_rows, client, args):
                accepted[str(record["request_id"])] = record
        return [accepted[request_id] for request_id in request_ids]
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        if len(rows) == 1:
            print(
                f"duplicate repair fell back to standard resilient generation for {request_ids[0]}: {error}",
                flush=True,
            )
            return _process_resilient(rows, client, args)
        midpoint = len(rows) // 2
        print(f"duplicate repair batch {len(rows)} failed closed; splitting: {error}", flush=True)
        return _repair_batch(rows[:midpoint], old_by_id, client, args) + _repair_batch(
            rows[midpoint:], old_by_id, client, args
        )


def repair(args) -> dict:
    requests_path = Path(args.requests).expanduser().resolve()
    responses_path = Path(args.responses).expanduser().resolve()
    plan_path = Path(args.plan).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output_metadata = output.with_suffix(output.suffix + ".metadata.json")
    output_receipts = output.with_suffix(output.suffix + ".api-receipts.jsonl")
    if any(path.exists() for path in (output, output_metadata, output_receipts)):
        raise FileExistsError("refusing to overwrite a duplicate-remediation output")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"{args.api_key_env} is required")

    request_rows, requests_by_id = _load_rows(requests_path)
    response_rows, responses_by_id = _load_rows(responses_path)
    if set(requests_by_id) != set(responses_by_id):
        raise ValueError("request/response coverage differs before duplicate remediation")
    source_metadata_path = responses_path.with_suffix(responses_path.suffix + ".metadata.json")
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("output", {}).get("sha256") != _sha256_file(responses_path):
        raise ValueError("source responses disagree with metadata")
    source_receipts = Path(str(source_metadata["api_receipts"]["path"])).expanduser()
    if not source_receipts.is_absolute():
        source_receipts = responses_path.parent / source_receipts
    if source_metadata["api_receipts"]["sha256"] != _sha256_file(source_receipts):
        raise ValueError("source API receipt ledger is corrupted")

    groups = defaultdict(list)
    for row in response_rows:
        groups[_normalized(str(row["description"]))].append(str(row["request_id"]))
    duplicate_groups = {key: ids for key, ids in groups.items() if key and len(ids) > 1}
    weights = _plan_weights(plan_path, set(requests_by_id))
    target_ids = set()
    keepers = {}
    for description, request_ids in duplicate_groups.items():
        keeper = max(request_ids, key=lambda request_id: (weights[request_id], request_id))
        keepers[description] = keeper
        target_ids.update(set(request_ids) - {keeper})
    targets = [row for row in request_rows if str(row["request_id"]) in target_ids]
    if not targets:
        raise ValueError("source responses contain no exact duplicate groups")

    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_descriptor, receipt_name = tempfile.mkstemp(
        prefix=output_receipts.name + ".", suffix=".tmp", dir=output.parent
    )
    os.close(receipt_descriptor)
    temporary_receipts = Path(receipt_name)
    temporary_output = None
    try:
        shutil.copyfile(source_receipts, temporary_receipts)
        batches = [
            targets[offset : offset + args.batch_size]
            for offset in range(0, len(targets), args.batch_size)
        ]
        repaired = {}
        with temporary_receipts.open("a", encoding="utf-8") as receipt_sink:
            client = _ApiClient(
                base_url=_normalize_base_url(args.base_url),
                api_key=api_key,
                requests_per_minute=args.requests_per_minute,
                timeout=args.timeout,
                max_retries=args.max_retries,
                receipts=receipt_sink,
            )
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                for records in executor.map(
                    lambda rows: _repair_batch(rows, responses_by_id, client, args), batches
                ):
                    for row in records:
                        repaired[str(row["request_id"])] = row
                    print(f"Duplicate remediation: {len(repaired)}/{len(targets)}", flush=True)
        if set(repaired) != target_ids:
            raise RuntimeError("duplicate remediation coverage is incomplete")

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as sink:
            temporary_output = Path(sink.name)
            for row in response_rows:
                sink.write(
                    json.dumps(repaired.get(str(row["request_id"]), row), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
            sink.flush()
            os.fsync(sink.fileno())

        # A successful remediation must strictly improve exact duplicate rows.
        new_groups = defaultdict(int)
        with temporary_output.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    new_groups[_normalized(str(json.loads(line)["description"]))] += 1
        duplicate_rows_before = sum(len(ids) for ids in duplicate_groups.values())
        duplicate_rows_after = sum(count for count in new_groups.values() if count > 1)
        if duplicate_rows_after >= duplicate_rows_before:
            raise ValueError("targeted remediation did not reduce exact duplicate rows")

        publish_file(temporary_receipts)
        os.replace(temporary_receipts, output_receipts)
        publish_file(temporary_output)
        os.replace(temporary_output, output)
        temporary_output = None
        metadata = dict(source_metadata)
        metadata.update(
            {
                "schema_version": 3,
                "generator": "kimodo.training.duplicate_response_repair_cli",
                "api_receipts": {
                    "path": str(output_receipts),
                    "sha256": _sha256_file(output_receipts),
                    **_receipt_summary(output_receipts),
                },
                "output": {
                    "path": str(output),
                    "sha256": _sha256_file(output),
                    "entries": len(response_rows),
                },
                "remediation": {
                    "source_responses_sha256": _sha256_file(responses_path),
                    "source_metadata_sha256": _sha256_file(source_metadata_path),
                    "timeline_plan_sha256": _sha256_file(plan_path),
                    "duplicate_groups_before": len(duplicate_groups),
                    "duplicate_rows_before": duplicate_rows_before,
                    "targeted_requests": len(target_ids),
                    "duplicate_rows_after": duplicate_rows_after,
                    "prompt_sha256": hashlib.sha256(
                        DUPLICATE_REPAIR_PROMPT.encode("utf-8")
                    ).hexdigest(),
                },
            }
        )
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as sink:
            metadata_temporary = Path(sink.name)
            json.dump(metadata, sink, indent=2, sort_keys=True)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        publish_file(metadata_temporary)
        os.replace(metadata_temporary, output_metadata)
        return metadata["remediation"]
    finally:
        temporary_receipts.unlink(missing_ok=True)
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=os.environ.get("PRODUCT_GRAPH_LLM_BASE_URL"))
    parser.add_argument("--api-key-env", default="PRODUCT_GRAPH_LLM_API_KEY")
    parser.add_argument("--model", default=os.environ.get("PRODUCT_GRAPH_LLM_MODEL", "mimo-v2.5-pro"))
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--revision", default="provider-managed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--requests-per-minute", type=float, default=90.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--judge-max-completion-tokens", type=int, default=1024)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.judge_model = args.judge_model or args.model
    if not args.base_url:
        raise ValueError("--base-url or PRODUCT_GRAPH_LLM_BASE_URL is required")
    result = repair(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
