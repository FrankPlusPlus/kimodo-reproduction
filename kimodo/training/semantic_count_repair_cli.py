# SPDX-License-Identifier: Apache-2.0
"""Target and API-repair only V2 captions that dropped an explicit source count."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .file_permissions import publish_file
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
from .llm_quality_cli import COUNT_UNITS, _count_signatures, missing_explicit_count_groups
from .timeline_multi_cli import _sha256_file, description_word_limit, validate_description

COUNT_REPAIR_PROMPT = """Rewrite each motion caption so it preserves every ordered source action
and every explicitly listed required_count_fact. The prior candidate dropped those exact quantitative
facts. Preserve chronology, directions, body parts, objects, interactions, and genuine repetitions;
do not invent anything and do not convert a cardinal fact such as two steps into an unrelated twice.
Use the source wording when needed to keep each count attached to the correct action. Obey max_words.
Return strict JSON only:
{"items":[{"request_id":"unchanged id","description":"corrected description"}]}.
Return every input request_id exactly once and no additional request_id."""


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


def _load_targets(path: Path) -> tuple[dict[str, dict], str]:
    targets = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = str(row["request_id"])
            facts = row.get("required_count_facts")
            if request_id in targets:
                raise ValueError(f"duplicate target id at {path}:{line_number}")
            if row.get("verdict") not in {"major", "minor"}:
                raise ValueError(f"invalid expert verdict at {path}:{line_number}")
            valid_units = set(COUNT_UNITS.values())
            valid_facts = isinstance(facts, list) and bool(facts)
            if valid_facts:
                for fact in facts:
                    try:
                        value_text, unit = fact.split(":", 1)
                        valid_facts = (
                            str(int(value_text)) == value_text
                            and 1 <= int(value_text) <= 5
                            and unit in valid_units
                        )
                    except (AttributeError, TypeError, ValueError):
                        valid_facts = False
                    if not valid_facts:
                        break
            if not valid_facts:
                raise ValueError(f"invalid required_count_facts at {path}:{line_number}")
            targets[request_id] = row
    if not targets:
        raise ValueError(f"expert target file is empty: {path}")
    return targets, _sha256_file(path)


def _missing_required_facts(required_facts: list[str], description: str) -> list[str]:
    observed = _count_signatures(description)
    missing = []
    for fact in required_facts:
        value_text, unit = fact.split(":", 1)
        value = int(value_text)
        if (value, unit) in observed:
            continue
        missing.append(fact)
    return missing


def _validated_record(
    request: dict,
    old_record: dict,
    description: str,
    verdict: dict,
    required_facts: list[str],
) -> dict:
    description = " ".join(description.split())
    validate_description(request["source_texts"], description)
    missing = _missing_required_facts(required_facts, description)
    if missing:
        raise ValueError(f"count remediation still drops explicit groups: {missing}")
    if verdict.get("accepted") is not True:
        raise ValueError(f"semantic judge rejected count remediation: {verdict.get('reason')}")
    record = dict(old_record)
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
            "description": description,
            "semantic_judge": verdict,
            "repair_attempted": True,
            "repair_succeeded": True,
            "semantic_count_remediation": {
                "prior_description_sha256": hashlib.sha256(
                    str(old_record["description"]).encode("utf-8")
                ).hexdigest(),
                "method": "targeted_api_rewrite_and_semantic_rejudge",
            },
        }
    )
    return record


def _repair_batch(
    rows: list[dict],
    old_by_id: dict[str, dict],
    target_specs: dict[str, dict],
    client: _ApiClient,
    args,
) -> list[dict]:
    request_ids = [str(row["request_id"]) for row in rows]
    items = [
        {
            "request_id": row["request_id"],
            "source_texts": row["source_texts"],
            "prior_candidate": old_by_id[str(row["request_id"])]["description"],
            "required_count_facts": target_specs[str(row["request_id"])][
                "required_count_facts"
            ],
            "max_words": description_word_limit(row["source_texts"]),
        }
        for row in rows
    ]
    try:
        generation = client.post(
            _chat_payload(
                args.model,
                COUNT_REPAIR_PROMPT,
                items,
                args.max_completion_tokens,
                args.temperature,
            ),
            request_kind="semantic_count_remediation_generation",
            request_ids=request_ids,
        )
        candidates = _strict_items(_content(generation), request_ids, {"description": str})
        judge_items = [
            {
                "request_id": row["request_id"],
                "source_texts": row["source_texts"],
                "candidate": candidate["description"],
            }
            for row, candidate in zip(rows, candidates, strict=True)
        ]
        # Check deterministic invariants before paying for the semantic judge.
        for row, candidate in zip(rows, candidates, strict=True):
            validate_description(row["source_texts"], candidate["description"])
            missing = _missing_required_facts(
                target_specs[str(row["request_id"])]["required_count_facts"],
                candidate["description"],
            )
            if missing:
                raise ValueError(f"generated repair still drops explicit groups: {missing}")
        judged = client.post(
            _chat_payload(
                args.judge_model,
                BATCH_JUDGE_PROMPT,
                judge_items,
                args.judge_max_completion_tokens,
                0,
            ),
            request_kind="semantic_count_remediation_judge",
            request_ids=request_ids,
        )
        verdicts = _strict_items(
            _content(judged), request_ids, {"accepted": bool, "reason": str}
        )
        return [
            _validated_record(
                row,
                old_by_id[str(row["request_id"])],
                candidate["description"],
                verdict,
                target_specs[str(row["request_id"])]["required_count_facts"],
            )
            for row, candidate, verdict in zip(rows, candidates, verdicts, strict=True)
        ]
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        if len(rows) > 1:
            midpoint = len(rows) // 2
            print(
                f"count repair batch {len(rows)} failed closed; splitting: {error}",
                flush=True,
            )
            return _repair_batch(
                rows[:midpoint], old_by_id, target_specs, client, args
            ) + _repair_batch(
                rows[midpoint:], old_by_id, target_specs, client, args
            )
        # The existing resilient path includes generation, semantic rejudge, and a
        # deterministic source-preserving fallback. The explicit-count invariant is
        # still checked below, so this cannot silently publish the original defect.
        print(
            f"count repair uses resilient singleton path for {request_ids[0]}: {error}",
            flush=True,
        )
        record = _process_resilient(rows, client, args)[0]
        required_facts = target_specs[request_ids[0]]["required_count_facts"]
        missing = _missing_required_facts(required_facts, record["description"])
        if missing:
            raise ValueError(
                f"resilient count repair still drops explicit groups for {request_ids[0]}: {missing}"
            )
        record["semantic_count_remediation"] = {
            "prior_description_sha256": hashlib.sha256(
                str(old_by_id[request_ids[0]]["description"]).encode("utf-8")
            ).hexdigest(),
            "method": "resilient_generation_or_source_preserving_fallback",
        }
        return [record]


def repair(args) -> dict:
    requests_path = Path(args.requests).expanduser().resolve()
    responses_path = Path(args.responses).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output_metadata = output.with_suffix(output.suffix + ".metadata.json")
    output_receipts = output.with_suffix(output.suffix + ".api-receipts.jsonl")
    if any(path.exists() for path in (output, output_metadata, output_receipts)):
        raise FileExistsError("refusing to overwrite a semantic-count remediation output")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"{args.api_key_env} is required")

    request_rows, requests_by_id = _load_rows(requests_path)
    response_rows, responses_by_id = _load_rows(responses_path)
    if set(requests_by_id) != set(responses_by_id):
        raise ValueError("request/response coverage differs before count remediation")
    source_metadata_path = responses_path.with_suffix(responses_path.suffix + ".metadata.json")
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("output", {}).get("sha256") != _sha256_file(responses_path):
        raise ValueError("source responses disagree with metadata")
    if source_metadata.get("requests", {}).get("sha256") != _sha256_file(requests_path):
        raise ValueError("source responses belong to different requests")
    source_receipts = Path(str(source_metadata["api_receipts"]["path"])).expanduser()
    if not source_receipts.is_absolute():
        source_receipts = responses_path.parent / source_receipts
    if source_metadata["api_receipts"]["sha256"] != _sha256_file(source_receipts):
        raise ValueError("source API receipt ledger is corrupted")

    targets_path = Path(args.targets).expanduser().resolve() if args.targets else None
    targets_sha256 = None
    if targets_path is not None:
        target_specs, targets_sha256 = _load_targets(targets_path)
        unknown = set(target_specs) - set(requests_by_id)
        if unknown:
            raise ValueError(f"expert target file contains {len(unknown)} unknown request ids")
        for request_id, spec in target_specs.items():
            source_facts = set().union(
                *(
                    _count_signatures(str(text))
                    for text in requests_by_id[request_id]["source_texts"]
                )
            )
            required_facts = {
                (int(fact.split(":", 1)[0]), fact.split(":", 1)[1])
                for fact in spec["required_count_facts"]
            }
            if not required_facts <= source_facts:
                raise ValueError(
                    f"expert target requires facts absent from source_texts: {request_id}: "
                    f"{sorted(required_facts - source_facts)}"
                )
            missing = _missing_required_facts(
                spec["required_count_facts"], responses_by_id[request_id]["description"]
            )
            if not missing:
                raise ValueError(
                    f"expert target is stale or already repaired: {request_id}"
                )
        targets = [
            row for row in request_rows if str(row["request_id"]) in target_specs
        ]
    else:
        target_specs = {}
        targets = []
        for row in request_rows:
            request_id = str(row["request_id"])
            missing = missing_explicit_count_groups(
                row["source_texts"], responses_by_id[request_id]["description"]
            )
            if missing:
                targets.append(row)
                target_specs[request_id] = {
                    "request_id": request_id,
                    "verdict": "deterministic",
                    "required_count_facts": missing,
                }
    target_ids = set(target_specs)
    if not targets:
        raise ValueError("source responses contain no explicit-count omissions")

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
                    lambda batch: _repair_batch(
                        batch, responses_by_id, target_specs, client, args
                    ),
                    batches,
                ):
                    for record in records:
                        repaired[str(record["request_id"])] = record
                    print(f"Count remediation: {len(repaired)}/{len(targets)}", flush=True)
        if set(repaired) != target_ids:
            raise RuntimeError("count remediation coverage is incomplete")

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as sink:
            temporary_output = Path(sink.name)
            for old_record in response_rows:
                request_id = str(old_record["request_id"])
                record = repaired.get(request_id, old_record)
                if request_id in target_ids:
                    missing = _missing_required_facts(
                        target_specs[request_id]["required_count_facts"],
                        record["description"],
                    )
                    if missing:
                        raise ValueError(
                            f"count remediation output remains invalid for {request_id}: {missing}"
                        )
                elif record != old_record:
                    raise AssertionError(f"non-target response changed: {request_id}")
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            sink.flush()
            os.fsync(sink.fileno())

        publish_file(temporary_receipts)
        os.replace(temporary_receipts, output_receipts)
        publish_file(temporary_output)
        os.replace(temporary_output, output)
        temporary_output = None
        metadata = dict(source_metadata)
        metadata.update(
            {
                "schema_version": 3,
                "generator": "kimodo.training.semantic_count_repair_cli",
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
                "semantic_count_remediation": {
                    "source_responses_sha256": _sha256_file(responses_path),
                    "source_metadata_sha256": _sha256_file(source_metadata_path),
                    "targeted_requests": len(target_ids),
                    "remaining_expert_required_count_facts": 0,
                    "requests_sha256": _sha256_file(requests_path),
                    "expert_targets": (
                        {
                            "path": str(targets_path),
                            "sha256": targets_sha256,
                        }
                        if targets_path is not None
                        else None
                    ),
                    "prompt_sha256": hashlib.sha256(
                        COUNT_REPAIR_PROMPT.encode("utf-8")
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
        return metadata["semantic_count_remediation"]
    finally:
        temporary_receipts.unlink(missing_ok=True)
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--targets")
    parser.add_argument("--base-url", default=os.environ.get("PRODUCT_GRAPH_LLM_BASE_URL"))
    parser.add_argument("--api-key-env", default="PRODUCT_GRAPH_LLM_API_KEY")
    parser.add_argument(
        "--model", default=os.environ.get("PRODUCT_GRAPH_LLM_MODEL", "mimo-v2.5-pro")
    )
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--revision", default="provider-managed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.25)
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
