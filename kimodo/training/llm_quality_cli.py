# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit LLM-generated V2 descriptions and create a deterministic review sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from .timeline_multi_cli import _sha256_file, validate_description

WORDS = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p05": None, "p50": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)

    return {
        "min": ordered[0],
        "p05": percentile(0.05),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _load_requests(path: Path) -> tuple[dict[str, dict], dict]:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("output", {}).get("sha256") != _sha256_file(path):
        raise ValueError("Request manifest hash disagrees with its metadata")
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = str(row["request_id"])
            if request_id in rows:
                raise ValueError(f"Duplicate request id at line {line_number}: {request_id}")
            rows[request_id] = row
    return rows, metadata


def _load_responses(paths: list[str], requests_sha256: str) -> tuple[dict[str, dict], list[dict]]:
    rows = {}
    sources = []
    producer_identities = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        metadata_path = path.with_suffix(path.suffix + ".metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("output", {}).get("sha256") != _sha256_file(path):
            raise ValueError(f"Response hash disagrees with metadata: {path}")
        if metadata.get("requests", {}).get("sha256") != requests_sha256:
            raise ValueError(f"Response was generated from different requests: {path}")
        producer_identity = metadata.get("producer_identity_sha256")
        if not producer_identity:
            raise ValueError(f"Response lacks a remote producer identity: {path}")
        producer_identities.add(producer_identity)
        receipts = metadata.get("api_receipts", {})
        receipts_path = Path(str(receipts.get("path", ""))).expanduser()
        if not receipts_path.is_absolute():
            receipts_path = path.parent / receipts_path
        if not receipts_path.is_file() or receipts.get("sha256") != _sha256_file(receipts_path):
            raise ValueError(f"API receipt ledger is missing or corrupted: {path}")
        sources.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "metadata_path": str(metadata_path),
                "metadata_sha256": _sha256_file(metadata_path),
                "producer_identity_sha256": producer_identity,
                "model": metadata.get("model"),
                "judge_model": metadata.get("judge_model"),
                "revision": metadata.get("revision"),
                "usage": receipts.get("usage", {}),
            }
        )
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                request_id = str(row["request_id"])
                if request_id in rows:
                    raise ValueError(f"Duplicate response id at {path}:{line_number}: {request_id}")
                rows[request_id] = row
    if len(producer_identities) != 1:
        raise ValueError("Response shards use inconsistent producer identities")
    return rows, sources


def _usage_and_cost(sources: list[dict], input_price: float, output_price: float) -> dict:
    usage = Counter()
    for source in sources:
        usage.update({key: int(value) for key, value in source.get("usage", {}).items()})
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    return {
        "raw_usage": dict(sorted(usage.items())),
        "normalized_input_tokens": input_tokens,
        "normalized_output_tokens": output_tokens,
        "price_assumption_cny_per_million": {"input": input_price, "output": output_price},
        "estimated_cost_cny": input_tokens / 1_000_000 * input_price + output_tokens / 1_000_000 * output_price,
    }


def audit(args) -> dict:
    requests_path = Path(args.requests).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    sample_path = Path(args.review_sample).expanduser().resolve()
    if report_path.exists() or sample_path.exists():
        raise FileExistsError("Refusing to overwrite an existing quality report or review sample")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    requests, requests_metadata = _load_requests(requests_path)
    responses, response_sources = _load_responses(args.responses, _sha256_file(requests_path))
    missing = sorted(requests.keys() - responses.keys())
    unexpected = sorted(responses.keys() - requests.keys())
    if (missing and not args.allow_partial) or unexpected:
        raise ValueError(f"Response coverage mismatch: missing={len(missing)}, unexpected={len(unexpected)}")

    counts = Counter()
    word_counts: list[float] = []
    lexical_recall: list[float] = []
    copy_similarity: list[float] = []
    judge_reasons = Counter()
    description_ids: dict[str, list[str]] = defaultdict(list)
    review_rows = []
    invalid_rows = []
    for request_id in sorted(responses):
        request = requests[request_id]
        response = responses[request_id]
        description = " ".join(str(response.get("description", "")).split())
        flags = []
        try:
            validate_description(request["source_texts"], description)
        except ValueError as error:
            flags.append(f"validator:{error}")
            invalid_rows.append({"request_id": request_id, "error": str(error)})
        words = WORDS.findall(description.lower())
        source_words = set(WORDS.findall(" ".join(request["source_texts"]).lower()))
        source_sequence = WORDS.findall(" ".join(request["source_texts"]).lower())
        output_words = set(words)
        recall = len(source_words & output_words) / max(1, len(source_words))
        word_counts.append(float(len(words)))
        lexical_recall.append(recall)
        similarity = SequenceMatcher(None, source_sequence, words, autojunk=False).ratio()
        copy_similarity.append(similarity)
        event_count = len(request["source_texts"])
        counts[f"event_count/{event_count}"] += 1
        fallback = response.get("fallback") == "deterministic_source_preserving_template"
        if fallback:
            counts["fallback"] += 1
            counts[f"fallback/event_count/{event_count}"] += 1
            flags.append("fallback")
        if response.get("repair_attempted"):
            counts["repair_attempted"] += 1
        if response.get("repair_succeeded"):
            counts["repair_succeeded"] += 1
        judge = response.get("semantic_judge")
        if isinstance(judge, dict):
            if judge.get("accepted") is True:
                counts["judge_accepted"] += 1
            else:
                counts["judge_rejected"] += 1
            reason = " ".join(str(judge.get("reason", "")).lower().split())
            if reason:
                judge_reasons[reason] += 1
        else:
            counts["judge_missing"] += 1
        normalized = " ".join(words)
        description_ids[normalized].append(request_id)
        if recall < args.low_lexical_recall:
            flags.append("low_lexical_recall")
        if similarity >= args.high_copy_similarity:
            flags.append("high_source_copy")
            counts["high_source_copy"] += 1
            counts[f"high_source_copy/event_count/{event_count}"] += 1
        review_rows.append(
            {
                "request_id": request_id,
                "event_count": event_count,
                "source_texts": request["source_texts"],
                "description": description,
                "semantic_judge": judge,
                "fallback": response.get("fallback"),
                "initial_rejected_candidate": response.get("initial_rejected_candidate"),
                "initial_semantic_judge": response.get("initial_semantic_judge"),
                "repair_attempted": bool(response.get("repair_attempted")),
                "repair_succeeded": bool(response.get("repair_succeeded")),
                "repair_error": response.get("repair_error"),
                "word_count": len(words),
                "source_lexical_recall": recall,
                "source_copy_similarity": similarity,
                "risk_flags": flags,
                "human_verdict": None,
                "human_notes": "",
            }
        )

    duplicate_groups = {
        description: ids for description, ids in description_ids.items() if description and len(ids) > 1
    }
    duplicate_rows = sum(len(ids) for ids in duplicate_groups.values())
    audited = len(responses)
    fallback_rate = counts["fallback"] / max(1, audited)
    duplicate_rate = duplicate_rows / max(1, audited)
    high_copy_rate = counts["high_source_copy"] / max(1, audited)
    gates = {
        "coverage_complete": not missing and not unexpected,
        "validator_invalid_zero": not invalid_rows,
        "fallback_rate": {
            "value": fallback_rate,
            "maximum": args.max_fallback_rate,
            "passed": fallback_rate <= args.max_fallback_rate,
        },
        "exact_duplicate_row_rate": {
            "value": duplicate_rate,
            "maximum": args.max_duplicate_rate,
            "passed": duplicate_rate <= args.max_duplicate_rate,
        },
        "high_source_copy_rate": {
            "value": high_copy_rate,
            "maximum": args.max_high_copy_rate,
            "similarity_threshold": args.high_copy_similarity,
            "passed": high_copy_rate <= args.max_high_copy_rate,
        },
    }
    eligible = (
        gates["coverage_complete"]
        and gates["validator_invalid_zero"]
        and gates["fallback_rate"]["passed"]
        and gates["exact_duplicate_row_rate"]["passed"]
        and gates["high_source_copy_rate"]["passed"]
    )

    # Risk rows are always sampled first; the remainder is deterministic and stratified by event count.
    selected = []
    selected_ids = set()
    for row in sorted(
        (row for row in review_rows if row["risk_flags"]),
        key=lambda row: (hashlib.sha256(row["request_id"].encode()).hexdigest(), row["request_id"]),
    )[: args.max_risk_samples]:
        selected.append(row)
        selected_ids.add(row["request_id"])
    for event_count in range(2, 6):
        candidates = [
            row for row in review_rows if row["event_count"] == event_count and row["request_id"] not in selected_ids
        ]
        candidates.sort(key=lambda row: hashlib.sha256(f"{args.sample_seed}:{row['request_id']}".encode()).hexdigest())
        for row in candidates[: args.sample_per_event_count]:
            selected.append(row)
            selected_ids.add(row["request_id"])

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=sample_path.parent, delete=False) as output:
        sample_temporary = Path(output.name)
        for row in selected:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(sample_temporary, sample_path)

    report = {
        "schema_version": 1,
        "auditor": "kimodo.training.llm_quality_cli",
        "quality_gate": {"eligible": eligible, **gates},
        "coverage": {
            "requests": len(requests),
            "responses": len(responses),
            "missing": len(missing),
            "unexpected": len(unexpected),
        },
        "counts": dict(sorted(counts.items())),
        "word_count": _quantiles(word_counts),
        "source_lexical_recall": _quantiles(lexical_recall),
        "source_copy_similarity": _quantiles(copy_similarity),
        "duplicates": {
            "groups": len(duplicate_groups),
            "rows": duplicate_rows,
            "largest_group": max((len(ids) for ids in duplicate_groups.values()), default=0),
            "examples": [
                {"description": description, "request_ids": ids[:10]}
                for description, ids in sorted(duplicate_groups.items(), key=lambda item: (-len(item[1]), item[0]))[:20]
            ],
        },
        "invalid_examples": invalid_rows[:50],
        "judge_reason_top": judge_reasons.most_common(30),
        "api_accounting": _usage_and_cost(
            response_sources, args.input_price_cny_per_million, args.output_price_cny_per_million
        ),
        "sources": {
            "requests": {
                "path": str(requests_path),
                "sha256": _sha256_file(requests_path),
                "metadata_sha256": _sha256_file(requests_path.with_suffix(requests_path.suffix + ".metadata.json")),
                "prompt": requests_metadata.get("prompt"),
            },
            "responses": response_sources,
        },
        "review_sample": {
            "path": str(sample_path),
            "sha256": _sha256_file(sample_path),
            "entries": len(selected),
            "sample_seed": args.sample_seed,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--responses", nargs="+", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--review-sample", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--sample-per-event-count", type=int, default=100)
    parser.add_argument("--max-risk-samples", type=int, default=400)
    parser.add_argument("--sample-seed", type=int, default=20260806)
    parser.add_argument("--low-lexical-recall", type=float, default=0.45)
    parser.add_argument("--max-fallback-rate", type=float, default=0.02)
    parser.add_argument("--max-duplicate-rate", type=float, default=0.005)
    parser.add_argument("--high-copy-similarity", type=float, default=0.92)
    parser.add_argument("--max-high-copy-rate", type=float, default=0.15)
    parser.add_argument("--input-price-cny-per-million", type=float, default=3.0)
    parser.add_argument("--output-price-cny-per-million", type=float, default=6.0)
    parser.add_argument("--report-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit(args)
    print(json.dumps(report["quality_gate"], indent=2, sort_keys=True))
    if not report["quality_gate"]["eligible"] and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
