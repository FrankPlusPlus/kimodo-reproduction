# SPDX-License-Identifier: Apache-2.0
"""Independently review the deterministic V2 quality sample with a local Qwen model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from kimodo.resources.pipeline import _atomic_json

CRITICAL_CATEGORIES = frozenset(
    {"action", "order", "direction", "repetition", "object", "body_part", "hallucination"}
)
SYSTEM_PROMPT = """You are an independent motion-caption quality auditor. Compare the candidate with
the ordered source actions. A major error changes, drops, reorders, or invents an action, direction,
repetition, object, interaction, or body part. A minor error is awkward wording that preserves all
semantics. Return JSON only with exactly these fields:
{"verdict":"pass|minor|major","categories":["..."],"reason":"brief evidence"}.
Be strict about repeated actions and chronology, but do not penalize concise paraphrases."""
FORMAT_RETRY_PROMPT = """Your previous answer could not be parsed as the required JSON object.
Return only one complete JSON object with exactly verdict, categories, and reason. Do not use a
Markdown fence or add commentary. Keep reason under 40 words."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected one JSON object: {path}")
    return value


def _parse_verdict(text: str) -> dict:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    if not candidate.startswith("{") or not candidate.endswith("}"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("reviewer did not return a JSON object")
        candidate = candidate[start : end + 1]
    value = json.loads(candidate)
    if set(value) != {"verdict", "categories", "reason"}:
        raise ValueError("review verdict has unexpected fields")
    if value["verdict"] not in {"pass", "minor", "major"}:
        raise ValueError("review verdict is invalid")
    if not isinstance(value["categories"], list) or not all(
        isinstance(item, str) for item in value["categories"]
    ):
        raise TypeError("review categories must be a string list")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise TypeError("review reason must be a non-empty string")
    value["categories"] = sorted(set(value["categories"]))
    value["reason"] = " ".join(value["reason"].split())
    return value


def _messages(row: dict) -> list[dict[str, str]]:
    actions = "\n".join(
        f"{index}. {text}" for index, text in enumerate(row["source_texts"], start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Ordered source actions:\n{actions}\n\nCandidate:\n{row['description']}",
        },
    ]


def _validate_bindings(review_sample: Path, responses: Path, quality: Path) -> tuple[list[dict], dict]:
    quality_report = _load_json(quality)
    if quality_report.get("quality_gate", {}).get("eligible") is not True:
        raise ValueError("deterministic quality gate must pass before independent review")
    if quality_report.get("review_sample", {}).get("sha256") != _sha256(review_sample):
        raise ValueError("review sample disagrees with the quality report")
    response_sha = _sha256(responses)
    if not any(
        source.get("sha256") == response_sha
        for source in quality_report.get("sources", {}).get("responses", [])
        if isinstance(source, dict)
    ):
        raise ValueError("quality report is not bound to the supplied responses")
    rows = []
    seen = set()
    with review_sample.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = str(row["request_id"])
            if request_id in seen:
                raise ValueError(f"duplicate review request at line {line_number}: {request_id}")
            seen.add(request_id)
            rows.append(row)
    if len(rows) < 1_200:
        raise ValueError(f"independent review sample has only {len(rows)} unique requests")
    return rows, quality_report


def review(args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sample = Path(args.review_sample).expanduser().resolve()
    responses = Path(args.responses).expanduser().resolve()
    quality = Path(args.quality_report).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    verdicts = Path(args.verdicts).expanduser().resolve()
    partial = verdicts.with_suffix(verdicts.suffix + ".partial")
    if output.exists() or verdicts.exists():
        raise FileExistsError("refusing to overwrite an expert review output")
    rows, quality_report = _validate_bindings(sample, responses, quality)
    by_id = {str(row["request_id"]): row for row in rows}
    completed = {}
    if partial.is_file():
        with partial.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                request_id = str(row["request_id"])
                if request_id not in by_id or request_id in completed:
                    raise ValueError(f"invalid partial expert verdict at line {line_number}")
                completed[request_id] = row
    else:
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.touch(exist_ok=False)

    model_path = Path(args.model).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(args.device)
    model.eval()

    def generate(message_batches: list[list[dict[str, str]]], max_new_tokens: int) -> list[str]:
        prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for messages in message_batches
        ]
        tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **tokens,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = tokens["input_ids"].shape[1]
        return tokenizer.batch_decode(
            generated[:, prompt_width:], skip_special_tokens=True
        )

    pending = [row for row in rows if str(row["request_id"]) not in completed]
    with partial.open("a", encoding="utf-8") as sink:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            decoded = generate([_messages(row) for row in batch], args.max_new_tokens)
            for source, raw_verdict in zip(batch, decoded, strict=True):
                format_retry_count = 0
                try:
                    verdict = _parse_verdict(raw_verdict)
                except (TypeError, ValueError, json.JSONDecodeError) as first_error:
                    format_retry_count = 1
                    retry_messages = [
                        *_messages(source),
                        {"role": "assistant", "content": raw_verdict},
                        {"role": "user", "content": FORMAT_RETRY_PROMPT},
                    ]
                    retried = generate(
                        [retry_messages], max(args.max_new_tokens, 256)
                    )[0]
                    try:
                        verdict = _parse_verdict(retried)
                    except (TypeError, ValueError, json.JSONDecodeError) as retry_error:
                        request_id = str(source["request_id"])
                        raise ValueError(
                            "independent reviewer returned invalid JSON twice for "
                            f"request {request_id}; first={first_error}; retry={retry_error}"
                        ) from retry_error
                record = {
                    "request_id": str(source["request_id"]),
                    "event_count": source["event_count"],
                    "risk_flags": source.get("risk_flags", []),
                    "plan_reuse_count": source.get("plan_reuse_count", 1),
                    "format_retry_count": format_retry_count,
                    **verdict,
                }
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                completed[record["request_id"]] = record
            sink.flush()
            os.fsync(sink.fileno())
            print(f"Independent Qwen review: {len(completed)}/{len(rows)}", flush=True)

    if set(completed) != set(by_id):
        raise RuntimeError("independent expert review coverage is incomplete")
    os.replace(partial, verdicts)
    counts = Counter(row["verdict"] for row in completed.values())
    category_counts = Counter(
        category for row in completed.values() for category in row["categories"]
    )
    major_rows = [row for row in completed.values() if row["verdict"] == "major"]
    format_retry_requests = sum(
        int(row.get("format_retry_count", 0) > 0) for row in completed.values()
    )
    critical_major = [
        row
        for row in major_rows
        if CRITICAL_CATEGORIES.intersection(row["categories"])
    ]
    major_rate = len(major_rows) / len(rows)
    approved = not major_rows
    report = {
        "schema_version": 1,
        "status": "approved" if approved else "rejected_requires_targeted_repair",
        "reviewer": {
            "kind": "independent_local_model",
            "model_path": model_path.name,
            "config_sha256": _sha256(model_path / "config.json"),
            "weights_index_sha256": _sha256(model_path / "model.safetensors.index.json"),
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "format_retry_prompt_sha256": hashlib.sha256(
                FORMAT_RETRY_PROMPT.encode()
            ).hexdigest(),
            "decoding": "greedy_no_thinking",
        },
        "bindings": {
            "responses_sha256": _sha256(responses),
            "quality_report_sha256": _sha256(quality),
            "review_sample_sha256": _sha256(sample),
            "verdicts_sha256": _sha256(verdicts),
        },
        "review": {
            "reviewed_unique_requests": len(rows),
            "verdict_counts": dict(sorted(counts.items())),
            "category_counts": dict(sorted(category_counts.items())),
            "major_semantic_errors": len(major_rows),
            "major_semantic_error_rate": major_rate,
            "critical_major_errors": len(critical_major),
            "unresolved_critical_errors": len(critical_major),
            "format_retry_requests": format_retry_requests,
            "major_request_ids": sorted(row["request_id"] for row in major_rows),
        },
        "deterministic_quality_gate": quality_report["quality_gate"],
    }
    _atomic_json(output, report)
    if not approved:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-sample", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--verdicts", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    report = review(build_parser().parse_args())
    print(json.dumps(report["review"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
