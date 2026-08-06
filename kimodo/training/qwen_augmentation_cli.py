# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate sharded, resumable Qwen descriptions for V2 timeline requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from .timeline_multi_cli import SYSTEM_PROMPT, validate_description

JUDGE_PROMPT = """Audit whether a candidate motion description preserves every ordered source action.
Reject if any action, order, left/right or forward/backward direction, body part, object, interaction,
or repetition count is dropped, changed, reordered, or invented. Fluency alone is not sufficient.
Return strict JSON only: {\"accepted\": true_or_false, \"reason\": \"brief reason\"}."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_snapshot_identity(value: str) -> dict | None:
    root = Path(value).expanduser()
    if not root.is_dir():
        return None
    root = root.resolve()
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Local Qwen snapshot has no weight index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_files = sorted(set(index.get("weight_map", {}).values()))
    required = ["config.json", "tokenizer.json", "tokenizer_config.json", *weight_files]
    records = {}
    for name in required:
        path = root / name
        if not path.is_file() or path.stat().st_size < 1:
            raise FileNotFoundError(f"Local Qwen snapshot is incomplete: {path}")
        records[name] = {"size": path.stat().st_size, "sha256": _sha256_file(path)}
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": records,
    }


def _shard(request_id: str, count: int) -> int:
    return int(request_id[:16], 16) % count


def _parse_description(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
        if text.startswith("json"):
            text = text[4:].strip()
    payload = json.loads(text)
    if set(payload) != {"description"} or not isinstance(payload["description"], str):
        raise ValueError("Qwen output must contain only a string description field")
    return " ".join(payload["description"].split())


def _messages(source_texts: list[str]) -> list[dict[str, str]]:
    actions = "\n".join(f"{index + 1}. {text}" for index, text in enumerate(source_texts))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Ordered source actions:\n{actions}"},
    ]


def _judge_messages(source_texts: list[str], description: str) -> list[dict[str, str]]:
    actions = "\n".join(f"{index + 1}. {text}" for index, text in enumerate(source_texts))
    return [
        {"role": "system", "content": JUDGE_PROMPT},
        {
            "role": "user",
            "content": f"Ordered source actions:\n{actions}\n\nCandidate:\n{description}",
        },
    ]


def _source_preserving_fallback(source_texts: list[str]) -> str:
    actions = []
    for index, text in enumerate(source_texts):
        cleaned = " ".join(text.split()).strip().rstrip(".!?;:")
        connector = "First" if index == 0 else "Then"
        actions.append(f"{connector}, {cleaned}")
    return "The ordered motion is as follows: " + ". ".join(actions) + "."


def _parse_judge(value: str) -> dict:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
        if text.startswith("json"):
            text = text[4:].strip()
    payload = json.loads(text)
    if set(payload) != {"accepted", "reason"}:
        raise ValueError("semantic judge must contain accepted and reason only")
    if not isinstance(payload["accepted"], bool) or not isinstance(payload["reason"], str):
        raise TypeError("semantic judge fields have invalid types")
    return payload


def generate(args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    requests_path = Path(args.requests).expanduser().resolve()
    requests_metadata_path = requests_path.with_suffix(requests_path.suffix + ".metadata.json")
    if not requests_metadata_path.is_file():
        raise FileNotFoundError(f"Qwen requests metadata is missing: {requests_metadata_path}")
    requests_metadata = json.loads(requests_metadata_path.read_text(encoding="utf-8"))
    if requests_metadata.get("output", {}).get("sha256") != _sha256_file(requests_path):
        raise ValueError("Qwen requests hash disagrees with its metadata")
    if requests_metadata.get("prompt", {}).get("sha256") != hashlib.sha256(
        SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest():
        raise ValueError("Qwen requests were prepared for a different prompt")
    output = Path(args.output).expanduser().resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    partial_metadata = partial.with_suffix(partial.suffix + ".metadata.json")
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed Qwen output: {output}")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    output.parent.mkdir(parents=True, exist_ok=True)

    model_snapshot = _local_snapshot_identity(args.model)
    binding = {
        "schema_version": 1,
        "requests_sha256": _sha256_file(requests_path),
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest(),
        "model": args.model_identity,
        "model_path": args.model,
        "revision": args.revision,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "max_new_tokens": args.max_new_tokens,
        "max_requests": args.max_requests,
        "semantic_judge": True,
        "local_model_snapshot": model_snapshot,
    }
    if partial.is_file() != partial_metadata.is_file():
        raise FileNotFoundError("Qwen partial output and its binding sidecar must exist together")
    if partial_metadata.is_file():
        recorded_binding = json.loads(partial_metadata.read_text(encoding="utf-8"))
        if recorded_binding != binding:
            raise ValueError("Qwen partial output belongs to different inputs/model/shard settings")
    else:
        with partial.open("x", encoding="utf-8"):
            pass
        partial_metadata.write_text(
            json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    completed = set()
    existing_fallbacks = 0
    if partial.is_file():
        valid_records = []
        with partial.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    judge = record.get("semantic_judge")
                    accepted_judge = isinstance(judge, dict) and judge.get("accepted") is True
                    accepted_fallback = bool(
                        record.get("fallback") == "deterministic_source_preserving_template"
                        and record.get("deterministic_source_preservation") is True
                    )
                    if (
                        not record.get("error")
                        and record.get("description")
                        and (accepted_judge or accepted_fallback)
                    ):
                        request_id = str(record["request_id"])
                        if request_id in completed:
                            raise ValueError(f"Partial output repeats valid request {request_id}")
                        completed.add(request_id)
                        valid_records.append(record)
                        if record.get("fallback") == "deterministic_source_preserving_template":
                            existing_fallbacks += 1
        # Failed generations are retryable. Compact them out before appending so
        # a successful retry cannot create duplicate response identities.
        compact = partial.with_suffix(partial.suffix + ".compact")
        with compact.open("x", encoding="utf-8") as sink:
            for record in valid_records:
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(compact, partial)

    requests = []
    eligible_ids = set()
    with requests_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if _shard(row["request_id"], args.shard_count) == args.shard_index:
                eligible_ids.add(row["request_id"])
                if row["request_id"] not in completed:
                    requests.append(row)
    if not completed <= eligible_ids:
        raise ValueError("Partial output contains request ids outside the selected shard")
    if args.max_requests is not None:
        if args.max_requests < 1:
            raise ValueError("max-requests must be positive")
        by_event_count = {count: [] for count in range(2, 6)}
        for row in requests:
            by_event_count[len(row["source_texts"])].append(row)
        stratified = []
        depth = 0
        while len(stratified) < min(args.max_requests, len(requests)):
            added = False
            for count in range(2, 6):
                rows = by_event_count[count]
                if depth < len(rows):
                    stratified.append(rows[depth])
                    added = True
                    if len(stratified) == args.max_requests:
                        break
            if not added:
                break
            depth += 1
        requests = stratified

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.eval()
    started = time.perf_counter()
    invalid = 0
    fallbacks = existing_fallbacks
    generated = 0
    with partial.open("a", encoding="utf-8") as sink:
        for offset in range(0, len(requests), args.batch_size):
            batch = requests[offset : offset + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    _messages(row["source_texts"]),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for row in batch
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)
            with torch.inference_mode():
                tokens = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            decoded = tokenizer.batch_decode(
                tokens[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
            )
            parsed = []
            for raw in decoded:
                try:
                    description = _parse_description(raw)
                    validate_description(batch[len(parsed)]["source_texts"], description)
                    error = None
                except (ValueError, json.JSONDecodeError) as exc:
                    description = None
                    error = str(exc)
                parsed.append({"description": description, "raw": raw, "error": error})

            judge_indices = [index for index, item in enumerate(parsed) if not item["error"]]
            if judge_indices:
                judge_prompts = [
                    tokenizer.apply_chat_template(
                        _judge_messages(batch[index]["source_texts"], parsed[index]["description"]),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for index in judge_indices
                ]
                judge_encoded = tokenizer(
                    judge_prompts, return_tensors="pt", padding=True
                ).to(args.device)
                with torch.inference_mode():
                    judge_tokens = model.generate(
                        **judge_encoded,
                        do_sample=False,
                        max_new_tokens=64,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                judge_decoded = tokenizer.batch_decode(
                    judge_tokens[:, judge_encoded["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                for index, raw_judge in zip(judge_indices, judge_decoded, strict=True):
                    try:
                        verdict = _parse_judge(raw_judge)
                        parsed[index]["semantic_judge"] = verdict
                        if not verdict["accepted"]:
                            parsed[index]["error"] = f"semantic judge rejected: {verdict['reason']}"
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        parsed[index]["semantic_judge"] = {
                            "accepted": False,
                            "reason": f"invalid judge JSON: {exc}",
                        }
                        parsed[index]["error"] = str(exc)

            for request, item in zip(batch, parsed, strict=True):
                description = item["description"]
                generation_error = item["error"]
                fallback = None
                if generation_error:
                    description = _source_preserving_fallback(request["source_texts"])
                    validate_description(request["source_texts"], description)
                    fallback = "deterministic_source_preserving_template"
                    fallbacks += 1
                error = None
                sink.write(
                    json.dumps(
                        {
                            "request_id": request["request_id"],
                            "description": description,
                            "raw_output": item["raw"] if generation_error else None,
                            "error": error,
                            "fallback": fallback,
                            "fallback_reason": generation_error,
                            "deterministic_source_preservation": bool(fallback),
                            "semantic_judge": item.get("semantic_judge"),
                            "model": args.model_identity,
                            "revision": args.revision,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                generated += 1
            sink.flush()
            os.fsync(sink.fileno())
            elapsed = time.perf_counter() - started
            print(
                f"Qwen shard {args.shard_index}: {generated}/{len(requests)} new, "
                f"{len(completed) + generated} total, {generated / max(elapsed, 1e-6):.2f} req/s",
                flush=True,
            )
    if invalid:
        raise RuntimeError(
            f"Shard contains {invalid} invalid JSON generations in {partial}; "
            "inspect and regenerate those requests before publication"
        )
    os.replace(partial, output)
    metadata = {
        "schema_version": 1,
        "generator": "kimodo.training.qwen_augmentation_cli",
        "model": args.model_identity,
        "model_path": args.model,
        "local_model_snapshot": model_snapshot,
        "revision": args.revision,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest(),
        "requests": {"path": str(requests_path), "sha256": _sha256_file(requests_path)},
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "output": {"path": str(output), "sha256": _sha256_file(output), "entries": len(completed) + generated},
        "quality": {"semantic_fallbacks": fallbacks, "invalid_published": invalid},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial_metadata.unlink()
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True, help="Complete local Qwen3-32B snapshot directory")
    parser.add_argument("--model-identity", default="Qwen/Qwen3-32B")
    parser.add_argument("--revision", default="9216db5781bf21249d130ec9da846c4624c16137")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser


def main() -> None:
    generate(build_parser().parse_args())


if __name__ == "__main__":
    main()
