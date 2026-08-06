# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate audited V2 timeline descriptions with an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import urllib3

from .qwen_augmentation_cli import JUDGE_PROMPT, _source_preserving_fallback
from .timeline_multi_cli import (
    SYSTEM_PROMPT,
    _sha256_file,
    description_word_limit,
    validate_description,
)

BATCH_GENERATION_PROMPT = """Write one motion description for every input item.
For each item, preserve every source action and chronological order, including left/right,
forward/backward, body parts, objects, repetitions, and interactions. Do not invent intent,
speed, direction, objects, actions, or transitions. Rewrite naturally: do not merely concatenate
or copy the source sentences. Remove redundant repeated subjects or stance phrases, use clear
temporal connectors, and vary the syntax across items, while retaining actions that genuinely
repeat in the timeline. Each input item provides max_words; its description must contain 7 to
max_words English words and usually 1-3 sentences. Return one strict JSON object only, with this shape:
{"items":[{"request_id":"the unchanged input id","description":"the description"}]}.
Return every input request_id exactly once and no additional request_id."""

BATCH_JUDGE_PROMPT = """Audit every candidate against its ordered source actions.
Reject a candidate if any action, chronological order, left/right or forward/backward direction,
body part, object, interaction, or repetition count is dropped, changed, reordered, or invented.
Paraphrasing and combining adjacent source actions into one grammatical clause are allowed and are
not reordering when their chronology is retained. Mentally map each ordered source item to the
candidate before deciding; do not reject merely because repeated subjects were merged or wording
was compressed. Genuine repeated actions must still each be represented. Fluency alone is not
sufficient. Return one strict JSON object only, with exactly this shape:
{"items":[{"request_id":"the unchanged input id","accepted":true,"reason":"brief reason"}]}.
Return every input request_id exactly once and no additional request_id."""

BATCH_REPAIR_PROMPT = """Repair every rejected motion description using its ordered source actions
and audit reason. Preserve every action and chronological order, including directions, body parts,
objects, interactions, and genuine repetitions. Correct the stated semantic defect without merely
concatenating the source sentences. Do not invent anything. Obey each item's max_words. Return one
strict JSON object only, with exactly this shape:
{"items":[{"request_id":"the unchanged input id","description":"the repaired description"}]}.
Return every input request_id exactly once and no additional request_id."""


def _canonical_sha256(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("LLM base URL must be an HTTPS URL without embedded credentials")
    return value


def _content(payload: dict) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("API response has no choices[0].message.content") from error
    if not isinstance(content, str):
        raise TypeError("API response content must be a string")
    return content


def _strict_items(content: str, expected_ids: list[str], fields: dict[str, type]) -> list[dict]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("API response content is not valid JSON") from error
    if set(payload) != {"items"} or not isinstance(payload["items"], list):
        raise ValueError("API response must contain only an items array")
    expected_keys = {"request_id", *fields}
    by_id = {}
    for item in payload["items"]:
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise ValueError(f"API item must contain exactly {sorted(expected_keys)}")
        request_id = item["request_id"]
        if not isinstance(request_id, str) or request_id in by_id:
            raise ValueError("API item has an invalid or duplicate request_id")
        for name, expected_type in fields.items():
            if type(item[name]) is not expected_type:  # bool must not pass as int
                raise TypeError(f"API item field {name!r} has the wrong type")
        by_id[request_id] = item
    if set(by_id) != set(expected_ids) or len(by_id) != len(expected_ids):
        raise ValueError("API response request_id coverage differs from its input batch")
    return [by_id[request_id] for request_id in expected_ids]


class _RateLimiter:
    def __init__(self, requests_per_minute: float):
        if requests_per_minute <= 0:
            raise ValueError("requests-per-minute must be positive")
        self._interval = 60.0 / requests_per_minute
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if delay:
            time.sleep(delay)


class _ApiClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        requests_per_minute: float,
        timeout: float,
        max_retries: int,
        receipts,
    ):
        self.endpoint = f"{base_url}/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.pool = urllib3.PoolManager(cert_reqs="CERT_REQUIRED")
        self.limiter = _RateLimiter(requests_per_minute)
        self.receipts = receipts
        self.receipt_lock = threading.Lock()

    def post(self, payload: dict, *, request_kind: str, request_ids: list[str]) -> dict:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        batch_sha256 = _canonical_sha256({"kind": request_kind, "request_ids": request_ids, "payload": payload})
        last_error = None
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                response = self.pool.request(
                    "POST",
                    self.endpoint,
                    body=encoded,
                    headers={"Content-Type": "application/json", "api-key": self.api_key},
                    timeout=urllib3.Timeout(total=self.timeout),
                    retries=False,
                )
                if response.status == 200:
                    result = json.loads(response.data.decode("utf-8"))
                    self._record_receipt(result, request_kind, request_ids, batch_sha256)
                    return result
                if response.status not in {408, 409, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"LLM API returned non-retryable HTTP {response.status}")
                last_error = RuntimeError(f"LLM API returned HTTP {response.status}")
            except (OSError, urllib3.exceptions.HTTPError, json.JSONDecodeError) as error:
                last_error = error
            if attempt < self.max_retries:
                time.sleep(min(30.0, (2**attempt) + random.random()))
        raise RuntimeError(
            f"LLM API failed after {self.max_retries + 1} attempts: {type(last_error).__name__}"
        ) from last_error

    def _record_receipt(self, response: dict, request_kind: str, request_ids: list[str], batch_sha256: str) -> None:
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        receipt = {
            "request_kind": request_kind,
            "logical_items": len(request_ids),
            "batch_sha256": batch_sha256,
            "response_id": response.get("id"),
            "created": response.get("created"),
            "response_model": response.get("model"),
            "usage": {
                key: int(value)
                for key, value in usage.items()
                if isinstance(value, int) and not isinstance(value, bool)
            },
        }
        with self.receipt_lock:
            self.receipts.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")
            self.receipts.flush()
            os.fsync(self.receipts.fileno())


def _chat_payload(model: str, system: str, items: list[dict], max_tokens: int, temperature: float) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps({"items": items}, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }


def _process_batch(rows: list[dict], client: _ApiClient, args) -> list[dict]:
    request_ids = [str(row["request_id"]) for row in rows]
    generation_items = [
        {
            "request_id": row["request_id"],
            "source_texts": row["source_texts"],
            "max_words": description_word_limit(row["source_texts"]),
        }
        for row in rows
    ]
    generation_response = client.post(
        _chat_payload(
            args.model,
            BATCH_GENERATION_PROMPT,
            generation_items,
            args.max_completion_tokens,
            args.temperature,
        ),
        request_kind="generation",
        request_ids=request_ids,
    )
    generated = _strict_items(_content(generation_response), request_ids, {"description": str})
    judged: list[dict | None] = [None] * len(rows)
    initial_rejections = {}
    judge_indices = []
    for index, (row, item) in enumerate(zip(rows, generated, strict=True)):
        item["description"] = " ".join(item["description"].split())
        try:
            validate_description(row["source_texts"], item["description"])
            judge_indices.append(index)
        except ValueError as error:
            verdict = {
                "request_id": row["request_id"],
                "accepted": False,
                "reason": f"deterministic validator rejected: {error}",
            }
            judged[index] = verdict
            initial_rejections[index] = {
                "candidate": item["description"],
                "judge": verdict,
            }

    judge_ids = [request_ids[index] for index in judge_indices]
    judge_items = [
        {
            "request_id": rows[index]["request_id"],
            "source_texts": rows[index]["source_texts"],
            "candidate": generated[index]["description"],
        }
        for index in judge_indices
    ]
    judge_response = None
    if judge_items:
        judge_response = client.post(
            _chat_payload(
                args.judge_model,
                BATCH_JUDGE_PROMPT,
                judge_items,
                args.judge_max_completion_tokens,
                0,
            ),
            request_kind="semantic_judge",
            request_ids=judge_ids,
        )
        judge_results = _strict_items(_content(judge_response), judge_ids, {"accepted": bool, "reason": str})
        for index, verdict in zip(judge_indices, judge_results, strict=True):
            judged[index] = verdict
            if not verdict["accepted"]:
                initial_rejections[index] = {
                    "candidate": generated[index]["description"],
                    "judge": verdict,
                }
    if any(verdict is None for verdict in judged):
        raise RuntimeError("Internal judge coverage is incomplete")
    repair_generation_response = None
    repair_judge_response = None
    repair_error = None
    if initial_rejections:
        repair_indices = sorted(initial_rejections)
        repair_ids = [request_ids[index] for index in repair_indices]
        repair_items = [
            {
                "request_id": rows[index]["request_id"],
                "source_texts": rows[index]["source_texts"],
                "rejected_candidate": generated[index]["description"],
                "audit_reason": judged[index]["reason"],
                "max_words": description_word_limit(rows[index]["source_texts"]),
            }
            for index in repair_indices
        ]
        try:
            repair_generation_response = client.post(
                _chat_payload(
                    args.model,
                    BATCH_REPAIR_PROMPT,
                    repair_items,
                    args.max_completion_tokens,
                    args.temperature,
                ),
                request_kind="repair_generation",
                request_ids=repair_ids,
            )
            repaired = _strict_items(_content(repair_generation_response), repair_ids, {"description": str})
            valid_repair_indices = []
            valid_repaired = []
            for index, item in zip(repair_indices, repaired, strict=True):
                item["description"] = " ".join(item["description"].split())
                generated[index] = item
                try:
                    validate_description(rows[index]["source_texts"], item["description"])
                    valid_repair_indices.append(index)
                    valid_repaired.append(item)
                except ValueError as error:
                    judged[index] = {
                        "request_id": rows[index]["request_id"],
                        "accepted": False,
                        "reason": f"repair validator rejected: {error}",
                    }
            repair_judge_items = [
                {
                    "request_id": rows[index]["request_id"],
                    "source_texts": rows[index]["source_texts"],
                    "candidate": item["description"],
                }
                for index, item in zip(valid_repair_indices, valid_repaired, strict=True)
            ]
            if repair_judge_items:
                valid_repair_ids = [request_ids[index] for index in valid_repair_indices]
                repair_judge_response = client.post(
                    _chat_payload(
                        args.judge_model,
                        BATCH_JUDGE_PROMPT,
                        repair_judge_items,
                        args.judge_max_completion_tokens,
                        0,
                    ),
                    request_kind="repair_judge",
                    request_ids=valid_repair_ids,
                )
                repaired_judges = _strict_items(
                    _content(repair_judge_response),
                    valid_repair_ids,
                    {"accepted": bool, "reason": str},
                )
                for index, verdict in zip(valid_repair_indices, repaired_judges, strict=True):
                    judged[index] = verdict
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            repair_error = f"{type(error).__name__}: {error}"

    records = []
    for index, (request, generation, verdict) in enumerate(zip(rows, generated, judged, strict=True)):
        fallback = None
        fallback_reason = None
        description = generation["description"]
        if not verdict["accepted"]:
            fallback = "deterministic_source_preserving_template"
            fallback_reason = (
                f"repair pass failed: {repair_error}"
                if repair_error
                else f"repair semantic judge rejected: {verdict['reason']}"
            )
            description = _source_preserving_fallback(request["source_texts"])
            validate_description(request["source_texts"], description)
        initial_rejection = initial_rejections.get(index)
        records.append(
            {
                "request_id": request["request_id"],
                "description": description,
                "raw_output": None,
                "error": None,
                "fallback": fallback,
                "fallback_reason": fallback_reason,
                "deterministic_source_preservation": bool(fallback),
                "semantic_judge": verdict,
                "initial_rejected_candidate": (initial_rejection["candidate"] if initial_rejection else None),
                "initial_semantic_judge": (initial_rejection["judge"] if initial_rejection else None),
                "repair_attempted": bool(initial_rejection),
                "repair_succeeded": bool(initial_rejection and verdict["accepted"]),
                "repair_error": repair_error if initial_rejection else None,
                "model": args.model,
                "revision": args.revision,
                "judge_model": args.judge_model,
                "api_generation_response_id": generation_response.get("id"),
                "api_judge_response_id": (judge_response.get("id") if judge_response else None),
                "api_repair_generation_response_id": (
                    repair_generation_response.get("id") if initial_rejection and repair_generation_response else None
                ),
                "api_repair_judge_response_id": (
                    repair_judge_response.get("id") if initial_rejection and repair_judge_response else None
                ),
            }
        )
    return records


def _process_resilient(rows: list[dict], client: _ApiClient, args) -> list[dict]:
    try:
        return _process_batch(rows, client, args)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        if len(rows) > 1:
            print(
                f"LLM batch size {len(rows)} failed closed; retrying halves: {type(error).__name__}: {error}",
                flush=True,
            )
            middle = len(rows) // 2
            return _process_resilient(rows[:middle], client, args) + _process_resilient(rows[middle:], client, args)
        request = rows[0]
        description = _source_preserving_fallback(request["source_texts"])
        validate_description(request["source_texts"], description)
        return [
            {
                "request_id": request["request_id"],
                "description": description,
                "raw_output": None,
                "error": None,
                "fallback": "deterministic_source_preserving_template",
                "fallback_reason": f"API generation/judge failure: {type(error).__name__}: {error}",
                "deterministic_source_preservation": True,
                "semantic_judge": None,
                "model": args.model,
                "revision": args.revision,
                "judge_model": args.judge_model,
                "api_generation_response_id": None,
                "api_judge_response_id": None,
            }
        ]


def _eligible_requests(path: Path, shard_index: int, shard_count: int) -> list[dict]:
    from .qwen_augmentation_cli import _shard

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if _shard(str(row["request_id"]), shard_count) == shard_index:
                    rows.append(row)
    return rows


def _stratified_limit(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return rows
    if limit < 1:
        raise ValueError("max-requests must be positive")
    buckets = {count: [] for count in range(2, 6)}
    for row in rows:
        buckets[len(row["source_texts"])].append(row)
    selected = []
    depth = 0
    while len(selected) < min(limit, len(rows)):
        added = False
        for count in range(2, 6):
            if depth < len(buckets[count]):
                selected.append(buckets[count][depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def _receipt_summary(path: Path) -> dict:
    counts = {"api_calls": 0, "logical_items": 0}
    usage: dict[str, int] = {}
    response_models = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            counts["api_calls"] += 1
            counts["logical_items"] += int(row.get("logical_items", 0))
            if row.get("response_model"):
                response_models.add(str(row["response_model"]))
            for key, value in row.get("usage", {}).items():
                usage[key] = usage.get(key, 0) + int(value)
    return {**counts, "usage": dict(sorted(usage.items())), "response_models": sorted(response_models)}


def generate(args) -> dict:
    requests_path = Path(args.requests).expanduser().resolve()
    requests_metadata_path = requests_path.with_suffix(requests_path.suffix + ".metadata.json")
    if not requests_metadata_path.is_file():
        raise FileNotFoundError(f"LLM requests metadata is missing: {requests_metadata_path}")
    requests_metadata = json.loads(requests_metadata_path.read_text(encoding="utf-8"))
    requests_sha256 = _sha256_file(requests_path)
    if requests_metadata.get("output", {}).get("sha256") != requests_sha256:
        raise ValueError("LLM requests hash disagrees with its metadata")
    prompt_sha256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    if requests_metadata.get("prompt", {}).get("sha256") != prompt_sha256:
        raise ValueError("LLM requests were prepared for a different semantic prompt")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")

    base_url = _normalize_base_url(args.base_url)
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Required API key environment variable {args.api_key_env!r} is not set")
    output = Path(args.output).expanduser().resolve()
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    partial = output.with_suffix(output.suffix + ".partial")
    partial_metadata = partial.with_suffix(partial.suffix + ".metadata.json")
    partial_receipts = partial.with_suffix(partial.suffix + ".api-receipts.jsonl")
    receipts_output = output.with_suffix(output.suffix + ".api-receipts.jsonl")
    if output.exists() or metadata_path.exists() or receipts_output.exists():
        raise FileExistsError(f"Refusing to overwrite completed LLM output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    producer_identity = {
        "provider": "openai_compatible",
        "base_url": base_url,
        "model": args.model,
        "judge_model": args.judge_model,
        "revision": args.revision,
        "generation_transport_prompt_sha256": hashlib.sha256(BATCH_GENERATION_PROMPT.encode("utf-8")).hexdigest(),
        "judge_transport_prompt_sha256": hashlib.sha256(BATCH_JUDGE_PROMPT.encode("utf-8")).hexdigest(),
        "repair_transport_prompt_sha256": hashlib.sha256(BATCH_REPAIR_PROMPT.encode("utf-8")).hexdigest(),
        "thinking": "disabled",
        "generation_temperature": args.temperature,
        "judge_temperature": 0,
    }
    binding = {
        "schema_version": 1,
        "requests_sha256": requests_sha256,
        "prompt_sha256": prompt_sha256,
        "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest(),
        "producer_identity_sha256": _canonical_sha256(producer_identity),
        "producer_identity": producer_identity,
        "api_key_env": args.api_key_env,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "batch_size": args.batch_size,
        "max_requests": args.max_requests,
        "max_completion_tokens": args.max_completion_tokens,
        "judge_max_completion_tokens": args.judge_max_completion_tokens,
        "generation_temperature": args.temperature,
    }
    states = [partial.is_file(), partial_metadata.is_file(), partial_receipts.is_file()]
    if any(states) and not all(states):
        raise FileNotFoundError("LLM partial output, binding, and API receipts must exist together")
    if all(states):
        if json.loads(partial_metadata.read_text(encoding="utf-8")) != binding:
            raise ValueError("LLM partial output belongs to different inputs/model/settings")
    else:
        partial.touch(exist_ok=False)
        partial_receipts.touch(exist_ok=False)
        partial_metadata.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    eligible = _stratified_limit(
        _eligible_requests(requests_path, args.shard_index, args.shard_count), args.max_requests
    )
    eligible_ids = {str(row["request_id"]) for row in eligible}
    completed = set()
    existing_records = []
    with partial.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = str(row["request_id"])
            judge = row.get("semantic_judge")
            valid = (
                not row.get("error")
                and row.get("description")
                and (
                    isinstance(judge, dict)
                    and judge.get("accepted") is True
                    or row.get("fallback") == "deterministic_source_preserving_template"
                    and row.get("deterministic_source_preservation") is True
                )
            )
            if not valid or request_id in completed:
                raise ValueError("LLM partial output contains an invalid or duplicate record")
            completed.add(request_id)
            existing_records.append(row)
    if not completed <= eligible_ids:
        raise ValueError("LLM partial output contains ids outside the selected shard/limit")
    pending = [row for row in eligible if str(row["request_id"]) not in completed]
    batches = [pending[offset : offset + args.batch_size] for offset in range(0, len(pending), args.batch_size)]

    started = time.perf_counter()
    generated = 0
    with partial_receipts.open("a", encoding="utf-8") as receipts, partial.open("a", encoding="utf-8") as sink:
        client = _ApiClient(
            base_url=base_url,
            api_key=api_key,
            requests_per_minute=args.requests_per_minute,
            timeout=args.timeout,
            max_retries=args.max_retries,
            receipts=receipts,
        )
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for records in executor.map(lambda batch: _process_resilient(batch, client, args), batches):
                for row in records:
                    sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    generated += 1
                sink.flush()
                os.fsync(sink.fileno())
                elapsed = time.perf_counter() - started
                print(
                    f"LLM shard {args.shard_index}: {generated}/{len(pending)} new, "
                    f"{len(completed) + generated}/{len(eligible)} total, "
                    f"{generated / max(elapsed, 1e-6):.2f} items/s",
                    flush=True,
                )

    total_entries = len(completed) + generated
    if total_entries != len(eligible):
        raise RuntimeError(f"LLM output coverage incomplete: {total_entries}/{len(eligible)}")
    fallbacks = 0
    repairs_attempted = 0
    repairs_succeeded = 0
    with partial.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                fallbacks += bool(row.get("fallback"))
                repairs_attempted += bool(row.get("repair_attempted"))
                repairs_succeeded += bool(row.get("repair_succeeded"))
    receipt_summary = _receipt_summary(partial_receipts)
    os.replace(partial, output)
    os.replace(partial_receipts, receipts_output)
    metadata = {
        "schema_version": 2,
        "generator": "kimodo.training.llm_api_augmentation_cli",
        "provider": "openai_compatible",
        "base_url": base_url,
        "api_key_env": args.api_key_env,
        "model": args.model,
        "judge_model": args.judge_model,
        "revision": args.revision,
        "model_weight_identity": "provider_managed_unavailable",
        "producer_identity": producer_identity,
        "producer_identity_sha256": binding["producer_identity_sha256"],
        "decoding": {
            "generation_temperature": args.temperature,
            "judge_temperature": 0,
            "thinking": "disabled",
            "response_format": "json_object",
            "batch_size": args.batch_size,
            "max_completion_tokens": args.max_completion_tokens,
            "judge_max_completion_tokens": args.judge_max_completion_tokens,
        },
        "prompt_sha256": prompt_sha256,
        "transport_prompt_sha256": producer_identity["generation_transport_prompt_sha256"],
        "judge_prompt_sha256": binding["judge_prompt_sha256"],
        "judge_transport_prompt_sha256": producer_identity["judge_transport_prompt_sha256"],
        "repair_transport_prompt_sha256": producer_identity["repair_transport_prompt_sha256"],
        "requests": {"path": str(requests_path), "sha256": requests_sha256},
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "api_receipts": {
            "path": str(receipts_output),
            "sha256": _sha256_file(receipts_output),
            **receipt_summary,
        },
        "output": {"path": str(output), "sha256": _sha256_file(output), "entries": total_entries},
        "quality": {
            "semantic_fallbacks": fallbacks,
            "repairs_attempted": repairs_attempted,
            "repairs_succeeded": repairs_succeeded,
            "invalid_published": 0,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial_metadata.unlink()
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("PRODUCT_GRAPH_LLM_BASE_URL", "https://api.xiaomimimo.com/v1"),
    )
    parser.add_argument("--api-key-env", default="PRODUCT_GRAPH_LLM_API_KEY")
    parser.add_argument("--model", default=os.environ.get("PRODUCT_GRAPH_LLM_MODEL", "mimo-v2.5-pro"))
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--revision", default="provider-managed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--requests-per-minute", type=float, default=90.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--judge-max-completion-tokens", type=int, default=1024)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.judge_model = args.judge_model or args.model
    if args.batch_size < 1 or args.concurrency < 1:
        raise ValueError("batch-size and concurrency must be positive")
    metadata = generate(args)
    print(json.dumps({"output": metadata["output"], "quality": metadata["quality"]}, indent=2))


if __name__ == "__main__":
    main()
