# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare leakage-safe V2 timeline spans and content-addressed Qwen requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

from kimodo.common.file_permissions import publish_file

PROMPT_VERSION = "kimodo-benchmark-multi-v2.2"
SYSTEM_PROMPT = """You rewrite ordered motion annotations into one natural English motion description.
Preserve every action and their chronological order. Preserve left/right, forward/backward, body parts,
objects, repetitions, and interactions exactly. Do not invent intent, speed, direction, objects, actions,
or transitions. Return strict JSON only: {\"description\": \"...\"}. Write one coherent description
of 7-90 words, usually in 1-3 sentences; combine events only where it is natural."""

BENCHMARK_MULTI_PROPORTIONS = {2: 0.7861, 3: 0.1817, 4: 0.0279, 5: 0.0043}
CRITICAL_GROUPS = (
    ("left",),
    ("right",),
    ("forward", "forwards"),
    ("backward", "backwards"),
    ("clockwise",),
    ("counterclockwise", "anticlockwise"),
)


def description_word_limit(source_texts: list[str]) -> int:
    """Allow extra room only when preserving the source cannot fit the usual 90 words."""
    source_words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", " ".join(source_texts).lower())
    return min(180, max(90, len(source_words) + 30))


def validate_description(source_texts: list[str], description: str) -> None:
    if not isinstance(description, str) or not description.strip():
        raise ValueError("empty LLM description")
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", description.lower())
    maximum = description_word_limit(source_texts)
    if not 7 <= len(words) <= maximum:
        raise ValueError(
            f"LLM description has {len(words)} words; expected 7..{maximum} for this source information load"
        )
    source_words = set(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", " ".join(source_texts).lower()))
    output_words = set(words)
    for alternatives in CRITICAL_GROUPS:
        if source_words.intersection(alternatives) and not output_words.intersection(alternatives):
            raise ValueError(f"LLM description dropped critical direction token {alternatives}")
    lowered = description.lower()
    if any(marker in lowered for marker in ("as an ai", "i cannot", "ordered source actions")):
        raise ValueError("LLM description contains generation boilerplate")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _split_keys(path: Path) -> set[str]:
    return {
        line.strip().replace("\\", "/").removesuffix(".bvh").removesuffix(".csv")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _motion_key(value: str) -> str:
    parts = Path(value.replace("\\", "/")).parts
    if len(parts) < 2:
        raise ValueError(f"Cannot derive date/name train key from motion path: {value!r}")
    return f"{parts[-2]}/{Path(parts[-1]).stem}"


def _event_index(row: dict) -> int:
    parts = str(row["id"]).rsplit(":event:", 1)
    if len(parts) != 2:
        raise ValueError(f"Event row has an unrecognized id: {row['id']!r}")
    return int(parts[1].split(":", 1)[0])


def _load_events(source: Path, train_keys: set[str], fps: int) -> tuple[dict, Counter]:
    groups: dict[str, list[dict]] = defaultdict(list)
    counts = Counter()
    seen_ids: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            counts[f"source_kind/{row.get('sample_kind', 'full')}"] += 1
            if row.get("split") != "train":
                raise ValueError(f"{source}:{line_number} is not a train row")
            key = _motion_key(str(row["motion"]))
            if key not in train_keys:
                raise ValueError(f"{source}:{line_number} motion is outside official train split: {key}")
            counts["source_rows"] += 1
            if row.get("sample_kind") != "event":
                continue
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate event id: {row['id']!r}")
            seen_ids.add(row["id"])
            start_frame = max(0, round(float(row["start_time"]) * fps))
            end_frame = min(int(row["frame_count"]), round(float(row["end_time"]) * fps))
            if end_frame <= start_frame:
                raise ValueError(f"Invalid event frame range for {row['id']!r}")
            groups[str(row["motion"])].append(
                {
                    **row,
                    "motion_key": key,
                    "event_index": _event_index(row),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )
    for motion, events in groups.items():
        events.sort(key=lambda row: row["event_index"])
        indices = [row["event_index"] for row in events]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Motion has repeated event indices: {motion}")
    counts["event_motions"] = len(groups)
    return groups, counts


def _enumerate_candidates(groups, *, fps: int, max_frames: int, max_gap_frames: int):
    result = {count: [] for count in range(2, 6)}
    for motion, events in groups.items():
        for start in range(len(events)):
            for event_count in range(2, 6):
                selected = events[start : start + event_count]
                if len(selected) != event_count:
                    break
                if any(right["event_index"] != left["event_index"] + 1 for left, right in pairwise(selected)):
                    continue
                if any(right["start_frame"] - left["end_frame"] > max_gap_frames for left, right in pairwise(selected)):
                    continue
                start_frame = selected[0]["start_frame"]
                end_frame = selected[-1]["end_frame"]
                if end_frame - start_frame > max_frames:
                    continue
                source_event_ids = [row["id"] for row in selected]
                source_texts = [str(row["text"]).strip() for row in selected]
                identity = {
                    "motion_key": selected[0]["motion_key"],
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "source_event_ids": source_event_ids,
                }
                request_id = _canonical_hash({"ordered_source_texts": source_texts})
                result[event_count].append(
                    {
                        "id": f"v2multi:{_canonical_hash(identity)[:24]}",
                        "motion": motion,
                        "motion_key": selected[0]["motion_key"],
                        "split": "train",
                        "source_fps": float(fps),
                        "frame_count": int(selected[0]["frame_count"]),
                        "start_time": start_frame / fps,
                        "end_time": end_frame / fps,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "event_count": event_count,
                        "source_event_ids": source_event_ids,
                        "source_time_ranges": [[row["start_frame"], row["end_frame"]] for row in selected],
                        "source_texts": source_texts,
                        "qwen_request_id": request_id,
                    }
                )
    return result


def _select(candidates: dict[int, list[dict]]) -> tuple[list[dict], dict[int, int]]:
    base = len(candidates[2])
    targets = {2: base}
    for count in range(3, 6):
        proportional = round(base * BENCHMARK_MULTI_PROPORTIONS[count] / BENCHMARK_MULTI_PROPORTIONS[2])
        targets[count] = min(len(candidates[count]), proportional)
    selected = []
    for count in range(2, 6):
        # Maximize semantic coverage before taking repeated source-text tuples.
        # This keeps frequent propagated annotations from crowding rare actions
        # out of the fixed benchmark-shaped row budget.
        by_request: dict[str, list[dict]] = defaultdict(list)
        for row in candidates[count]:
            by_request[row["qwen_request_id"]].append(row)
        for rows in by_request.values():
            rows.sort(key=lambda row: _canonical_hash(row["id"]))
        request_ids = sorted(by_request)
        ordered = []
        depth = 0
        while len(ordered) < len(candidates[count]):
            added = False
            for request_id in request_ids:
                rows = by_request[request_id]
                if depth < len(rows):
                    ordered.append(rows[depth])
                    added = True
            if not added:
                break
            depth += 1
        selected.extend(ordered[: targets[count]])
    selected.sort(key=lambda row: (row["motion_key"], row["start_frame"], row["event_count"]))
    return selected, targets


def prepare(args) -> dict:
    source = Path(args.source_manifest).expanduser().resolve()
    split = Path(args.train_split).expanduser().resolve()
    plan = Path(args.output_plan).expanduser().resolve()
    requests = Path(args.output_requests).expanduser().resolve()
    sidecar = plan.with_suffix(plan.suffix + ".metadata.json")
    requests_sidecar = requests.with_suffix(requests.suffix + ".metadata.json")
    existing = [path for path in (plan, requests, sidecar, requests_sidecar) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite V2 preparation outputs: {existing}")
    plan.parent.mkdir(parents=True, exist_ok=True)
    requests.parent.mkdir(parents=True, exist_ok=True)
    train_keys = _split_keys(split)
    groups, source_counts = _load_events(source, train_keys, args.fps)
    candidates = _enumerate_candidates(
        groups,
        fps=args.fps,
        max_frames=round(args.max_seconds * args.fps),
        max_gap_frames=round(args.max_gap_seconds * args.fps),
    )
    selected, targets = _select(candidates)
    unique_requests = {}
    with plan.open("x", encoding="utf-8") as output:
        for row in selected:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            unique_requests.setdefault(
                row["qwen_request_id"],
                {
                    "request_id": row["qwen_request_id"],
                    "source_texts": row["source_texts"],
                    "prompt_version": PROMPT_VERSION,
                    "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                },
            )
    with requests.open("x", encoding="utf-8") as output:
        for request_id in sorted(unique_requests):
            output.write(json.dumps(unique_requests[request_id], ensure_ascii=False, sort_keys=True) + "\n")
    publish_file(plan)
    publish_file(requests)
    metadata = {
        "schema_version": 1,
        "builder": "kimodo.training.timeline_multi_cli",
        "algorithm": {
            "fps": args.fps,
            "time_to_frame": "python_round_then_clip_end_to_frame_count",
            "event_counts": [2, 3, 4, 5],
            "maximum_frames": round(args.max_seconds * args.fps),
            "maximum_gap_frames": round(args.max_gap_seconds * args.fps),
            "selection": "semantic_request_round_robin_with_benchmark_event_count_targets",
            "target_proportions": BENCHMARK_MULTI_PROPORTIONS,
        },
        "source_manifest": {"path": str(source), "sha256": _sha256_file(source)},
        "official_train_split": {"path": str(split), "sha256": _sha256_file(split), "entries": len(train_keys)},
        "prompt": {"version": PROMPT_VERSION, "sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()},
        "counts": {
            **dict(source_counts),
            "candidates": {str(k): len(v) for k, v in candidates.items()},
            "selected": {str(k): targets[k] for k in targets},
            "selected_total": len(selected),
            "unique_qwen_requests": len(unique_requests),
        },
        "outputs": {
            "plan": {"path": str(plan), "sha256": _sha256_file(plan)},
            "requests": {"path": str(requests), "sha256": _sha256_file(requests)},
        },
        "leakage_policy": "inputs_limited_to_v1_train_manifest_and_pinned_official_train_whitelist",
    }
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    publish_file(sidecar)
    requests_sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "output": metadata["outputs"]["requests"],
                "prompt": metadata["prompt"],
                "source_plan_metadata_sha256": _sha256_file(sidecar),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    publish_file(requests_sidecar)
    print(json.dumps(metadata["counts"], indent=2, sort_keys=True))
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--output-plan", required=True)
    parser.add_argument("--output-requests", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--max-gap-seconds", type=float, default=1.5)
    return parser


def main() -> None:
    prepare(build_parser().parse_args())


if __name__ == "__main__":
    main()
