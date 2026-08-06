# SPDX-License-Identifier: Apache-2.0
"""Build a deterministic, zero-copy, metadata-stratified motion subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .file_permissions import publish_file

SCHEMA_VERSION = 1
_PATH_FIELDS = ("motion", "text_embedding", "text_embedding_metadata")
_STRATIFY_FIELDS = (
    "package",
    "category",
    "content_all_rigplay_styles",
    "content_uniform_style",
    "content_type_of_movement",
    "content_body_position",
    "content_horizontal_move",
    "content_vertical_move",
    "content_props",
    "content_complex_action",
    "content_repeated_action",
    "actor_gender",
)


@dataclass(frozen=True)
class Candidate:
    motion: str
    frame_count: int
    duration_seconds: float
    take_group: str
    tags: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".metadata.json")


def _portable(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _stable_unit(seed: int, *parts: str) -> float:
    payload = "\x1f".join((str(seed), *parts)).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (value + 1) / (2**64 + 1)


def _normal(value: object) -> str:
    text = str(value or "unknown").strip().lower()
    return " ".join(text.split()) or "unknown"


def _duration_bin(seconds: float) -> str:
    if seconds < 4:
        return "short_lt4"
    if seconds < 8:
        return "medium_4_8"
    if seconds < 15:
        return "long_8_15"
    return "very_long_ge15"


def _manifest_motion_index(manifest: Path, split: str) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split", "")) != split:
                continue
            missing = {"motion", "frame_count"} - row.keys()
            if missing:
                raise ValueError(f"{manifest}:{line_number} missing {sorted(missing)}")
            motion = str(row["motion"])
            frame_count = int(row["frame_count"])
            if frame_count < 2:
                raise ValueError(f"{manifest}:{line_number} has invalid frame_count={frame_count}")
            stem = Path(motion).stem
            previous = result.setdefault(stem, (frame_count, motion))
            if previous != (frame_count, motion):
                raise ValueError(f"Motion stem {stem!r} is ambiguous in {manifest}")
    if not result:
        raise ValueError(f"No split={split!r} motions found in {manifest}")
    return result


def _metadata_candidates(
    metadata_csv: Path,
    motion_index: dict[str, tuple[int, str]],
    fps: int,
) -> tuple[list[Candidate], dict[str, int]]:
    candidates: list[Candidate] = []
    counters = Counter()
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = {"filename", "is_mirror", "take_name"} - set(
            reader.fieldnames or ()
        )
        if missing_columns:
            raise ValueError(f"Metadata CSV lacks columns: {sorted(missing_columns)}")
        for row in reader:
            # Converted NPZs retain the source ``filename`` stem. ``move_name``
            # is a semantic/internal identifier and differs for older sessions.
            filename = str(row.get("filename", ""))
            indexed = motion_index.get(filename)
            if indexed is None:
                continue
            counters["metadata_matches"] += 1
            if _normal(row.get("is_mirror")) in {"true", "1", "1.0", "yes"}:
                counters["mirrors_excluded"] += 1
                continue
            frame_count, motion = indexed
            duration = frame_count / float(fps)
            tags = [f"{field}={_normal(row.get(field))}" for field in _STRATIFY_FIELDS]
            tags.append(f"duration={_duration_bin(duration)}")
            take = _normal(row.get("take_name"))
            candidates.append(
                Candidate(
                    motion=motion,
                    frame_count=frame_count,
                    duration_seconds=duration,
                    take_group=f"{take}",
                    tags=tuple(tags),
                )
            )
    if not candidates:
        raise ValueError("No non-mirrored metadata rows matched the training manifest")
    counters["non_mirror_candidates"] = len(candidates)
    counters["manifest_motion_count"] = len(motion_index)
    counters["take_group_count"] = len({candidate.take_group for candidate in candidates})
    return candidates, dict(counters)


def _select_candidates(candidates: list[Candidate], target_seconds: float, seed: int) -> list[Candidate]:
    frequencies = Counter(tag for candidate in candidates for tag in candidate.tags)
    by_tag: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        for tag in candidate.tags:
            by_tag[tag].append(candidate)

    selected: dict[str, Candidate] = {}
    elapsed = 0.0
    # First guarantee at least one example of every metadata value that fits.
    for tag in sorted(frequencies, key=lambda value: (frequencies[value], value)):
        if any(candidate.motion in selected for candidate in by_tag[tag]):
            continue
        choices = sorted(
            by_tag[tag], key=lambda candidate: _stable_unit(seed, "coverage", tag, candidate.motion)
        )
        choice = next(
            (
                candidate
                for candidate in choices
                if elapsed + candidate.duration_seconds <= target_seconds
            ),
            None,
        )
        if choice is not None:
            selected[choice.motion] = choice
            elapsed += choice.duration_seconds

    def priority(candidate: Candidate) -> tuple[float, str]:
        rarity = sum(1.0 / math.sqrt(frequencies[tag]) for tag in candidate.tags)
        # Exponential-race weighted sampling: deterministic, without replacement,
        # and biased toward under-represented metadata values without allowing one
        # rare tag to dominate the complete subset.
        score = -math.log(_stable_unit(seed, "fill", candidate.motion)) / (1.0 + rarity)
        return score, candidate.motion

    remaining = sorted(
        (candidate for candidate in candidates if candidate.motion not in selected),
        key=priority,
    )
    for candidate in remaining:
        if elapsed >= target_seconds:
            break
        selected[candidate.motion] = candidate
        elapsed += candidate.duration_seconds
    return list(selected.values())


def _validation_groups(
    selected: list[Candidate], validation_fraction: float, seed: int
) -> set[str]:
    durations = Counter()
    for candidate in selected:
        durations[candidate.take_group] += candidate.duration_seconds
    target = sum(durations.values()) * validation_fraction
    chosen: set[str] = set()
    elapsed = 0.0
    for group in sorted(durations, key=lambda value: _stable_unit(seed, "validation", value)):
        if elapsed >= target:
            break
        chosen.add(group)
        elapsed += durations[group]
    return chosen


def build_core_subset(args) -> dict:
    source = Path(args.source_manifest).expanduser().resolve()
    metadata_csv = Path(args.metadata_csv).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output_metadata = _sidecar(output)
    if not source.is_file() or not metadata_csv.is_file():
        raise FileNotFoundError(source if not source.is_file() else metadata_csv)
    if output.exists() or output_metadata.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {output_metadata}")
    fps = int(args.fps)
    target_seconds = float(args.target_hours) * 3600.0
    validation_fraction = float(args.validation_fraction)
    if fps <= 0 or target_seconds <= 0:
        raise ValueError("fps and target_hours must be positive")
    if not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in [0, 0.5)")

    motion_index = _manifest_motion_index(source, args.source_split)
    candidates, input_counts = _metadata_candidates(metadata_csv, motion_index, fps)
    selected = _select_candidates(candidates, target_seconds, int(args.seed))
    selected_by_motion = {candidate.motion: candidate for candidate in selected}
    validation_groups = _validation_groups(selected, validation_fraction, int(args.seed))

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent
    )
    row_counts = Counter()
    seen_ids: set[str] = set()
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target, source.open(
            encoding="utf-8"
        ) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                candidate = selected_by_motion.get(str(row.get("motion", "")))
                if candidate is None or str(row.get("split", "")) != args.source_split:
                    continue
                rewritten = dict(row)
                split = "validation" if candidate.take_group in validation_groups else "train"
                rewritten["split"] = split
                rewritten["core_subset"] = args.name
                for field in _PATH_FIELDS:
                    value = rewritten.get(field)
                    if value is None:
                        continue
                    path = Path(str(value)).expanduser()
                    if not path.is_absolute():
                        path = source.parent / path
                    rewritten[field] = _portable(path, output.parent)
                sample_id = str(rewritten.get("id", ""))
                if not sample_id or sample_id in seen_ids:
                    raise ValueError(f"{source}:{line_number} has a missing/duplicate id")
                seen_ids.add(sample_id)
                target.write(json.dumps(rewritten, ensure_ascii=False, sort_keys=True) + "\n")
                row_counts[f"rows/{split}"] += 1
                row_counts[f"sample_kind/{rewritten.get('sample_kind', 'full')}"] += 1
        publish_file(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    duration_by_split = Counter()
    motion_count_by_split = Counter()
    tag_selected = Counter()
    for candidate in selected:
        split = "validation" if candidate.take_group in validation_groups else "train"
        duration_by_split[split] += candidate.duration_seconds
        motion_count_by_split[split] += 1
        tag_selected.update(candidate.tags)
    source_metadata = _sidecar(source)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "builder": "kimodo.training.core_subset_cli",
        "name": args.name,
        "seed": int(args.seed),
        "fps": fps,
        "source_split": args.source_split,
        "target_hours": float(args.target_hours),
        "selection_policy": {
            "mirror_policy": "exclude metadata rows marked is_mirror",
            "coverage": "one candidate per available metadata tag before weighted fill",
            "fill": "deterministic inverse-sqrt-frequency weighted sampling without replacement",
            "validation_group": "take_name",
            "validation_fraction_target": validation_fraction,
            "stratify_fields": list(_STRATIFY_FIELDS),
        },
        "source_manifest": _portable(source, output.parent),
        "source_manifest_sha256": _sha256(source),
        "source_manifest_metadata": (
            _portable(source_metadata, output.parent) if source_metadata.is_file() else None
        ),
        "source_manifest_metadata_sha256": (
            _sha256(source_metadata) if source_metadata.is_file() else None
        ),
        "metadata_csv": _portable(metadata_csv, output.parent),
        "metadata_csv_sha256": _sha256(metadata_csv),
        "input_counts": input_counts,
        "selected": {
            "motions": len(selected),
            "hours": sum(candidate.duration_seconds for candidate in selected) / 3600.0,
            "motions_by_split": dict(sorted(motion_count_by_split.items())),
            "hours_by_split": {
                key: value / 3600.0 for key, value in sorted(duration_by_split.items())
            },
            "rows": dict(sorted(row_counts.items())),
            "tag_coverage": len(tag_selected),
        },
        "producer": {
            "path": "kimodo/training/core_subset_cli.py",
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "output": {
            "path": output.name,
            "sha256": _sha256(output),
            "entries": sum(value for key, value in row_counts.items() if key.startswith("rows/")),
        },
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output_metadata.name + ".", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        publish_file(temporary)
        os.replace(temporary, output_metadata)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="core10-v1")
    parser.add_argument("--target-hours", type=float, default=10.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main() -> None:
    print(json.dumps(build_core_subset(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
