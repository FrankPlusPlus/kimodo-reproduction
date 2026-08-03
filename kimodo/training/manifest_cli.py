# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a leakage-safe Kimodo JSONL manifest from BONES-SEED metadata.

This builder deliberately does not pretend to recreate the paper's unavailable
Qwen3-32B paraphrases or cross-motion transition clips.  Every emitted row and
the sidecar metadata identify what was actually constructed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


TEXT_COLUMNS = (
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
    "content_technical_description",
    "content_short_description",
    "content_short_description_2",
)
PATH_COLUMNS = {
    "soma_uniform": "move_soma_uniform_path",
    "soma_proportional": "move_soma_proportional_path",
    "g1": "move_g1_mujoco_path",
}


def _read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError as error:
            raise RuntimeError("Parquet metadata requires pandas+pyarrow; use the CSV release otherwise") from error
        return pd.read_parquet(path).fillna("").to_dict(orient="records")
    raise ValueError("--metadata must be .csv or .parquet")


def _read_temporal(path: Path | None) -> dict[str, list[dict]]:
    if path is None:
        return {}
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                result[str(record["filename"])] = list(record.get("events", []))
    return result


def _split_keys(path: Path) -> set[str]:
    return {line.strip().replace("\\", "/").removesuffix(".bvh").removesuffix(".csv") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _row_key(row: dict, motion_value: str) -> str:
    filename = str(row.get("filename") or Path(motion_value).stem)
    normalized = motion_value.replace("\\", "/")
    parts = Path(normalized).parts
    for marker in ("bvh", "csv", "BVH", "CSV"):
        if marker in parts:
            tail = list(parts[parts.index(marker) + 1 :])
            if tail:
                tail[-1] = Path(tail[-1]).stem
                return "/".join(tail)
    date = str(row.get("take_date", "")).strip()
    return f"{date}/{filename}" if date else filename


def _clean_text(value) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "0"}:
        return None
    return text


def _emit(record: dict, output) -> None:
    output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(path: Path | None) -> dict | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size": resolved.stat().st_size,
    }


def build_manifest(args) -> dict[str, int]:
    metadata = Path(args.metadata).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    split_file = Path(args.split_file).expanduser().resolve()
    temporal_labels = (
        Path(args.temporal_labels).expanduser().resolve() if args.temporal_labels else None
    )
    split_keys = _split_keys(split_file)
    temporal = _read_temporal(
        temporal_labels
    )
    path_column = PATH_COLUMNS[args.skeleton]
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = {"motions": 0, "full": 0, "event": 0, "combined": 0, "missing": 0}
    with destination.open("x", encoding="utf-8") as output:
        for row in _read_rows(metadata):
            motion_value = _clean_text(row.get(path_column))
            if motion_value is None:
                continue
            key = _row_key(row, motion_value)
            if key not in split_keys:
                continue
            motion_path = Path(motion_value)
            if not motion_path.is_absolute():
                motion_path = dataset_root / motion_path
            if not motion_path.is_file():
                counts["missing"] += 1
                if not args.allow_missing:
                    raise FileNotFoundError(f"Metadata motion path is missing: {motion_path}")
                continue
            filename = str(row.get("filename") or motion_path.stem)
            counts["motions"] += 1
            descriptions = [
                (column, text)
                for column in TEXT_COLUMNS
                if (text := _clean_text(row.get(column)))
            ]
            for text_index, (text_column, text) in enumerate(descriptions):
                for repeat in range(args.full_repeats):
                    _emit(
                        {
                            "id": f"{filename}:full:{text_index}:{repeat}",
                            "motion": str(motion_path.resolve()),
                            "text": text,
                            "split": args.split_name,
                            "source_fps": args.source_fps,
                            "sample_kind": "full",
                            "text_source": f"bones_seed_metadata:{text_column}",
                            "augmentation_provenance": "dataset_annotation",
                        },
                        output,
                    )
                    counts["full"] += 1
            events = temporal.get(filename, [])
            for event_index, event in enumerate(events):
                text = _clean_text(event.get("description"))
                if text is None:
                    continue
                for repeat in range(args.event_repeats):
                    _emit(
                        {
                            "id": f"{filename}:event:{event_index}:{repeat}",
                            "motion": str(motion_path.resolve()),
                            "text": text,
                            "split": args.split_name,
                            "source_fps": args.source_fps,
                            "start_time": float(event["start_time"]),
                            "end_time": float(event["end_time"]),
                            "sample_kind": "event",
                            "text_source": "bones_seed_temporal_label",
                            "augmentation_provenance": "single_action_subclip",
                        },
                        output,
                    )
                    counts["event"] += 1
            for event_index in range(max(0, len(events) - 1)):
                first, second = events[event_index : event_index + 2]
                first_text = _clean_text(first.get("description"))
                second_text = _clean_text(second.get("description"))
                if first_text is None or second_text is None:
                    continue
                for repeat in range(args.combined_event_repeats):
                    _emit(
                        {
                            "id": f"{filename}:combined2:{event_index}:{repeat}",
                            "motion": str(motion_path.resolve()),
                            "text": f"{first_text} Then, {second_text}",
                            "split": args.split_name,
                            "source_fps": args.source_fps,
                            "start_time": float(first["start_time"]),
                            "end_time": float(second["end_time"]),
                            "sample_kind": "combined_events",
                            "text_source": "bones_seed_temporal_labels",
                            "augmentation_provenance": "adjacent_same_motion_events",
                        },
                        output,
                    )
                    counts["combined"] += 1
    metadata_record = {
        "schema_version": 1,
        "builder": "kimodo.training.manifest_cli",
        "dataset_root": str(dataset_root),
        "skeleton": args.skeleton,
        "split_name": args.split_name,
        "source_fps": args.source_fps,
        "repeat_counts": {
            "full": args.full_repeats,
            "event": args.event_repeats,
            "combined_event": args.combined_event_repeats,
        },
        "paper_data_recipe": {
            "full_motion_clips": "implemented_from_dataset_annotations",
            "single_action_subclips": (
                "implemented_from_external_temporal_labels"
                if temporal_labels is not None
                else "not_available_no_temporal_labels_supplied"
            ),
            "combined_action_subclips": (
                "implemented_as_adjacent_events_within_one_motion"
                if temporal_labels is not None
                else "not_available_no_temporal_labels_supplied"
            ),
            "qwen3_32b_paraphrases": "not_generated_external_asset_required",
            "random_cross_motion_stitching": "not_generated_external_asset_required",
            "diffusion_transition_clips": "not_generated_transition_model_required",
            "official_mixture_distribution": "not_disclosed_repeat_counts_are_reproduction_assumptions",
        },
        "paper_parity_gate": {
            "eligible": False,
            "status": "blocked_missing_external_augmentation_assets",
            "blockers": [
                "qwen3_32b_paraphrases",
                "random_cross_motion_stitching",
                "diffusion_transition_clips",
            ],
            "required_external_row_schema": {
                "llm_paraphrase": [
                    "id",
                    "motion",
                    "text",
                    "split",
                    "sample_kind=llm_paraphrase",
                    "source_text_id",
                    "text_generator_model",
                    "text_generator_prompt_sha256",
                ],
                "stitched_transition": [
                    "id",
                    "motion",
                    "text",
                    "split",
                    "sample_kind=stitched_transition",
                    "source_motion_ids",
                    "source_time_ranges",
                    "transition_model_sha256",
                    "transition_frame_range",
                ],
            },
        },
        "sources": {
            "metadata": _source_record(metadata),
            "split_file": _source_record(split_file),
            "temporal_labels": _source_record(temporal_labels),
        },
        "output": {"path": str(destination), "sha256": _sha256(destination)},
        "counts": counts,
    }
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata_record, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output": str(destination), **counts}, indent=2, sort_keys=True))
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--temporal-labels")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--skeleton", choices=tuple(PATH_COLUMNS), default="soma_uniform")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--source-fps", type=float, default=120.0)
    parser.add_argument("--full-repeats", type=int, default=1)
    parser.add_argument("--event-repeats", type=int, default=1)
    parser.add_argument("--combined-event-repeats", type=int, default=1)
    parser.add_argument("--allow-missing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ("full_repeats", "event_repeats", "combined_event_repeats"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    build_manifest(args)


if __name__ == "__main__":
    main()
