#!/usr/bin/env python3
"""Sharded motion-feature cache builder for high-throughput JuiceFS runs.

Avoids loading the full 1.4M manifest into one ProcessPool parent (IPC/fork stall).
Each shard process loads only its JSONL slice and writes features + a shard index.
A final merge step assembles index.jsonl + meta.json in manifest order.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton
from kimodo.training.data import ManifestEntry, _resolve_path
from kimodo.training.feature_cache import (
    INDEX_NAME,
    META_NAME,
    build_cache_fingerprint,
    feature_relpath,
    load_feature_array,
    materialize_entry_features,
    save_feature_array,
    write_meta,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="Full train JSONL (merge) or shard JSONL (build)")
    p.add_argument(
        "--path-base",
        default=None,
        help="Directory used to resolve relative motion/embedding paths "
        "(defaults to manifest parent). Required when --manifest is a split shard.",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--stats-path", required=True)
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--skeleton-joints", type=int, default=30)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--min-frames", type=int, default=2)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--work-root",
        default=None,
        help="Optional local fast disk (e.g. /tmp). Features land here then "
        "are hardlinked/copied into --output as each file completes.",
    )
    return p.parse_args()


def _load_shard_entries(
    manifest: Path,
    *,
    path_base: Path,
    split: str,
) -> list[ManifestEntry]:
    """Parse a pre-split shard JSONL (already contains only this shard's rows)."""
    entries: list[ManifestEntry] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if str(raw.get("split", split)) != split:
                continue
            motion_path = _resolve_path(path_base, raw["motion"])
            assert motion_path is not None
            entries.append(
                ManifestEntry(
                    sample_id=str(raw["id"]),
                    motion_path=motion_path,
                    text=str(raw.get("text", "")),
                    split=str(raw["split"]),
                    source_fps=raw.get("source_fps"),
                    text_embedding_path=_resolve_path(path_base, raw.get("text_embedding")),
                    start_time=raw.get("start_time"),
                    end_time=raw.get("end_time"),
                    sample_kind=str(raw.get("sample_kind", "full")),
                    mixture_source=str(raw.get("mixture_source", "base")),
                    event_count=raw.get("event_count"),
                    frame_count=raw.get("frame_count"),
                )
            )
    return entries


def run_shard(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    manifest = Path(args.manifest).expanduser().resolve()
    path_base = (
        Path(args.path_base).expanduser().resolve()
        if args.path_base
        else manifest.parent
    )
    output = Path(args.output).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve() if args.work_root else output
    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "features").mkdir(parents=True, exist_ok=True)
    (output / "features").mkdir(parents=True, exist_ok=True)

    entries = _load_shard_entries(manifest, path_base=path_base, split=args.split)
    skeleton = build_skeleton(args.skeleton_joints)
    motion_rep = KimodoMotionRep(skeleton=skeleton, fps=args.fps, stats_path=None)

    started = time.perf_counter()
    rows: list[dict] = []
    written = 0
    skipped = 0
    for index, entry in enumerate(entries, start=1):
        relpath = feature_relpath(entry.sample_id)
        final_path = output / relpath
        work_path = work_root / relpath
        if final_path.is_file() and not args.overwrite:
            array = load_feature_array(final_path)
            rows.append(
                {
                    "id": entry.sample_id,
                    "path": relpath,
                    "frames": int(array.shape[0]),
                    "feature_dim": int(array.shape[1]),
                }
            )
            skipped += 1
        else:
            features = materialize_entry_features(
                entry, motion_rep, min_frames=args.min_frames
            )
            work_path.parent.mkdir(parents=True, exist_ok=True)
            save_feature_array(work_path, features)
            if work_path != final_path:
                final_path.parent.mkdir(parents=True, exist_ok=True)
                if final_path.exists():
                    final_path.unlink()
                try:
                    os.link(work_path, final_path)
                except OSError:
                    # Cross-device: fall back to replace copy via temp.
                    tmp = final_path.with_suffix(final_path.suffix + ".copy")
                    tmp.write_bytes(work_path.read_bytes())
                    tmp.replace(final_path)
                    work_path.unlink(missing_ok=True)
            rows.append(
                {
                    "id": entry.sample_id,
                    "path": relpath,
                    "frames": int(features.shape[0]),
                    "feature_dim": int(features.shape[1]),
                }
            )
            written += 1
        if index % 100 == 0 or index == len(entries):
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0.0
            print(
                f"[shard {args.shard_id}/{args.num_shards}] {index}/{len(entries)} "
                f"({written} new, {skipped} skip) {rate:.2f}/s",
                flush=True,
            )

    shard_index = shard_dir / f"index.{args.shard_id:04d}.jsonl"
    tmp = shard_index.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(shard_index)
    print(
        f"[shard {args.shard_id}/{args.num_shards}] DONE rows={len(rows)} "
        f"written={written} skipped={skipped}",
        flush=True,
    )


def merge_shards(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    stats_path = Path(args.stats_path).expanduser().resolve()
    shard_dir = output / "shards"
    # Load ids in manifest order (stream; no heavy validation).
    ordered_ids: list[str] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if str(raw.get("split", args.split)) != args.split:
                continue
            ordered_ids.append(str(raw["id"]))

    index: dict[str, dict] = {}
    for path in sorted(shard_dir.glob("index.*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                index[str(row["id"])] = row

    missing = [sample_id for sample_id in ordered_ids if sample_id not in index]
    if missing:
        raise SystemExit(
            f"Merge incomplete: missing {len(missing)} ids "
            f"(example {missing[:3]})"
        )

    skeleton = build_skeleton(args.skeleton_joints)
    motion_rep = KimodoMotionRep(skeleton=skeleton, fps=args.fps, stats_path=None)
    fingerprint = build_cache_fingerprint(
        fps=args.fps,
        feature_dim=motion_rep.motion_rep_dim,
        skeleton_joints=args.skeleton_joints,
        stats_path=stats_path,
    )

    index_path = output / INDEX_NAME
    tmp_index = index_path.with_suffix(".tmp")
    with tmp_index.open("w", encoding="utf-8") as handle:
        for sample_id in ordered_ids:
            row = index[sample_id]
            if int(row["feature_dim"]) != int(fingerprint["feature_dim"]):
                raise SystemExit(
                    f"Feature dim mismatch for {sample_id}: "
                    f"{row['feature_dim']} != {fingerprint['feature_dim']}"
                )
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "path": row["path"],
                        "frames": int(row["frames"]),
                        "feature_dim": int(row["feature_dim"]),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    tmp_index.replace(index_path)
    write_meta(output, fingerprint, entry_count=len(ordered_ids))
    print(
        f"Merged {len(ordered_ids)} rows into {index_path} (+ {META_NAME})",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    # Reuse --shard-id=-1 as merge mode to keep one entrypoint.
    if args.shard_id < 0:
        merge_shards(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
