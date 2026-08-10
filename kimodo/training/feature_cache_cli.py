# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline builder for the motion feature cache used by training."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton
from kimodo.training.data import load_manifest
from kimodo.training.feature_cache import (
    INDEX_NAME,
    assert_cache_fingerprint,
    build_cache_fingerprint,
    feature_relpath,
    load_feature_array,
    load_index,
    load_meta,
    materialize_entry_features,
    save_feature_array,
    write_meta,
)

_WORKER_STATE: dict | None = None


def _initialize_worker(skeleton_joints: int, fps: int, min_frames: int) -> None:
    global _WORKER_STATE
    torch.set_num_threads(1)
    skeleton = build_skeleton(skeleton_joints)
    _WORKER_STATE = {
        "motion_rep": KimodoMotionRep(skeleton=skeleton, fps=fps, stats_path=None),
        "min_frames": min_frames,
    }


def _build_one(task: dict) -> dict:
    if _WORKER_STATE is None:
        raise RuntimeError("Feature-cache worker was not initialized")
    entry = task["entry"]
    output = Path(task["output"])
    relpath = feature_relpath(entry.sample_id)
    path = output / relpath
    if path.is_file() and not task["overwrite"]:
        array = load_feature_array(path)
        return {
            "id": entry.sample_id,
            "path": relpath,
            "frames": int(array.shape[0]),
            "feature_dim": int(array.shape[1]),
            "skipped": True,
        }
    features = materialize_entry_features(
        entry, _WORKER_STATE["motion_rep"], min_frames=_WORKER_STATE["min_frames"]
    )
    save_feature_array(path, features)
    return {
        "id": entry.sample_id,
        "path": relpath,
        "frames": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "skipped": False,
    }


def _verify_samples(
    *,
    cache_dir: Path,
    index: dict[str, dict],
    entries_by_id: dict,
    motion_rep: KimodoMotionRep,
    min_frames: int,
    sample_count: int,
    seed: int,
) -> None:
    if sample_count <= 0:
        return
    ids = sorted(index)
    generator = np.random.default_rng(seed)
    chosen = generator.choice(ids, size=min(sample_count, len(ids)), replace=False)
    for sample_id in chosen:
        entry = entries_by_id[str(sample_id)]
        expected = materialize_entry_features(entry, motion_rep, min_frames=min_frames)
        cached = np.array(load_feature_array(cache_dir / index[str(sample_id)]["path"]), copy=True)
        if cached.shape != tuple(expected.shape):
            raise ValueError(
                f"Verify shape mismatch for {sample_id!r}: cache={cached.shape} live={tuple(expected.shape)}"
            )
        # float16 round-trip; allow tiny absolute error after cast.
        live = expected.detach().cpu().numpy().astype(np.float16).astype(np.float32)
        if not np.allclose(cached.astype(np.float32), live, rtol=0.0, atol=2e-3):
            max_abs = float(np.max(np.abs(cached.astype(np.float32) - live)))
            raise ValueError(f"Verify value mismatch for {sample_id!r}: max_abs={max_abs}")


def build_feature_cache(args: argparse.Namespace) -> Path:
    manifest = Path(args.manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    stats_path = Path(args.stats_path).expanduser().resolve() if args.stats_path else None
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if stats_path is not None and not stats_path.is_dir():
        raise FileNotFoundError(f"Stats path not found: {stats_path}")

    entries = load_manifest(manifest, args.split)
    entries_by_id = {entry.sample_id: entry for entry in entries}
    skeleton = build_skeleton(args.skeleton_joints)
    motion_rep = KimodoMotionRep(skeleton=skeleton, fps=args.fps, stats_path=None)
    fingerprint = build_cache_fingerprint(
        fps=args.fps,
        feature_dim=motion_rep.motion_rep_dim,
        skeleton_joints=args.skeleton_joints,
        stats_path=stats_path,
    )

    if output.exists() and any(output.iterdir()) and not args.overwrite and not (output / INDEX_NAME).is_file():
        raise FileExistsError(
            f"Output directory is non-empty and has no index: {output}. Pass --overwrite to rebuild."
        )
    output.mkdir(parents=True, exist_ok=True)

    existing_index: dict[str, dict] = {}
    if (output / INDEX_NAME).is_file() and not args.overwrite:
        meta = load_meta(output)
        assert_cache_fingerprint(meta, fingerprint)
        existing_index = load_index(output)

    tasks = [
        {
            "entry": entry,
            "output": str(output),
            "overwrite": bool(args.overwrite),
        }
        for entry in entries
        if args.overwrite or entry.sample_id not in existing_index
    ]
    # Always refresh rows for entries already present when not overwriting files.
    refresh_only = [
        entry for entry in entries if (not args.overwrite) and entry.sample_id in existing_index
    ]

    num_workers = max(1, int(args.num_workers))
    started = time.perf_counter()
    built_rows: list[dict] = []
    if num_workers == 1:
        _initialize_worker(args.skeleton_joints, args.fps, args.min_frames)
        results = map(_build_one, tasks)
        for index, result in enumerate(results, start=1):
            built_rows.append(result)
            if index % 500 == 0 or index == len(tasks):
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed else 0.0
                print(
                    f"Feature cache: {index}/{len(tasks)} built/skipped, {rate:.2f}/s",
                    flush=True,
                )
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_initialize_worker,
            initargs=(args.skeleton_joints, args.fps, args.min_frames),
        ) as executor:
            for index, result in enumerate(
                executor.map(_build_one, tasks, chunksize=8), start=1
            ):
                built_rows.append(result)
                if index % 500 == 0 or index == len(tasks):
                    elapsed = time.perf_counter() - started
                    rate = index / elapsed if elapsed else 0.0
                    print(
                        f"Feature cache: {index}/{len(tasks)} built/skipped, {rate:.2f}/s",
                        flush=True,
                    )

    index_rows = {row["id"]: row for row in built_rows}
    for entry in refresh_only:
        row = existing_index[entry.sample_id]
        path = output / row["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Indexed feature file missing: {path}")
        array = load_feature_array(path)
        index_rows[entry.sample_id] = {
            "id": entry.sample_id,
            "path": row["path"],
            "frames": int(array.shape[0]),
            "feature_dim": int(array.shape[1]),
            "skipped": True,
        }

    # Preserve manifest order in the index for stable diffs.
    ordered = [index_rows[entry.sample_id] for entry in entries]
    for row in ordered:
        if int(row["feature_dim"]) != int(fingerprint["feature_dim"]):
            raise ValueError(
                f"Feature dim mismatch for {row['id']!r}: "
                f"{row['feature_dim']} != {fingerprint['feature_dim']}"
            )

    index_path = output / INDEX_NAME
    tmp_index = index_path.with_suffix(".tmp")
    with tmp_index.open("w", encoding="utf-8") as handle:
        for row in ordered:
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
    write_meta(output, fingerprint, entry_count=len(ordered))

    verify_rep = KimodoMotionRep(skeleton=skeleton, fps=args.fps, stats_path=None)
    _verify_samples(
        cache_dir=output,
        index={row["id"]: row for row in ordered},
        entries_by_id=entries_by_id,
        motion_rep=verify_rep,
        min_frames=args.min_frames,
        sample_count=args.verify_sample,
        seed=args.seed,
    )
    written = sum(1 for row in built_rows if not row.get("skipped"))
    skipped = len(ordered) - written
    print(
        f"Saved feature cache to {output} ({len(ordered)} entries, "
        f"{written} written, {skipped} reused)",
        flush=True,
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Train-split JSONL manifest")
    parser.add_argument("--output", required=True, help="Feature-cache directory")
    parser.add_argument(
        "--stats-path",
        default=None,
        help="Normalization stats directory used for fingerprinting (not applied offline)",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--skeleton-joints", type=int, default=30, choices=(22, 30, 34, 77))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-sample", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main() -> None:
    build_feature_cache(build_parser().parse_args())


if __name__ == "__main__":
    main()
