# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute split normalization statistics for Kimodo motion features."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton

from .data import (
    _convert_rotations_to_model_skeleton,
    _load_training_motion_file,
    load_manifest,
)


class OnlineMoments:
    def __init__(self, dimension: int) -> None:
        self.count = 0
        self.sum = torch.zeros(dimension, dtype=torch.float64)
        self.sum_sq = torch.zeros(dimension, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        values = values.reshape(-1, values.shape[-1]).double()
        self.count += len(values)
        self.sum += values.sum(0)
        self.sum_sq += (values * values).sum(0)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError("At least two frames are required to compute statistics")
        mean = self.sum / self.count
        variance = (self.sum_sq / self.count - mean.square()).clamp_min(0.0)
        return mean.float().numpy(), variance.sqrt().float().numpy()

    def merge(self, count: int, total: np.ndarray, total_sq: np.ndarray) -> None:
        self.count += int(count)
        self.sum += torch.from_numpy(total)
        self.sum_sq += torch.from_numpy(total_sq)


def _covering_windows(length: int, maximum: int) -> list[tuple[int, int]]:
    """Partition all frames into deterministic non-overlapping windows.

    A one-frame tail cannot be converted to Kimodo velocities.  In that one
    case, one frame is moved from the preceding maximum-size window so both
    final windows contain at least two frames.
    """
    if length < 2:
        raise ValueError("At least two frames are required per source motion/span")
    if maximum < 3:
        raise ValueError("Stats window size must be at least three frames")
    windows: list[tuple[int, int]] = []
    start = 0
    while length - start > maximum:
        size = maximum
        if length - (start + size) == 1:
            size -= 1
        windows.append((start, start + size))
        start += size
    windows.append((start, length))
    return windows


def _window_seed(base_seed: int, key: tuple, window_index: int) -> int:
    payload = json.dumps([base_seed, *map(str, key), window_index], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


_STATS_WORKER_STATE = None


def _initialize_stats_worker(
    skeleton_joints: int, fps: int, max_frames: int, min_frames: int, seed: int
) -> None:
    global _STATS_WORKER_STATE
    # Each process handles independent motions.  Prevent PyTorch's intra-op
    # pool from multiplying the requested process count on a shared server.
    torch.set_num_threads(1)
    skeleton = build_skeleton(skeleton_joints)
    _STATS_WORKER_STATE = {
        "skeleton": skeleton,
        "motion_rep": KimodoMotionRep(skeleton=skeleton, fps=fps, stats_path=None),
        "fps": fps,
        "max_frames": max_frames,
        "min_frames": min_frames,
        "seed": seed,
    }


def _compute_motion_group(task) -> dict:
    if _STATS_WORKER_STATE is None:
        raise RuntimeError("Stats worker was not initialized")
    motion_path, motion_entries = task
    skeleton = _STATS_WORKER_STATE["skeleton"]
    motion_rep = _STATS_WORKER_STATE["motion_rep"]
    fps = _STATS_WORKER_STATE["fps"]
    max_frames = _STATS_WORKER_STATE["max_frames"]
    min_frames = _STATS_WORKER_STATE["min_frames"]
    seed = _STATS_WORKER_STATE["seed"]

    first_entry = motion_entries[0]
    motion, source_joints = _load_training_motion_file(
        motion_path, first_entry.source_fps, float(fps)
    )
    local_rotations = _convert_rotations_to_model_skeleton(
        motion["local_rot_mats"].float(), source_joints, skeleton
    )
    root_positions = motion["root_positions"].float()
    source_length = int(local_rotations.shape[0])
    moments = (
        OnlineMoments(motion_rep.global_root_dim),
        OnlineMoments(motion_rep.local_root_dim),
        OnlineMoments(motion_rep.body_dim),
    )
    window_count = 0
    window_min = None
    window_max = 0
    span_count = 0
    excluded_short_spans = 0
    for entry in motion_entries:
        start_time = 0.0 if entry.start_time is None else entry.start_time
        end_time = source_length / fps if entry.end_time is None else entry.end_time
        start_frame = max(0, round(start_time * fps))
        end_frame = min(source_length, round(end_time * fps))
        key = (entry.sample_id,)
        if start_time < 0 or end_time <= start_time:
            raise ValueError(
                f"Invalid temporal span for stats: {entry.sample_id!r} "
                f"({start_time}..{end_time})"
            )
        if end_frame - start_frame < min_frames:
            excluded_short_spans += 1
            continue
        span_count += 1
        span_local_rotations = local_rotations[start_frame:end_frame]
        span_root_positions = root_positions[start_frame:end_frame]
        for window_index, (window_start, window_end) in enumerate(
            _covering_windows(end_frame - start_frame, max_frames)
        ):
            window_length = window_end - window_start
            lengths = torch.tensor([window_length])
            features = motion_rep(
                span_local_rotations[window_start:window_end].unsqueeze(0),
                span_root_positions[window_start:window_end].unsqueeze(0),
                to_normalize=False,
                lengths=lengths,
            )
            features = motion_rep.translate_2d_to_zero(features)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_window_seed(seed, key, window_index))
            target_heading = torch.rand((1,), generator=generator, dtype=features.dtype) * (
                2.0 * torch.pi
            )
            features = motion_rep.rotate_to(features, target_heading)
            global_root = features[..., motion_rep.root_slice]
            local_root = motion_rep.global_root_to_local_root(
                global_root, normalized=False, lengths=lengths
            )
            body = features[..., motion_rep.body_slice]
            moments[0].update(global_root[0])
            moments[1].update(local_root[0])
            moments[2].update(body[0])
            window_count += 1
            window_min = window_length if window_min is None else min(window_min, window_length)
            window_max = max(window_max, window_length)

    return {
        "span_count": span_count,
        "excluded_short_spans": excluded_short_spans,
        "window_count": window_count,
        "window_min": window_min,
        "window_max": window_max,
        "moments": [
            (item.count, item.sum.numpy(), item.sum_sq.numpy()) for item in moments
        ],
    }


def compute_stats(args) -> None:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing stats directory: {output}")
    skeleton = build_skeleton(args.skeleton_joints)
    motion_rep = KimodoMotionRep(skeleton=skeleton, fps=args.fps, stats_path=None)
    # The paper does not disclose how normalization statistics were fit.  For
    # each distinct manifest span, the reconstruction policy below partitions
    # that span into <=10 s windows and applies the training transforms once per
    # window.  Full/event/combined spans can overlap and therefore can weight a
    # source frame more than once.  This is a documented reconstruction choice,
    # not a paper-exact or official-statistics fact.
    max_seconds = float(getattr(args, "max_seconds", 10.0))
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    max_frames = round(max_seconds * args.fps)
    if max_frames < 3:
        raise ValueError("max_seconds at the requested fps must permit at least three frames")
    min_frames = int(getattr(args, "min_frames", 2))
    if min_frames < 2:
        raise ValueError("min_frames must be at least two")
    entries = load_manifest(args.manifest, args.split)
    # Caption/paraphrase variants can duplicate one motion span.  Fit each
    # distinct (motion,start,end) span once, retaining both full clips and
    # genuinely different temporal action spans.
    source_entries = entries
    unique = {}
    for entry in source_entries:
        key = (entry.motion_path, entry.start_time, entry.end_time)
        unique.setdefault(key, entry)
    # Caption variants are contiguous in manifests produced by our builder, but
    # temporal spans for the same source motion remain distinct.  Group those
    # spans so a large 120 Hz BVH is parsed and resampled exactly once.  The
    # order of both motion groups and spans within a group is stable.
    grouped_entries = {}
    source_fps_by_motion = {}
    for entry in unique.values():
        prior_source_fps = source_fps_by_motion.setdefault(entry.motion_path, entry.source_fps)
        if prior_source_fps != entry.source_fps:
            raise ValueError(
                "A motion cannot have multiple source_fps values while computing stats: "
                f"{entry.motion_path} ({prior_source_fps!r} and {entry.source_fps!r})"
            )
        grouped_entries.setdefault(entry.motion_path, []).append(entry)
    global_moments = OnlineMoments(motion_rep.global_root_dim)
    local_moments = OnlineMoments(motion_rep.local_root_dim)
    body_moments = OnlineMoments(motion_rep.body_dim)
    window_count = 0
    window_min = None
    window_max = 0
    span_count = 0
    excluded_short_spans = 0
    tasks = list(grouped_entries.items())
    num_workers = int(getattr(args, "num_workers", 1))
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    started = time.perf_counter()
    if num_workers == 1:
        _initialize_stats_worker(
            args.skeleton_joints, args.fps, max_frames, min_frames, args.seed
        )
        results = map(_compute_motion_group, tasks)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_initialize_stats_worker,
            initargs=(args.skeleton_joints, args.fps, max_frames, min_frames, args.seed),
        )
        results = executor.map(_compute_motion_group, tasks, chunksize=4)
    try:
        for motion_index, result in enumerate(results, start=1):
            span_count += result["span_count"]
            excluded_short_spans += result["excluded_short_spans"]
            window_count += result["window_count"]
            if result["window_min"] is not None:
                window_min = (
                    result["window_min"]
                    if window_min is None
                    else min(window_min, result["window_min"])
                )
            window_max = max(window_max, result["window_max"])
            for accumulator, values in zip(
                (global_moments, local_moments, body_moments),
                result["moments"],
                strict=True,
            ):
                accumulator.merge(*values)
            if motion_index % 1000 == 0 or motion_index == len(tasks):
                elapsed = time.perf_counter() - started
                rate = motion_index / elapsed
                remaining = (len(tasks) - motion_index) / rate if rate else float("inf")
                print(
                    f"Stats progress: {motion_index}/{len(tasks)} motions, "
                    f"{span_count} spans, {rate:.2f} motions/s, ETA {remaining/3600:.2f} h",
                    flush=True,
                )
    finally:
        if num_workers > 1:
            executor.shutdown(wait=True, cancel_futures=True)

    for name, moments in (
        ("global_root", global_moments),
        ("local_root", local_moments),
        ("body", body_moments),
    ):
        mean, std = moments.finalize()
        folder = output / name
        folder.mkdir(parents=True, exist_ok=False)
        np.save(folder / "mean.npy", mean)
        np.save(folder / "std.npy", std)
    stats_files = {}
    for group, expected_dimension in (("global_root", 5), ("local_root", 4), ("body", 364)):
        for filename in ("mean.npy", "std.npy"):
            path = output / group / filename
            array = np.load(path, allow_pickle=False)
            if array.dtype != np.float32 or array.shape != (expected_dimension,):
                raise ValueError(f"unexpected saved stats array contract: {path}")
            if not np.isfinite(array).all():
                raise ValueError(f"saved stats array contains non-finite values: {path}")
            stats_files[f"{group}/{filename}"] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
                "dtype": "float32",
                "shape": [expected_dimension],
            }
    manifest = Path(args.manifest).expanduser().resolve()
    metadata = {
        "schema_version": 3,
        # Keep the binding valid when a complete prepared bundle is moved or
        # when a ``.building`` directory is atomically published under its
        # final name.  The digest remains the authoritative content binding.
        "manifest": Path(os.path.relpath(manifest, output)).as_posix(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "split": args.split,
        "skeleton_joints": args.skeleton_joints,
        "fps": args.fps,
        "seed": args.seed,
        "heading_augmentation": "deterministic_uniform",
        "preprocessing": {
            "target_fps": args.fps,
            "maximum_seconds": max_seconds,
            "minimum_frames": min_frames,
            "root_origin": "first_frame_smoothed_root_xz_to_zero",
            "heading": "one_stable_seeded_uniform_target_heading_per_window",
            "caption_deduplication": "each_unique_motion_start_end_span_once",
            "stats_windowing": "all_frames_non_overlapping_windows_with_one_frame_tail_rebalanced",
            "worker_processes": num_workers,
            "deterministic_reduction": "per-motion float64 sums merged in manifest motion order",
            "stats_window_count": window_count,
            "stats_window_frames_min": window_min,
            "stats_window_frames_max": window_max,
        },
        "paper_disclosure": {
            "motion_fps_and_maximum_seconds": "explicit",
            "root_origin_and_random_first_heading": "explicit",
            "normalization_statistics_fitting_procedure": "not_disclosed_reconstruction_assumption",
        },
        "unique_clips": len(unique),
        "processed_spans": span_count,
        "excluded_short_spans": excluded_short_spans,
        "frame_counts": {
            "global_root": global_moments.count,
            "local_root": local_moments.count,
            "body": body_moments.count,
        },
        "files": stats_files,
    }
    (output / "stats.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved normalization statistics to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--skeleton-joints", type=int, default=30, choices=(22, 30, 34, 77))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-workers", type=int, default=1)
    return parser


def main() -> None:
    compute_stats(build_parser().parse_args())


if __name__ == "__main__":
    main()
