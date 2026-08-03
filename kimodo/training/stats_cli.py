# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute split normalization statistics for Kimodo motion features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from kimodo.exports.motion_io import load_motion_file
from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton

from .data import _convert_rotations_to_model_skeleton, load_manifest


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


def compute_stats(args) -> None:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing stats directory: {output}")
    skeleton = build_skeleton(args.skeleton_joints)
    motion_rep = KimodoMotionRep(skeleton=skeleton, fps=args.fps, stats_path=None)
    # The paper does not disclose how normalization statistics were fit.  The
    # reconstruction policy below covers every valid source frame exactly once
    # using <=10 s windows, while applying the same per-window transforms as
    # training.  This is a documented industry default, not a paper-exact fact.
    max_seconds = float(getattr(args, "max_seconds", 10.0))
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    max_frames = int(round(max_seconds * args.fps))
    if max_frames < 3:
        raise ValueError("max_seconds at the requested fps must permit at least three frames")
    entries = load_manifest(args.manifest, args.split)
    # Caption/paraphrase variants can duplicate one motion span.  Fit each
    # distinct (motion,start,end) span once, retaining both full clips and
    # genuinely different temporal action spans.
    source_entries = entries
    unique = {}
    for entry in source_entries:
        key = (entry.motion_path, entry.start_time, entry.end_time)
        unique.setdefault(key, entry)
    global_moments = OnlineMoments(motion_rep.global_root_dim)
    local_moments = OnlineMoments(motion_rep.local_root_dim)
    body_moments = OnlineMoments(motion_rep.body_dim)
    window_count = 0
    window_lengths: list[int] = []
    for key, entry in unique.items():
        motion, source_joints = load_motion_file(
            str(entry.motion_path),
            source_fps=entry.source_fps,
            target_fps=float(args.fps),
        )
        local_rotations = _convert_rotations_to_model_skeleton(
            motion["local_rot_mats"].float(), source_joints, skeleton
        )
        root_positions = motion["root_positions"].float()
        source_length = int(local_rotations.shape[0])
        start_time = 0.0 if entry.start_time is None else entry.start_time
        end_time = source_length / args.fps if entry.end_time is None else entry.end_time
        start_frame = max(0, int(round(start_time * args.fps)))
        end_frame = min(source_length, int(round(end_time * args.fps)))
        if start_time < 0 or end_time <= start_time or end_frame - start_frame < 2:
            raise ValueError(
                f"Invalid or too-short temporal span for stats: {entry.sample_id!r} "
                f"({start_time}..{end_time})"
            )
        local_rotations = local_rotations[start_frame:end_frame]
        root_positions = root_positions[start_frame:end_frame]
        for window_index, (window_start, window_end) in enumerate(
            _covering_windows(end_frame - start_frame, max_frames)
        ):
            window_length = window_end - window_start
            lengths = torch.tensor([window_length])
            features = motion_rep(
                local_rotations[window_start:window_end].unsqueeze(0),
                root_positions[window_start:window_end].unsqueeze(0),
                to_normalize=False,
                lengths=lengths,
            )
            features = motion_rep.translate_2d_to_zero(features)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_window_seed(args.seed, key, window_index))
            target_heading = torch.rand((1,), generator=generator, dtype=features.dtype) * (
                2.0 * torch.pi
            )
            features = motion_rep.rotate_to(features, target_heading)
            global_root = features[..., motion_rep.root_slice]
            local_root = motion_rep.global_root_to_local_root(
                global_root, normalized=False, lengths=lengths
            )
            body = features[..., motion_rep.body_slice]
            global_moments.update(global_root[0])
            local_moments.update(local_root[0])
            body_moments.update(body[0])
            window_count += 1
            window_lengths.append(window_length)

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
    manifest = Path(args.manifest).expanduser().resolve()
    metadata = {
        "schema_version": 1,
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "split": args.split,
        "skeleton_joints": args.skeleton_joints,
        "fps": args.fps,
        "seed": args.seed,
        "heading_augmentation": "deterministic_uniform",
        "preprocessing": {
            "target_fps": args.fps,
            "maximum_seconds": max_seconds,
            "root_origin": "first_frame_smoothed_root_xz_to_zero",
            "heading": "one_stable_seeded_uniform_target_heading_per_window",
            "caption_deduplication": "each_unique_motion_start_end_span_once",
            "stats_windowing": "all_frames_non_overlapping_windows_with_one_frame_tail_rebalanced",
            "stats_window_count": window_count,
            "stats_window_frames_min": min(window_lengths),
            "stats_window_frames_max": max(window_lengths),
        },
        "paper_disclosure": {
            "motion_fps_and_maximum_seconds": "explicit",
            "root_origin_and_random_first_heading": "explicit",
            "normalization_statistics_fitting_procedure": "not_disclosed_reconstruction_assumption",
        },
        "unique_clips": len(unique),
        "frame_counts": {
            "global_root": global_moments.count,
            "local_root": local_moments.count,
            "body": body_moments.count,
        },
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
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main() -> None:
    compute_stats(build_parser().parse_args())


if __name__ == "__main__":
    main()
