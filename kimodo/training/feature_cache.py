# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""On-disk motion-feature cache: precompute FK once, mmap at train time."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

try:  # Present in the production image; stdlib remains the portable fallback.
    import orjson

    _json_loads = orjson.loads
except ImportError:  # pragma: no cover - depends on the deployment image
    _json_loads = json.loads

if TYPE_CHECKING:
    from kimodo.training.data import ManifestEntry

FEATURE_CACHE_SCHEMA_VERSION = 1
META_NAME = "meta.json"
INDEX_NAME = "index.jsonl"


def sample_cache_key(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()


def feature_relpath(sample_id: str) -> str:
    key = sample_cache_key(sample_id)
    return f"features/{key[:2]}/{key}.f16.npy"


def stats_fingerprint(stats_path: str | Path | None) -> str:
    if not stats_path:
        return ""
    root = Path(stats_path).expanduser().resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.npy")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_cache_fingerprint(
    *,
    fps: int,
    feature_dim: int,
    skeleton_joints: int,
    stats_path: str | Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "fps": int(fps),
        "feature_dim": int(feature_dim),
        "skeleton_joints": int(skeleton_joints),
        "stats_sha256": stats_fingerprint(stats_path),
        "dtype": "float16",
        # Features are stored after motion_rep + translate_2d_to_zero only.
        "pipeline": "motion_rep_translate2d",
    }


def write_meta(cache_dir: Path, fingerprint: dict[str, Any], entry_count: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {**fingerprint, "entry_count": int(entry_count)}
    path = cache_dir / META_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_meta(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / META_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache meta missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported feature-cache schema: {payload.get('schema_version')!r}"
        )
    return payload


def assert_cache_fingerprint(meta: dict[str, Any], expected: dict[str, Any]) -> None:
    keys = (
        "schema_version",
        "fps",
        "feature_dim",
        "skeleton_joints",
        "stats_sha256",
        "dtype",
        "pipeline",
    )
    mismatches = {
        key: (meta.get(key), expected.get(key))
        for key in keys
        if meta.get(key) != expected.get(key)
    }
    if mismatches:
        raise ValueError(f"Feature-cache fingerprint mismatch: {mismatches}")


def load_index(
    cache_dir: Path,
    *,
    read_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    path = cache_dir / INDEX_NAME if read_path is None else Path(read_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache index missing: {path}")
    index: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _json_loads(line)
            sample_id = str(row["id"])
            if sample_id in index:
                raise ValueError(f"Duplicate feature-cache id at line {line_number}: {sample_id}")
            index[sample_id] = row
    return index


def materialize_entry_features(
    entry: ManifestEntry,
    motion_rep,
    *,
    min_frames: int,
) -> torch.Tensor:
    """Build unnormalized, translated features for the manifest temporal span.

    Does not apply random windows, heading augmentation, or stats normalization.
    """
    from kimodo.training.data import (
        _convert_rotations_to_model_skeleton,
        _load_training_motion_file,
    )

    fps = float(motion_rep.fps)
    motion, source_joints = _load_training_motion_file(
        entry.motion_path, entry.source_fps, fps
    )
    local_rotations = _convert_rotations_to_model_skeleton(
        motion["local_rot_mats"].float(), source_joints, motion_rep.skeleton
    )
    root_positions = motion["root_positions"].float()
    length = int(local_rotations.shape[0])
    if entry.start_time is not None or entry.end_time is not None:
        start_time = 0.0 if entry.start_time is None else entry.start_time
        end_time = length / fps if entry.end_time is None else entry.end_time
        if start_time < 0 or end_time <= start_time:
            raise ValueError(
                f"Invalid temporal crop for {entry.sample_id!r}: {start_time}..{end_time}"
            )
        start_frame = max(0, round(start_time * fps))
        end_frame = min(length, round(end_time * fps))
        if end_frame - start_frame < min_frames:
            raise ValueError(
                f"Temporal crop for {entry.sample_id!r} has only {end_frame - start_frame} frames"
            )
        local_rotations = local_rotations[start_frame:end_frame]
        root_positions = root_positions[start_frame:end_frame]
        length = end_frame - start_frame
    if length < min_frames:
        raise ValueError(f"Motion {entry.sample_id!r} has only {length} frames")

    features = motion_rep(
        local_rotations.unsqueeze(0),
        root_positions.unsqueeze(0),
        to_normalize=False,
        lengths=torch.tensor([length]),
    )
    features = motion_rep.translate_2d_to_zero(features)
    return features[0].contiguous()


def save_feature_array(path: Path, features: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = features.detach().cpu().numpy().astype(np.float16, copy=False)
    # Write via a sibling temp file; np.save requires a .npy suffix.
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, array, allow_pickle=False)
    tmp.replace(path)


def load_feature_array(path: Path) -> np.ndarray:
    # mmap shares pages across DataLoader workers via the OS page cache.
    return np.load(path, allow_pickle=False, mmap_mode="r")
