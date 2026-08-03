# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manifest-based motion dataset for reproducible Kimodo training."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from kimodo.exports.motion_io import load_motion_file
from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77


@dataclass(frozen=True)
class ManifestEntry:
    sample_id: str
    motion_path: Path
    text: str
    split: str
    source_fps: float | None = None
    text_embedding_path: Path | None = None
    start_time: float | None = None
    end_time: float | None = None
    sample_kind: str = "full"


def _resolve_path(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _load_training_motion_file(
    path: Path, source_fps: float | None, target_fps: float
) -> tuple[dict[str, torch.Tensor], int]:
    """Load canonical training NPZs without deriving channels that are discarded.

    The generic export loader completes a Kimodo motion with FK, velocities,
    contacts and an ADMM-smoothed root. Training subsequently keeps only local
    rotations/root positions, crops them, and derives all those channels again.
    Canonical same-FPS Kimodo NPZs can therefore take this exact raw fast path;
    every other source format or resampling case retains the generic loader.
    """
    effective_source = 30.0 if source_fps is None else float(source_fps)
    if path.suffix.lower() == ".npz" and effective_source == float(target_fps):
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            is_kimodo = {"local_rot_mats", "root_positions"} <= keys
            is_amass = {"trans", "pose_body", "root_orient"} <= keys
            if is_kimodo and not is_amass:
                local = torch.from_numpy(np.asarray(archive["local_rot_mats"]).copy()).float()
                root = torch.from_numpy(np.asarray(archive["root_positions"]).copy()).float()
                if local.ndim != 4 or local.shape[-2:] != (3, 3):
                    raise ValueError(
                        f"Canonical local_rot_mats must be [T,J,3,3], got {tuple(local.shape)}: {path}"
                    )
                if root.ndim != 2 or root.shape[-1] != 3 or len(root) != len(local):
                    raise ValueError(
                        f"Canonical root_positions must be [T,3] and match rotations, "
                        f"got {tuple(root.shape)}: {path}"
                    )
                return {"local_rot_mats": local, "root_positions": root}, int(local.shape[1])
    return load_motion_file(
        str(path), source_fps=source_fps, target_fps=float(target_fps)
    )


def _manifest_sidecar_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(manifest_path.suffix + ".metadata.json")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def validate_paper_data_parity_manifest(path: str | Path) -> dict[str, Any]:
    """Fail closed unless required augmentation rows have auditable provenance.

    The released BONES-SEED inputs cannot satisfy this gate by themselves.  A
    separately generated manifest must include the Qwen paraphrase and stitched
    transition artifacts plus their generator/checkpoint provenance. This is
    only a self-consistency/schema gate: it cannot prove the unpublished prompt,
    sampling mixture, or transition protocol matches NVIDIA's private recipe.
    """
    manifest_path = Path(path).expanduser().resolve()
    sidecar_path = _manifest_sidecar_path(manifest_path)
    if not sidecar_path.is_file():
        raise RuntimeError(
            "Paper-data parity was requested, but the manifest sidecar is missing: "
            f"{sidecar_path}"
        )
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    gate = metadata.get("paper_parity_gate")
    if not isinstance(gate, dict):
        raise RuntimeError("Paper-data parity was requested, but paper_parity_gate is absent")
    blockers = list(gate.get("blockers") or [])
    if gate.get("eligible") is not True or blockers:
        details = ", ".join(blockers) if blockers else str(gate.get("status", "not eligible"))
        raise RuntimeError(f"Manifest is not eligible for paper-data parity: {details}")

    output_record = metadata.get("output")
    if not isinstance(output_record, dict):
        raise RuntimeError("Paper-data parity sidecar must fingerprint its manifest output")
    recorded_path = Path(str(output_record.get("path", ""))).expanduser().resolve()
    recorded_hash = output_record.get("sha256")
    actual_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if recorded_path != manifest_path or not _is_sha256(recorded_hash) or recorded_hash != actual_hash:
        raise RuntimeError("Paper-data parity sidecar output path/hash does not match the manifest")

    required_by_kind = {
        "llm_paraphrase": {
            "source_text_id",
            "text_generator_model",
            "text_generator_prompt_sha256",
        },
        "stitched_transition": {
            "source_motion_ids",
            "source_time_ranges",
            "transition_model_sha256",
            "transition_frame_range",
        },
    }
    seen = {kind: 0 for kind in required_by_kind}
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            kind = str(record.get("sample_kind", "full"))
            required = required_by_kind.get(kind)
            if required is None:
                continue
            missing = required - record.keys()
            if missing:
                raise RuntimeError(
                    f"{manifest_path}:{line_number} {kind} lacks paper-parity provenance: "
                    f"{sorted(missing)}"
                )
            if kind == "llm_paraphrase":
                if not isinstance(record["source_text_id"], str) or not record["source_text_id"].strip():
                    raise RuntimeError(f"{manifest_path}:{line_number} has an invalid source_text_id")
                model_name = record["text_generator_model"]
                if not isinstance(model_name, str) or "qwen3-32b" not in model_name.lower():
                    raise RuntimeError(
                        f"{manifest_path}:{line_number} paraphrase was not attributed to Qwen3-32B"
                    )
                if not _is_sha256(record["text_generator_prompt_sha256"]):
                    raise RuntimeError(
                        f"{manifest_path}:{line_number} has an invalid text-generator prompt hash"
                    )
            else:
                source_ids = record["source_motion_ids"]
                source_ranges = record["source_time_ranges"]
                transition_range = record["transition_frame_range"]
                if (
                    not isinstance(source_ids, list)
                    or len(source_ids) != 2
                    or source_ids[0] == source_ids[1]
                    or not all(isinstance(value, str) and value for value in source_ids)
                ):
                    raise RuntimeError(
                        f"{manifest_path}:{line_number} must identify two distinct source motions"
                    )
                valid_time_ranges = (
                    isinstance(source_ranges, list)
                    and len(source_ranges) == 2
                    and all(
                        isinstance(value, list)
                        and len(value) == 2
                        and all(isinstance(point, (int, float)) for point in value)
                        and value[0] < value[1]
                        for value in source_ranges
                    )
                )
                if not valid_time_ranges:
                    raise RuntimeError(
                        f"{manifest_path}:{line_number} has invalid source_time_ranges"
                    )
                if not _is_sha256(record["transition_model_sha256"]):
                    raise RuntimeError(
                        f"{manifest_path}:{line_number} has an invalid transition-model hash"
                    )
                if (
                    not isinstance(transition_range, list)
                    or len(transition_range) != 2
                    or not all(isinstance(value, int) and not isinstance(value, bool) for value in transition_range)
                    or transition_range[0] < 0
                    or transition_range[0] >= transition_range[1]
                ):
                    raise RuntimeError(
                        f"{manifest_path}:{line_number} has an invalid transition_frame_range"
                    )
            seen[kind] += 1
    missing_kinds = [kind for kind, count in seen.items() if count == 0]
    if missing_kinds:
        raise RuntimeError(
            "Manifest claims paper-data parity but has no rows for: " + ", ".join(missing_kinds)
        )
    return metadata


def load_manifest(path: str | Path, split: str | None = None) -> list[ManifestEntry]:
    """Load JSONL entries and reject ambiguous IDs or missing source files."""
    manifest_path = Path(path).expanduser().resolve()
    base = manifest_path.parent
    entries: list[ManifestEntry] = []
    seen_ids: set[str] = set()
    seen_motion_split: dict[Path, str] = {}
    # Large manifests repeat one motion across captions/events and one text
    # embedding across duplicate descriptions. Avoid millions of redundant NFS
    # metadata operations while retaining fail-closed path validation.
    file_exists: dict[Path, bool] = {}

    def is_file_cached(path: Path) -> bool:
        if path not in file_exists:
            file_exists[path] = path.is_file()
        return file_exists[path]

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            required = {"id", "motion", "text", "split"}
            missing = required - raw.keys()
            if missing:
                raise ValueError(f"{manifest_path}:{line_number} missing fields: {sorted(missing)}")
            sample_id = str(raw["id"])
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate manifest id: {sample_id}")
            seen_ids.add(sample_id)
            motion_path = _resolve_path(base, raw["motion"])
            assert motion_path is not None
            entry_split = str(raw["split"])
            prior_split = seen_motion_split.get(motion_path)
            if prior_split is not None and prior_split != entry_split:
                raise ValueError(
                    f"Train/eval leakage: {motion_path} appears in both {prior_split!r} and {entry_split!r}"
                )
            seen_motion_split[motion_path] = entry_split
            if split is not None and entry_split != split:
                continue
            embedding_path = _resolve_path(base, raw.get("text_embedding"))
            if not is_file_cached(motion_path):
                raise FileNotFoundError(f"Motion file does not exist: {motion_path}")
            if embedding_path is not None and not is_file_cached(embedding_path):
                raise FileNotFoundError(f"Text embedding does not exist: {embedding_path}")
            entries.append(
                ManifestEntry(
                    sample_id=sample_id,
                    motion_path=motion_path,
                    text=str(raw["text"]),
                    split=entry_split,
                    source_fps=float(raw["source_fps"]) if raw.get("source_fps") is not None else None,
                    text_embedding_path=embedding_path,
                    start_time=float(raw["start_time"]) if raw.get("start_time") is not None else None,
                    end_time=float(raw["end_time"]) if raw.get("end_time") is not None else None,
                    sample_kind=str(raw.get("sample_kind", "full")),
                )
            )
    if not entries:
        raise ValueError(f"Manifest contains no entries for split={split!r}: {manifest_path}")
    return entries


def _convert_rotations_to_model_skeleton(local_rotations: torch.Tensor, source_joints: int, skeleton):
    if source_joints == skeleton.nbjoints:
        return local_rotations
    if source_joints == 77 and isinstance(skeleton, SOMASkeleton30):
        return skeleton.from_SOMASkeleton77(local_rotations)
    if source_joints == 30 and isinstance(skeleton, SOMASkeleton77):
        source = SOMASkeleton30()
        return source.to_SOMASkeleton77(local_rotations)
    raise ValueError(
        f"Motion has {source_joints} joints but model skeleton has {skeleton.nbjoints}. "
        "Retarget the dataset before training; only SOMA 30<->77 projection is automatic."
    )


class MotionManifestDataset(Dataset):
    """Load formal Kimodo/AMASS/BVH/G1 inputs through the released converters."""

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        motion_rep,
        max_seconds: float | None = 10.0,
        min_frames: int = 2,
        seed: int = 1234,
        require_cached_text: bool = True,
        normalize: bool = True,
        augment: bool = True,
        require_paper_data_parity: bool = False,
    ) -> None:
        if require_paper_data_parity:
            validate_paper_data_parity_manifest(manifest)
        self.entries = load_manifest(manifest, split)
        self.motion_rep = motion_rep
        self.skeleton = motion_rep.skeleton
        self.fps = motion_rep.fps
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be positive or None")
        if min_frames < 1:
            raise ValueError("min_frames must be positive")
        self.max_frames = None if max_seconds is None else int(round(max_seconds * self.fps))
        if self.max_frames is not None and self.max_frames < min_frames:
            raise ValueError(
                f"max_seconds={max_seconds} at {self.fps} fps permits only "
                f"{self.max_frames} frames, fewer than min_frames={min_frames}"
            )
        self.min_frames = min_frames
        self.seed = seed
        self.epoch = 0
        self.require_cached_text = require_cached_text
        self.normalize = normalize
        self.augment = augment
        if require_cached_text:
            missing = [entry.sample_id for entry in self.entries if entry.text_embedding_path is None]
            if missing:
                preview = ", ".join(missing[:5])
                raise ValueError(f"Cached text embeddings are required; missing for: {preview}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.entries)

    def _generator(self, index: int) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch * len(self.entries) + index)
        return generator

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        generator = self._generator(index)
        motion, source_joints = _load_training_motion_file(
            entry.motion_path, entry.source_fps, float(self.fps)
        )
        local_rotations = _convert_rotations_to_model_skeleton(
            motion["local_rot_mats"].float(), source_joints, self.skeleton
        )
        root_positions = motion["root_positions"].float()
        length = int(local_rotations.shape[0])
        if entry.start_time is not None or entry.end_time is not None:
            start_time = 0.0 if entry.start_time is None else entry.start_time
            end_time = length / self.fps if entry.end_time is None else entry.end_time
            if start_time < 0 or end_time <= start_time:
                raise ValueError(
                    f"Invalid temporal crop for {entry.sample_id!r}: {start_time}..{end_time}"
                )
            start_frame = max(0, int(round(start_time * self.fps)))
            end_frame = min(length, int(round(end_time * self.fps)))
            if end_frame - start_frame < self.min_frames:
                raise ValueError(
                    f"Temporal crop for {entry.sample_id!r} has only {end_frame-start_frame} frames"
                )
            local_rotations = local_rotations[start_frame:end_frame]
            root_positions = root_positions[start_frame:end_frame]
            length = end_frame - start_frame
        if length < self.min_frames:
            raise ValueError(f"Motion {entry.sample_id!r} has only {length} frames")
        if self.max_frames is not None and length > self.max_frames:
            start = int(torch.randint(length - self.max_frames + 1, (), generator=generator).item())
            local_rotations = local_rotations[start : start + self.max_frames]
            root_positions = root_positions[start : start + self.max_frames]
            length = self.max_frames

        features = self.motion_rep(
            local_rotations.unsqueeze(0),
            root_positions.unsqueeze(0),
            to_normalize=False,
            lengths=torch.tensor([length]),
        )
        # Paper: translate the first-frame smoothed-root XZ above the origin.
        features = self.motion_rep.translate_2d_to_zero(features)
        if self.augment:
            target_heading = torch.rand((1,), generator=generator, dtype=features.dtype) * (2.0 * torch.pi)
            features = self.motion_rep.rotate_to(features, target_heading)
            first_heading = target_heading[0]
        else:
            first_heading = self.motion_rep.get_root_heading_angle(features)[0, 0]
        if self.normalize:
            features = self.motion_rep.normalize(features)
        features = features[0]

        text_features = None
        text_length = 0
        if entry.text_embedding_path is not None:
            array = np.load(entry.text_embedding_path, allow_pickle=False)
            if array.ndim == 1:
                array = array[None]
            if array.ndim != 2:
                raise ValueError(f"Text embedding must be [P,D], got {array.shape}: {entry.text_embedding_path}")
            text_features = torch.from_numpy(np.asarray(array)).float()
            text_length = int(text_features.shape[0])

        return {
            "id": entry.sample_id,
            "clean_motion": features,
            "length": length,
            "first_heading_angle": first_heading,
            "text": entry.text,
            "text_features": text_features,
            "text_length": text_length,
        }


def collate_motion_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([sample["length"] for sample in samples], dtype=torch.long)
    maximum = int(lengths.max().item())
    feature_dim = int(samples[0]["clean_motion"].shape[-1])
    clean_motion = torch.zeros(len(samples), maximum, feature_dim, dtype=torch.float32)
    valid_frames = torch.arange(maximum).expand(len(samples), maximum) < lengths[:, None]
    for index, sample in enumerate(samples):
        if sample["clean_motion"].shape[-1] != feature_dim:
            raise ValueError("All batch items must use the same motion representation")
        clean_motion[index, : sample["length"]] = sample["clean_motion"]

    have_cached = [sample["text_features"] is not None for sample in samples]
    if any(have_cached) and not all(have_cached):
        raise ValueError("A batch cannot mix cached and uncached text features")
    text_features = None
    text_pad_mask = None
    if all(have_cached):
        text_lengths = torch.tensor([sample["text_length"] for sample in samples], dtype=torch.long)
        maximum_text = int(text_lengths.max().item())
        text_dim = int(samples[0]["text_features"].shape[-1])
        text_features = torch.zeros(len(samples), maximum_text, text_dim, dtype=torch.float32)
        text_pad_mask = torch.arange(maximum_text).expand(len(samples), maximum_text) < text_lengths[:, None]
        for index, sample in enumerate(samples):
            value = sample["text_features"]
            if value.shape[-1] != text_dim:
                raise ValueError("All cached text embeddings must have the same width")
            text_features[index, : value.shape[0]] = value

    return {
        "ids": [sample["id"] for sample in samples],
        "clean_motion": clean_motion,
        "lengths": lengths,
        "valid_frames": valid_frames,
        "first_heading_angle": torch.stack([sample["first_heading_angle"] for sample in samples]),
        "texts": [sample["text"] for sample in samples],
        "text_features": text_features,
        "text_pad_mask": text_pad_mask,
    }
