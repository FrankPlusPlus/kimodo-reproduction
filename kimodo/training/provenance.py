# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hashes and environment facts stored with every training run."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from kimodo.assets import SKELETONS_ROOT


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def stats_fingerprints(stats_path: str | Path) -> dict[str, str]:
    root = Path(stats_path)
    result = {}
    for group in ("global_root", "local_root", "body"):
        for filename in ("mean.npy", "std.npy"):
            path = root / group / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing stats file: {path}")
            result[f"{group}/{filename}"] = sha256_file(path)
    metadata = root / "stats.metadata.json"
    if metadata.is_file():
        result[metadata.name] = sha256_file(metadata)
    return result


def manifest_reference_fingerprints(manifest_path: str | Path) -> dict[str, dict[str, str]]:
    manifest = Path(manifest_path).expanduser().resolve()
    base = manifest.parent
    paths: set[Path] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            for key in ("motion", "text_embedding"):
                value = entry.get(key)
                if value:
                    path = Path(value).expanduser()
                    paths.add(path.resolve() if path.is_absolute() else (base / path).resolve())
    metadata = manifest.with_suffix(manifest.suffix + ".metadata.json")
    if metadata.is_file():
        paths.add(metadata)
        metadata_record = json.loads(metadata.read_text(encoding="utf-8"))
        source_manifest = metadata_record.get("source_manifest")
        if source_manifest:
            source = Path(source_manifest).expanduser().resolve()
            paths.add(source)
            source_metadata = source.with_suffix(source.suffix + ".metadata.json")
            if source_metadata.is_file():
                paths.add(source_metadata)
    result = {}
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"Manifest provenance reference is missing: {path}")
        result[str(path)] = {"sha256": sha256_file(path), "size": str(path.stat().st_size)}
    return result


def code_fingerprints(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root)
    # Use the whole Python package rather than a hand-maintained dependency
    # whitelist: motion loaders dynamically import BVH/G1/SMPL-X converters,
    # and asset/path helpers also affect the produced training tensors.
    candidates = list((root / "kimodo").rglob("*.py"))
    candidates.extend((root / "configs" / "training").glob("*.yaml"))
    candidates.extend(
        root / relative
        for relative in (
            "benchmark/generate_eval.py",
            "pyproject.toml",
            "setup.py",
            "docker_requirements.txt",
        )
    )
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(set(candidates))
        if path.is_file()
    }


def skeleton_fingerprints(nbjoints: int) -> dict[str, str]:
    folder_by_joints = {22: "smplx22", 30: "somaskel30", 34: "g1skel34", 77: "somaskel77"}
    try:
        root = SKELETONS_ROOT / folder_by_joints[nbjoints]
    except KeyError as error:
        raise ValueError(f"No registered skeleton assets for {nbjoints} joints") from error
    return {
        str(path.relative_to(SKELETONS_ROOT)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def collect_provenance(config, project_root: str | Path) -> dict:
    stats_path = config.model.stats_path
    if config.model.checkpoint_dir:
        # Official config normally resolves this internally.  Record only the
        # bundle files here; the trainer validates loaded dimensions separately.
        bundle = Path(config.model.checkpoint_dir)
        bundle_files = {
            str(path.relative_to(bundle)): sha256_file(path)
            for path in sorted(bundle.rglob("*"))
            if path.is_file() and ".cache" not in path.relative_to(bundle).parts
        }
    else:
        bundle_files = {}
    return {
        "schema_version": 2,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "git_commit": _git_commit(Path(project_root)),
        "manifest": {
            "path": str(Path(config.data.manifest).resolve()),
            "sha256": sha256_file(config.data.manifest),
            "references": manifest_reference_fingerprints(config.data.manifest),
        },
        "stats": stats_fingerprints(stats_path) if stats_path else {},
        "official_bundle": bundle_files,
        "skeleton_assets": skeleton_fingerprints(config.model.skeleton_joints),
        "code_snapshot": code_fingerprints(project_root),
    }


def save_provenance(value: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
