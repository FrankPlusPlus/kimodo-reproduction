# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adopt a verified legacy Kimodo bundle without re-encoding LLM2Vec text."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from kimodo.sanitize import sanitize_texts
from kimodo.training.reference_inventory import (
    build_reference_inventory,
    load_inventory_summary,
    verify_reference_inventory_full,
)

from .pipeline import PipelineError, _atomic_json, _atomic_yaml, _validate_stats_bundle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _frame_inventory(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    result: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            cached = record.get("cached")
            frames = record.get("frames")
            if not isinstance(cached, str) or not isinstance(frames, int) or frames < 1:
                raise PipelineError(
                    f"{path}:{line_number} lacks a canonical cached path/frame count"
                )
            result[Path(cached).as_posix()] = frames
    return result


def _npz_frames(path: Path) -> int:
    with np.load(path, allow_pickle=False) as archive:
        if "local_rot_mats" not in archive.files or "root_positions" not in archive.files:
            raise PipelineError(f"legacy canonical motion has no Kimodo arrays: {path}")
        local = archive["local_rot_mats"]
        root = archive["root_positions"]
        if local.ndim != 4 or local.shape[1:] != (30, 3, 3):
            raise PipelineError(f"legacy motion has invalid rotation shape {local.shape}: {path}")
        if root.shape != (local.shape[0], 3):
            raise PipelineError(f"legacy motion has invalid root shape {root.shape}: {path}")
        return int(local.shape[0])


def _publish_asset(source: Path, destination: Path, mode: str) -> None:
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise PipelineError(f"adoption asset collision: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as error:
            raise PipelineError(
                f"cannot hardlink {source} into {destination}; place both roots on one filesystem "
                "or set pipeline.adoption_asset_mode=copy"
            ) from error
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise PipelineError("adoption asset mode must be hardlink or copy")


def _asset_relative(source: Path, legacy_root: Path, category: str) -> Path:
    try:
        relative = source.resolve().relative_to(legacy_root.resolve())
    except ValueError as error:
        raise PipelineError(
            f"legacy {category} lies outside the declared bundle and cannot be adopted safely: {source}"
        ) from error
    if not relative.parts or relative.parts[0] != category:
        raise PipelineError(
            f"legacy {category} must be stored below {legacy_root / category}: {source}"
        )
    return relative


def _stats_files(stats: Path) -> dict[str, dict]:
    result = {}
    for group, dimension in (("global_root", 5), ("local_root", 4), ("body", 364)):
        for filename in ("mean.npy", "std.npy"):
            path = stats / group / filename
            array = np.load(path, allow_pickle=False)
            if array.dtype != np.float32 or array.shape != (dimension,) or not np.isfinite(array).all():
                raise PipelineError(f"legacy stats array has invalid contract: {path}")
            result[f"{group}/{filename}"] = {
                "sha256": _sha256(path),
                "size": path.stat().st_size,
                "dtype": "float32",
                "shape": [dimension],
            }
    return result


def _write_manifest_metadata(
    path: Path, *, schema: int, source_metadata: dict, adoption: dict, output_sha: str
) -> None:
    payload = {
        key: value
        for key, value in source_metadata.items()
        if key
        in {
            "builder",
            "counts",
            "paper_data_recipe",
            "paper_parity_gate",
            "repeat_counts",
            "skeleton",
            "split_accounting",
            "split_name",
        }
    }
    payload.update(
        schema_version=schema,
        path_mode="relative",
        adoption=adoption,
        output={"path": path.name, "sha256": output_sha},
    )
    _atomic_json(path.with_suffix(path.suffix + ".metadata.json"), payload)


def _training_paths_payload(destination: Path, run: Path) -> dict:
    return {
        "schema_version": 1,
        "data": {
            "manifest": str(destination / "train.cached.jsonl"),
            "reference_inventory": str(
                destination / "train.cached.references.jsonl"
            ),
        },
        "model": {
            "stats_path": str(destination / "stats/repro-soma30-30fps"),
            "checkpoint_dir": None,
            "checkpoint_weights": None,
        },
        "runtime": {
            "output_dir": str(run / "repro-soma30"),
            "resume": None,
        },
    }


def bind_prepared_bundle(
    *,
    prepared_root: str | Path,
    run_root: str | Path,
    repro_paths_yaml: str | Path,
) -> dict:
    """Verify a relocated prepared bundle and write its machine-local paths YAML."""

    prepared = Path(prepared_root).expanduser().resolve()
    run = Path(run_root).expanduser().resolve()
    paths_yaml = Path(repro_paths_yaml).expanduser().resolve()
    receipt_path = prepared / "resource-state.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "repro_train_ready":
        raise PipelineError(
            f"prepared bundle is not marked repro_train_ready: {receipt_path}"
        )
    manifest = prepared / "train.cached.jsonl"
    inventory = prepared / "train.cached.references.jsonl"
    summary = load_inventory_summary(manifest, inventory)
    verification = verify_reference_inventory_full(manifest, inventory)
    stats = prepared / "stats/repro-soma30-30fps"
    stats_files = _validate_stats_bundle(stats)
    preflight = _preflight_prepared_bundle(prepared, min_frames=2)
    expected = receipt.get("outputs") or {}
    if expected.get("manifest_sha256") not in (None, _sha256(manifest)):
        raise PipelineError("prepared manifest does not match resource-state.json")
    if expected.get("inventory_sha256") not in (None, _sha256(inventory)):
        raise PipelineError("prepared inventory does not match resource-state.json")
    _atomic_yaml(paths_yaml, _training_paths_payload(prepared, run))
    return {
        "status": "prepared_bundle_bound",
        "prepared_root": str(prepared),
        "paths_yaml": str(paths_yaml),
        "inventory": summary,
        "reference_verification": verification["verification"],
        "data_preflight": preflight,
        "stats_files": stats_files,
    }


def _preflight_prepared_bundle(prepared: Path, min_frames: int) -> dict:
    from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
    from kimodo.skeleton.registry import build_skeleton
    from kimodo.training.data import MotionManifestDataset, collate_motion_batch

    motion_rep = KimodoMotionRep(
        skeleton=build_skeleton(30),
        fps=30,
        stats_path=prepared / "stats/repro-soma30-30fps",
    )
    dataset = MotionManifestDataset(
        prepared / "train.cached.jsonl",
        "train",
        motion_rep,
        max_seconds=10.0,
        min_frames=min_frames,
        seed=1234,
        require_cached_text=True,
        require_paper_data_parity=False,
        normalize=True,
        augment=True,
    )
    if dataset.excluded_short_entries:
        raise PipelineError(
            "adopted manifest unexpectedly contains "
            f"{dataset.excluded_short_entries} rows below min_frames={min_frames}"
        )
    sample_count = min(128, len(dataset))
    batch = collate_motion_batch([dataset[index] for index in range(sample_count)])
    text = batch.get("text_features")
    if text is None or tuple(text.shape[1:]) != (1, 4096):
        raise PipelineError("adopted text batch does not satisfy [B,1,4096]")
    return {
        "status": "full_manifest_contract_passed",
        "manifest_entries_validated": dataset.manifest_entries,
        "dataset_entries": len(dataset),
        "excluded_short_entries": dataset.excluded_short_entries,
        "sampled_entries": sample_count,
        "motion_shape": list(batch["clean_motion"].shape),
        "text_shape": list(text.shape),
    }


def adopt_legacy_bundle(
    *,
    legacy_root: str | Path,
    output_root: str | Path,
    run_root: str | Path,
    repro_paths_yaml: str | Path,
    conversion_inventory: str | Path | None = None,
    asset_mode: str = "hardlink",
    min_frames: int = 2,
) -> dict:
    """Create an atomic, self-contained portable bundle from legacy assets.

    The old provider identity is retained in an adoption receipt.  Existing
    vectors are never assigned the new encoder identity and the 8B encoder is
    never loaded.
    """

    if min_frames != 2:
        raise PipelineError(
            "legacy stats cover every source span; adoption currently requires min_frames=2"
        )
    legacy = Path(legacy_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    run = Path(run_root).expanduser().resolve()
    paths_yaml = Path(repro_paths_yaml).expanduser().resolve()
    conversion = (
        Path(conversion_inventory).expanduser().resolve()
        if conversion_inventory is not None
        else None
    )
    if destination.exists():
        receipt_path = destination / "resource-state.json"
        if not receipt_path.is_file():
            raise FileExistsError(
                f"adopted output exists without a train-ready receipt: {destination}"
            )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "repro_train_ready"
            or receipt.get("mode") != "verified_legacy_no_reencode"
        ):
            raise PipelineError(f"unsupported existing adopted bundle: {destination}")
        verification = verify_reference_inventory_full(
            destination / "train.cached.jsonl",
            destination / "train.cached.references.jsonl",
        )
        _validate_stats_bundle(destination / "stats/repro-soma30-30fps")
        preflight = _preflight_prepared_bundle(destination, min_frames)
        receipt = {**receipt, "data_preflight": preflight}
        _atomic_json(receipt_path, receipt)
        _atomic_yaml(paths_yaml, _training_paths_payload(destination, run))
        return {
            **receipt,
            "status": "repro_train_ready_reused",
            "reference_verification": verification["verification"],
            "paths_yaml": str(paths_yaml),
        }
    for required in (
        legacy / "train.raw.jsonl",
        legacy / "train.cached.jsonl",
        legacy / "train.raw.jsonl.metadata.json",
        legacy / "train.cached.jsonl.metadata.json",
        legacy / "stats/repro-soma30-30fps/stats.metadata.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if conversion is not None and not conversion.is_file():
        raise FileNotFoundError(conversion)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.adopting.", dir=destination.parent)
    )
    raw_source = legacy / "train.raw.jsonl"
    cached_source = legacy / "train.cached.jsonl"
    raw_output = staging / "train.raw.jsonl"
    cached_output = staging / "train.cached.jsonl"
    inventory_frames = _frame_inventory(conversion)
    motion_frames: dict[Path, int] = {}
    linked_assets: set[Path] = set()
    embedding_records: dict[Path, dict] = {}
    provider_identities: set[str] = set()
    try:
        def adopt_motion(value: str) -> tuple[Path, int]:
            source = _resolve(legacy, value)
            relative = _asset_relative(source, legacy, "motions")
            target = staging / relative
            if target not in linked_assets:
                _publish_asset(source, target, asset_mode)
                linked_assets.add(target)
            if source not in motion_frames:
                suffix = relative.as_posix().removeprefix("motions/soma30-30fps/")
                motion_frames[source] = inventory_frames.get(suffix) or _npz_frames(source)
            return target, motion_frames[source]

        with raw_source.open(encoding="utf-8") as source, raw_output.open(
            "x", encoding="utf-8"
        ) as output:
            raw_rows = 0
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get("motion"), str):
                    raise PipelineError(f"{raw_source}:{line_number} lacks motion")
                target, frames = adopt_motion(row["motion"])
                row["motion"] = _relative(target, staging)
                row["frame_count"] = frames
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                raw_rows += 1

        with cached_source.open(encoding="utf-8") as source, cached_output.open(
            "x", encoding="utf-8"
        ) as output:
            cached_rows = 0
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                target_motion, frames = adopt_motion(str(row.get("motion", "")))
                embedding_source = _resolve(legacy, str(row.get("text_embedding", "")))
                embedding_relative = _asset_relative(
                    embedding_source, legacy, "text-cache"
                )
                embedding_target = staging / embedding_relative
                if embedding_target not in linked_assets:
                    _publish_asset(embedding_source, embedding_target, asset_mode)
                    linked_assets.add(embedding_target)
                legacy_sidecar = embedding_source.with_suffix(".metadata.json")
                if not legacy_sidecar.is_file():
                    raise FileNotFoundError(legacy_sidecar)
                cache_key = str(row.get("text_cache_key", ""))
                if cache_key != embedding_source.stem:
                    raise PipelineError(
                        f"legacy cache key/path mismatch at {cached_source}:{line_number}"
                    )
                if embedding_target not in embedding_records:
                    legacy_record = json.loads(legacy_sidecar.read_text(encoding="utf-8"))
                    sanitized = sanitize_texts([str(row.get("text", ""))])[0]
                    text_sha = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()
                    provider = str(legacy_record.get("provider_identity", ""))
                    content_sha = _sha256(embedding_source)
                    if (
                        legacy_record.get("schema_version") != 2
                        or legacy_record.get("cache_key") != cache_key
                        or legacy_record.get("normalized_text_sha256") != text_sha
                        or legacy_record.get("dtype") != "float32"
                        or legacy_record.get("shape") != [1, 4096]
                        or legacy_record.get("sha256") != content_sha
                        or not provider
                    ):
                        raise PipelineError(
                            f"legacy embedding identity failed verification: {embedding_source}"
                        )
                    provider_identities.add(provider)
                    adopted_record = {
                        "schema_version": 1,
                        "cache_key": cache_key,
                        "sanitized_text_sha256": text_sha,
                        "encoder_identity_sha256": hashlib.sha256(
                            provider.encode("utf-8")
                        ).hexdigest(),
                        "dtype": "float32",
                        "shape": [1, 4096],
                        "size": embedding_source.stat().st_size,
                        "sha256": content_sha,
                        "adoption": {
                            "source_schema_version": 2,
                            "source_metadata_sha256": _sha256(legacy_sidecar),
                            "legacy_provider_identity": provider,
                        },
                    }
                    _atomic_json(
                        embedding_target.with_suffix(
                            embedding_target.suffix + ".metadata.json"
                        ),
                        adopted_record,
                    )
                    embedding_records[embedding_target] = adopted_record
                adopted_record = embedding_records[embedding_target]
                row["motion"] = _relative(target_motion, staging)
                row["frame_count"] = frames
                row["text_embedding"] = _relative(embedding_target, staging)
                row["text_embedding_metadata"] = _relative(
                    embedding_target.with_suffix(
                        embedding_target.suffix + ".metadata.json"
                    ),
                    staging,
                )
                row["text_embedding_sha256"] = adopted_record["sha256"]
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                cached_rows += 1

        if len(provider_identities) != 1:
            raise PipelineError(
                f"legacy bundle must use exactly one encoder identity, got {len(provider_identities)}"
            )
        legacy_raw_metadata = json.loads(
            (legacy / "train.raw.jsonl.metadata.json").read_text(encoding="utf-8")
        )
        legacy_cache_metadata = json.loads(
            (legacy / "train.cached.jsonl.metadata.json").read_text(encoding="utf-8")
        )
        adoption = {
            "mode": "verified_legacy_no_reencode",
            "legacy_root_basename": legacy.name,
            "legacy_raw_manifest_sha256": _sha256(raw_source),
            "legacy_cached_manifest_sha256": _sha256(cached_source),
            "legacy_cache_metadata_sha256": _sha256(
                legacy / "train.cached.jsonl.metadata.json"
            ),
            "asset_mode": asset_mode,
            "minimum_frames": min_frames,
        }
        _write_manifest_metadata(
            raw_output,
            schema=2,
            source_metadata=legacy_raw_metadata,
            adoption=adoption,
            output_sha=_sha256(raw_output),
        )
        provider = next(iter(provider_identities))
        cached_metadata = {
            "schema_version": 5,
            "path_mode": "relative",
            "encoder": provider,
            "dtype": "float32",
            "embedding_shape": [1, 4096],
            "source_manifest": raw_output.name,
            "source_manifest_sha256": _sha256(raw_output),
            "source_manifest_metadata": raw_output.name + ".metadata.json",
            "source_manifest_metadata_sha256": _sha256(
                raw_output.with_suffix(raw_output.suffix + ".metadata.json")
            ),
            "adoption": adoption,
            "legacy_cache_provenance": legacy_cache_metadata.get("cache_provenance"),
            "legacy_encoder_artifacts": legacy_cache_metadata.get("encoder_artifacts"),
            "output": {
                "path": cached_output.name,
                "sha256": _sha256(cached_output),
                "entries": cached_rows,
            },
        }
        _atomic_json(
            cached_output.with_suffix(cached_output.suffix + ".metadata.json"),
            cached_metadata,
        )

        legacy_stats = legacy / "stats/repro-soma30-30fps"
        adopted_stats = staging / "stats/repro-soma30-30fps"
        for relative in _stats_files(legacy_stats):
            _publish_asset(legacy_stats / relative, adopted_stats / relative, asset_mode)
        legacy_stats_metadata = json.loads(
            (legacy_stats / "stats.metadata.json").read_text(encoding="utf-8")
        )
        preprocessing = dict(legacy_stats_metadata.get("preprocessing") or {})
        preprocessing["minimum_frames"] = min_frames
        stats_metadata = {
            **legacy_stats_metadata,
            "schema_version": 3,
            "manifest": "../../train.cached.jsonl",
            "manifest_sha256": _sha256(cached_output),
            "preprocessing": preprocessing,
            "files": _stats_files(adopted_stats),
            "adoption": adoption,
        }
        _atomic_json(adopted_stats / "stats.metadata.json", stats_metadata)
        _validate_stats_bundle(adopted_stats)

        reference_inventory = staging / "train.cached.references.jsonl"
        build_reference_inventory(cached_output, reference_inventory)
        verification = verify_reference_inventory_full(
            cached_output, reference_inventory
        )

        # Schema-5 dataset construction scans every row/semantic sidecar and
        # frame count; collating a real batch also exercises motion/stats/text.
        preflight = _preflight_prepared_bundle(staging, min_frames)

        receipt = {
            "schema_version": 2,
            "status": "repro_train_ready",
            "mode": "verified_legacy_no_reencode",
            "adoption": adoption,
            "full_manifest_entries": preflight["manifest_entries_validated"],
            "unique_embeddings": len(embedding_records),
            "unique_motions": len(motion_frames),
            "reference_verification": verification["verification"],
            "data_preflight": preflight,
            "outputs": {
                "manifest_sha256": _sha256(cached_output),
                "inventory_sha256": _sha256(reference_inventory),
                "stats_files": _validate_stats_bundle(adopted_stats),
            },
        }
        _atomic_json(staging / "resource-state.json", receipt)
        os.replace(staging, destination)
        staging = None

        _atomic_yaml(paths_yaml, _training_paths_payload(destination, run))
        return {**receipt, "paths_yaml": str(paths_yaml)}
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
