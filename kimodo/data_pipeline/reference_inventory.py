# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed inventories for large manifest reference sets.

Building or fully verifying an inventory intentionally reads every referenced
file. Trainer startup uses :func:`load_inventory_summary` instead, which reads
only the manifest, inventory, and small metadata sidecar.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from kimodo.common.file_permissions import publish_file

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_metadata_path(path: str | Path) -> Path:
    inventory = Path(path)
    return inventory.with_suffix(inventory.suffix + ".metadata.json")


def _resolve_relative_to(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _portable_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def manifest_reference_paths(manifest_path: str | Path) -> set[Path]:
    """Resolve all motion/text references represented by a cached manifest."""
    manifest = Path(manifest_path).expanduser().resolve()
    base = manifest.parent
    paths: set[Path] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            for key in ("motion", "text_embedding", "text_embedding_metadata"):
                value = entry.get(key)
                if value:
                    path = Path(value).expanduser()
                    paths.add(path.resolve() if path.is_absolute() else (base / path).resolve())

    metadata = manifest.with_suffix(manifest.suffix + ".metadata.json")
    if metadata.is_file():
        paths.add(metadata.resolve())
        metadata_record = json.loads(metadata.read_text(encoding="utf-8"))
        source_manifest = metadata_record.get("source_manifest")
        if source_manifest:
            source = _resolve_relative_to(base, str(source_manifest))
            paths.add(source)
            recorded_source_metadata = metadata_record.get("source_manifest_metadata")
            source_metadata = (
                _resolve_relative_to(base, str(recorded_source_metadata))
                if recorded_source_metadata
                else source.with_suffix(source.suffix + ".metadata.json")
            )
            if source_metadata.is_file():
                paths.add(source_metadata)
        for source_record in metadata_record.get("source_manifests", []):
            if not isinstance(source_record, dict) or not source_record.get("path"):
                raise ValueError("source_manifests entries must contain a path")
            source = _resolve_relative_to(base, str(source_record["path"]))
            if not source.is_file():
                raise FileNotFoundError(f"source manifest is missing: {source}")
            recorded_sha = source_record.get("sha256")
            recorded_size = source_record.get("size")
            if recorded_sha != sha256_file(source) or recorded_size != source.stat().st_size:
                raise ValueError(
                    f"source manifest differs from overlay provenance: {source}"
                )
            paths.add(source)
            recorded_metadata = source_record.get("metadata_path")
            source_metadata = (
                _resolve_relative_to(base, str(recorded_metadata))
                if recorded_metadata
                else source.with_suffix(source.suffix + ".metadata.json")
            )
            if source_metadata.is_file():
                expected_metadata_sha = source_record.get("metadata_sha256")
                if (
                    expected_metadata_sha is not None
                    and expected_metadata_sha != sha256_file(source_metadata)
                ):
                    raise ValueError(
                        f"source manifest metadata differs from overlay provenance: {source_metadata}"
                    )
                paths.add(source_metadata)
    return paths


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        publish_file(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_reference_inventory(manifest_path: str | Path, output_path: str | Path) -> dict:
    """Hash every unique manifest reference once and write an atomic JSONL inventory."""
    manifest = Path(manifest_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    metadata_path = inventory_metadata_path(output)
    if output.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite inventory or metadata: {output}, {metadata_path}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    reference_count = 0
    total_bytes = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            records = [
                (_portable_path(path, manifest.parent), path)
                for path in manifest_reference_paths(manifest)
            ]
            for serialized_path, path in sorted(records, key=lambda item: item[0]):
                if not path.is_file():
                    raise FileNotFoundError(f"Manifest provenance reference is missing: {path}")
                size = path.stat().st_size
                record = {
                    "path": serialized_path,
                    "sha256": sha256_file(path),
                    "size": size,
                }
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                reference_count += 1
                total_bytes += size
        publish_file(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    inventory_sha256 = sha256_file(output)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "builder": "kimodo.training.reference_inventory",
        "manifest": {
            "path": _portable_path(manifest, metadata_path.parent),
            "sha256": sha256_file(manifest),
            "size": manifest.stat().st_size,
        },
        "inventory": {
            "path": _portable_path(output, metadata_path.parent),
            "sha256": inventory_sha256,
            "size": output.stat().st_size,
        },
        # The inventory is canonical compact JSONL sorted by portable path, so
        # its file digest is also the aggregate digest of all reference
        # path/size/content-digest records.
        "aggregate": {
            "algorithm": "sha256(canonical-jsonl-portable-v2)",
            "sha256": inventory_sha256,
        },
        "reference_count": reference_count,
        "total_reference_bytes": total_bytes,
    }
    _atomic_json(metadata_path, metadata)
    return metadata


def _require_sha256(value, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"Inventory metadata has invalid {label}")
    return value.lower()


def load_inventory_summary(manifest_path: str | Path, inventory_path: str | Path) -> dict:
    """Quickly validate inventory identity without opening referenced assets."""
    manifest = Path(manifest_path).expanduser().resolve()
    inventory = Path(inventory_path).expanduser().resolve()
    metadata_path = inventory_metadata_path(inventory)
    if not inventory.is_file():
        raise FileNotFoundError(f"Reference inventory is missing: {inventory}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Reference inventory metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    schema_version = metadata.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported reference inventory schema: {metadata.get('schema_version')}"
        )
    manifest_record = metadata.get("manifest")
    inventory_record = metadata.get("inventory")
    aggregate_record = metadata.get("aggregate")
    if not isinstance(manifest_record, dict) or not isinstance(inventory_record, dict):
        raise TypeError("Inventory metadata must contain manifest and inventory records")
    recorded_manifest = _resolve_relative_to(
        metadata_path.parent, str(manifest_record.get("path", ""))
    )
    recorded_inventory = _resolve_relative_to(
        metadata_path.parent, str(inventory_record.get("path", ""))
    )
    if recorded_manifest != manifest:
        raise ValueError("Reference inventory was built for a different manifest path")
    if recorded_inventory != inventory:
        raise ValueError("Reference inventory metadata points to a different inventory path")
    manifest_sha = _require_sha256(manifest_record.get("sha256"), "manifest sha256")
    inventory_sha = _require_sha256(inventory_record.get("sha256"), "inventory sha256")
    if not isinstance(aggregate_record, dict):
        raise TypeError("Inventory metadata must contain an aggregate record")
    aggregate_sha = _require_sha256(aggregate_record.get("sha256"), "aggregate sha256")
    expected_algorithm = (
        "sha256(canonical-jsonl-v1)"
        if schema_version == 1
        else "sha256(canonical-jsonl-portable-v2)"
    )
    if aggregate_record.get("algorithm") != expected_algorithm:
        raise ValueError("Unsupported inventory aggregate algorithm")
    if aggregate_sha != inventory_sha:
        raise ValueError("Inventory and aggregate SHA-256 values differ")
    if sha256_file(manifest) != manifest_sha:
        raise ValueError("Training manifest content differs from the reference inventory")
    if sha256_file(inventory) != inventory_sha:
        raise ValueError("Reference inventory content differs from its metadata")
    reference_count = metadata.get("reference_count")
    total_bytes = metadata.get("total_reference_bytes")
    if not isinstance(reference_count, int) or reference_count < 1:
        raise ValueError("Inventory metadata reference_count must be positive")
    if not isinstance(total_bytes, int) or total_bytes < 0:
        raise ValueError("Inventory metadata total_reference_bytes must be non-negative")
    return {
        "path": str(inventory),
        "sha256": inventory_sha,
        "aggregate_sha256": aggregate_sha,
        "manifest_sha256": manifest_sha,
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "reference_count": reference_count,
        "total_reference_bytes": total_bytes,
        "verification": "inventory_identity_only",
        "schema_version": schema_version,
    }


def _inventory_records(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not {"path", "sha256", "size"} <= record.keys():
                raise ValueError(f"{path}:{line_number} is not a reference inventory record")
            yield record


def verify_reference_inventory_full(
    manifest_path: str | Path, inventory_path: str | Path
) -> dict:
    """Independently rescan and hash every reference recorded by an inventory."""
    manifest = Path(manifest_path).expanduser().resolve()
    inventory = Path(inventory_path).expanduser().resolve()
    summary = load_inventory_summary(manifest, inventory)
    expected_paths = manifest_reference_paths(manifest)
    seen_paths: set[Path] = set()
    verified_bytes = 0
    for line_number, record in enumerate(_inventory_records(inventory), start=1):
        path = _resolve_relative_to(manifest.parent, str(record["path"]))
        if path in seen_paths:
            raise ValueError(f"Duplicate inventory path at record {line_number}: {path}")
        seen_paths.add(path)
        if not path.is_file():
            raise FileNotFoundError(f"Inventory reference is missing: {path}")
        size = path.stat().st_size
        if record["size"] != size:
            raise ValueError(f"Inventory size mismatch: {path}")
        expected_sha = _require_sha256(record["sha256"], f"sha256 for {path}")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"Inventory SHA-256 mismatch: {path}")
        verified_bytes += size
    if seen_paths != expected_paths:
        missing = sorted(map(str, expected_paths - seen_paths))
        extra = sorted(map(str, seen_paths - expected_paths))
        raise ValueError(
            f"Inventory reference set differs from manifest; missing={missing[:5]}, extra={extra[:5]}"
        )
    if len(seen_paths) != summary["reference_count"]:
        raise ValueError("Inventory reference count differs from metadata")
    if verified_bytes != summary["total_reference_bytes"]:
        raise ValueError("Inventory byte count differs from metadata")
    return {
        **summary,
        "verification": "full_content_verified",
        "verified_reference_count": len(seen_paths),
        "verified_reference_bytes": verified_bytes,
    }
