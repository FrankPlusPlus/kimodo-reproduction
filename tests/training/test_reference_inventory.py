from __future__ import annotations

import json
from pathlib import Path

import pytest

import kimodo.data_pipeline.reference_inventory as inventory_module
from kimodo.data_pipeline.reference_inventory import (
    build_reference_inventory,
    load_inventory_summary,
    verify_reference_inventory_full,
)


def test_inventory_quick_startup_and_independent_full_verify(training_fixture, tmp_path):
    manifest = training_fixture["manifest"]
    inventory = tmp_path / "references.jsonl"
    metadata = build_reference_inventory(manifest, inventory)

    assert metadata["reference_count"] == 2
    quick = load_inventory_summary(manifest, inventory)
    assert quick["reference_count"] == 2
    assert quick["verification"] == "inventory_identity_only"

    full = verify_reference_inventory_full(manifest, inventory)
    assert full["verification"] == "full_content_verified"
    assert full["verified_reference_count"] == 2

    # Quick trainer startup intentionally validates the content-addressed
    # inventory identity, not hundreds of GB of referenced content.
    training_fixture["embedding"].write_bytes(b"changed-after-full-verification")
    assert load_inventory_summary(manifest, inventory)["verification"] == "inventory_identity_only"
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        verify_reference_inventory_full(manifest, inventory)


def test_inventory_rejects_manifest_or_inventory_tampering(training_fixture, tmp_path):
    manifest = training_fixture["manifest"]
    inventory = tmp_path / "references.jsonl"
    build_reference_inventory(manifest, inventory)

    original_manifest = manifest.read_text(encoding="utf-8")
    manifest.write_text(original_manifest + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest content differs"):
        load_inventory_summary(manifest, inventory)
    manifest.write_text(original_manifest, encoding="utf-8")

    with inventory.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"path": "/tmp/extra", "sha256": "0" * 64, "size": 0}) + "\n")
    with pytest.raises(ValueError, match="inventory content differs"):
        load_inventory_summary(manifest, inventory)


def test_inventory_build_refuses_overwrite(training_fixture, tmp_path):
    inventory = tmp_path / "references.jsonl"
    build_reference_inventory(training_fixture["manifest"], inventory)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_reference_inventory(training_fixture["manifest"], inventory)


def test_quick_inventory_validation_never_hashes_referenced_assets(
    training_fixture, tmp_path, monkeypatch
):
    manifest = training_fixture["manifest"]
    inventory = tmp_path / "references.jsonl"
    build_reference_inventory(manifest, inventory)
    original = inventory_module.sha256_file
    hashed_paths = []

    def recording_sha256(path):
        hashed_paths.append(Path(path).resolve())
        return original(path)

    monkeypatch.setattr(inventory_module, "sha256_file", recording_sha256)
    load_inventory_summary(manifest, inventory)
    assert training_fixture["motion"].resolve() not in hashed_paths
    assert training_fixture["embedding"].resolve() not in hashed_paths
    assert manifest.resolve() in hashed_paths
    assert inventory.resolve() in hashed_paths
