from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kimodo.resources import bones


class _SOMA30Fixture:
    def from_SOMASkeleton77(self, rotations: torch.Tensor) -> torch.Tensor:
        # The production class uses the named non-contiguous SOMA mapping.  The
        # unit fixture only isolates converter ownership and output contracts.
        return rotations[:, :30]


class _InlineProcessPool:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def map(self, function, tasks, *, chunksize):
        return map(function, tasks)


def test_local_bones_converter_is_self_contained_and_reusable(tmp_path, monkeypatch) -> None:
    rotations = torch.eye(3).expand(8, 77, 3, 3).clone()
    roots = torch.zeros(8, 3)
    roots[:, 0] = torch.arange(8, dtype=torch.float32)
    monkeypatch.setattr(
        bones,
        "load_motion_file",
        lambda *args, **kwargs: (
            {"local_rot_mats": rotations, "root_positions": roots},
            77,
        ),
    )
    monkeypatch.setattr(bones, "SOMASkeleton30", _SOMA30Fixture)
    source = tmp_path / "sample.bvh"
    source.write_text("fixture", encoding="utf-8")
    output = tmp_path / "sample.npz"

    first = bones.convert_soma_uniform_bvh(source, output)
    second = bones.convert_soma_uniform_bvh(source, output)
    assert first["status"] == "converted"
    assert second["status"] == "reused"
    with np.load(output, allow_pickle=False) as payload:
        assert payload["local_rot_mats"].shape == (8, 30, 3, 3)
        assert payload["root_positions"].shape == (8, 3)
        assert float(payload["fps"].item()) == 30.0
        provenance = json.loads(str(payload["source_provenance_json"].item()))
    assert provenance["converter"] == "kimodo.resources.bones.convert_soma_uniform_bvh"
    assert provenance["conversion_revision"] == bones.CONVERSION_REVISION
    producer = bones.motion_converter_identity()
    assert provenance["motion_converter_producer"] == producer
    assert provenance["producer_fingerprint_sha256"] == producer["producer_fingerprint_sha256"]
    assert first["producer_fingerprint_sha256"] == producer["producer_fingerprint_sha256"]


def test_converter_uses_named_non_contiguous_soma77_to_30_mapping(
    tmp_path, monkeypatch
) -> None:
    frames = 3
    angles = torch.arange(77, dtype=torch.float32) * 0.01
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    rotations = torch.zeros(frames, 77, 3, 3)
    rotations[:, :, 0, 0] = cosine
    rotations[:, :, 0, 1] = -sine
    rotations[:, :, 1, 0] = sine
    rotations[:, :, 1, 1] = cosine
    rotations[:, :, 2, 2] = 1.0
    roots = torch.arange(frames * 3, dtype=torch.float32).reshape(frames, 3)
    monkeypatch.setattr(
        bones,
        "load_motion_file",
        lambda *args, **kwargs: (
            {"local_rot_mats": rotations, "root_positions": roots},
            77,
        ),
    )
    source = tmp_path / "mapping.bvh"
    source.write_text("fixture", encoding="utf-8")
    output = tmp_path / "mapping.npz"

    bones.convert_soma_uniform_bvh(source, output)

    soma30 = bones.SOMASkeleton30()
    soma77_names = soma30.somaskel77.bone_order_names
    named_indices = [soma77_names.index(name) for name in soma30.bone_order_names]
    assert named_indices != list(range(30))
    with np.load(output, allow_pickle=False) as payload:
        assert np.array_equal(payload["local_rot_mats"], rotations[:, named_indices].numpy())
        assert np.array_equal(payload["root_positions"], roots.numpy())


def test_cached_motion_rejects_numerical_dependency_version_drift(
    tmp_path, monkeypatch
) -> None:
    rotations = torch.eye(3).expand(4, 30, 3, 3).clone()
    roots = torch.zeros(4, 3)
    monkeypatch.setattr(
        bones,
        "load_motion_file",
        lambda *args, **kwargs: (
            {"local_rot_mats": rotations, "root_positions": roots},
            30,
        ),
    )
    source = tmp_path / "dependency.bvh"
    source.write_text("fixture", encoding="utf-8")
    output = tmp_path / "dependency.npz"
    bones.motion_converter_identity.cache_clear()
    first = bones.convert_soma_uniform_bvh(source, output)

    real_version = bones.importlib.metadata.version
    with monkeypatch.context() as patch:
        patch.setattr(
            bones.importlib.metadata,
            "version",
            lambda name: "changed-for-regression-test" if name == "scipy" else real_version(name),
        )
        bones.motion_converter_identity.cache_clear()
        changed = bones.motion_converter_identity()
        assert changed["producer_fingerprint_sha256"] != first["producer_fingerprint_sha256"]
        with pytest.raises(ValueError, match="cached motion provenance is stale"):
            bones.convert_soma_uniform_bvh(source, output)
    bones.motion_converter_identity.cache_clear()


def test_batch_metadata_and_inventory_bind_same_producer_as_npz(
    tmp_path, monkeypatch
) -> None:
    rotations = torch.eye(3).expand(4, 30, 3, 3).clone()
    roots = torch.zeros(4, 3)
    monkeypatch.setattr(
        bones,
        "load_motion_file",
        lambda *args, **kwargs: (
            {"local_rot_mats": rotations, "root_positions": roots},
            30,
        ),
    )
    monkeypatch.setattr(bones, "ProcessPoolExecutor", _InlineProcessPool)
    dataset = tmp_path / "dataset"
    source = dataset / "soma_uniform/bvh/230101/sample.bvh"
    source.parent.mkdir(parents=True)
    source.write_text("fixture", encoding="utf-8")
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["move_soma_uniform_path"])
        writer.writeheader()
        writer.writerow({"move_soma_uniform_path": "soma_uniform/bvh/230101/sample.bvh"})
    split = tmp_path / "split.txt"
    split.write_text("230101/sample\n", encoding="utf-8")
    inventory = tmp_path / "conversion/inventory.jsonl"
    output_root = tmp_path / "motions"

    bones.prepare_bones_seed(
        argparse.Namespace(
            dataset_root=str(dataset),
            metadata=str(metadata),
            split_file=str(split),
            output_root=str(output_root),
            inventory=str(inventory),
            source_fps=120.0,
            target_fps=30.0,
            workers=1,
            threads_per_worker=1,
            expected_split_entries=None,
            expected_effective_entries=None,
            expected_missing_sha256=None,
        )
    )

    producer = bones.motion_converter_identity()
    record = json.loads(inventory.read_text(encoding="utf-8"))
    report = json.loads(
        inventory.with_suffix(".jsonl.metadata.json").read_text(encoding="utf-8")
    )
    with np.load(output_root / record["cached"], allow_pickle=False) as payload:
        provenance = json.loads(str(payload["source_provenance_json"].item()))
    assert record["producer_fingerprint_sha256"] == producer["producer_fingerprint_sha256"]
    assert report["motion_converter_producer"] == producer
    assert report["producer_fingerprint_sha256"] == producer["producer_fingerprint_sha256"]
    assert provenance["motion_converter_producer"] == producer


def test_reproduction_source_has_no_flowmatching_import_or_lock() -> None:
    root = Path(__file__).resolve().parents[2]
    assert not (root / "resources/dependencies.lock.yaml").exists()
    for path in (root / "kimodo").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import kimodo_flow" not in source
        assert "from kimodo_flow" not in source


def test_public_docs_do_not_restore_disproven_or_removed_contract_claims() -> None:
    root = Path(__file__).resolve().parents[2]
    current_docs = "\n".join(
        path.read_text(encoding="utf-8") for path in (root / "docs").glob("*.md")
    )
    catalog = (root / "resources/catalog.public.yaml").read_text(encoding="utf-8")

    assert "FM adapter" not in current_docs
    assert "clone/锁定 FM converter" not in current_docs
    assert "semantic_contract_json" not in current_docs
    assert "Non-gated byte-equivalent" not in catalog
    assert "paper profile 不 detach" not in current_docs
    assert "P0/P1/P2 = 0" not in current_docs
