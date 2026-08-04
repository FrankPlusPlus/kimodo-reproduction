from __future__ import annotations

import argparse
import csv
import json
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest
from omegaconf.errors import ConfigKeyError

from kimodo.training import text_cache_cli
from kimodo.training.config import load_training_config
from kimodo.training.data import load_manifest
from kimodo.training.manifest_cli import build_manifest
from kimodo.training.reference_inventory import (
    build_reference_inventory,
    inventory_metadata_path,
    load_inventory_summary,
    sha256_file,
    verify_reference_inventory_full,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_manifest_and_inventory_survive_bundle_relocation(tmp_path):
    first = tmp_path / "first" / "bundle"
    motion = first / "motions" / "230101" / "motion.npz"
    motion.parent.mkdir(parents=True)
    motion.write_bytes(b"portable-motion")
    metadata = first / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "take_date",
                "move_soma_uniform_path",
                "content_natural_desc_1",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": "motion",
                "take_date": "230101",
                "move_soma_uniform_path": "bvh/230101/motion.npz",
                "content_natural_desc_1": "A person walks.",
            }
        )
    split = first / "split.txt"
    split.write_text("230101/motion\n", encoding="utf-8")
    manifest = first / "train.raw.jsonl"
    build_manifest(
        argparse.Namespace(
            metadata=str(metadata),
            temporal_labels=None,
            split_file=str(split),
            dataset_root=str(first),
            motion_cache_root=str(first / "motions"),
            motion_cache_fps=30.0,
            skeleton="soma_uniform",
            output=str(manifest),
            split_name="train",
            source_fps=120.0,
            full_repeats=1,
            event_repeats=1,
            combined_event_repeats=1,
            allow_missing=False,
            path_mode="relative",
        )
    )
    row = json.loads(manifest.read_text(encoding="utf-8"))
    sidecar = json.loads(
        manifest.with_suffix(".jsonl.metadata.json").read_text(encoding="utf-8")
    )
    assert not Path(row["motion"]).is_absolute()
    assert sidecar["output"]["path"] == "train.raw.jsonl"

    inventory = first / "train.references.jsonl"
    build_reference_inventory(manifest, inventory)
    inventory_bytes = inventory.read_bytes()
    inventory_metadata_bytes = inventory_metadata_path(inventory).read_bytes()

    second = tmp_path / "different-server" / "bundle"
    shutil.copytree(first, second)
    moved_manifest = second / manifest.name
    moved_inventory = second / inventory.name
    assert len(load_manifest(moved_manifest, "train")) == 1
    assert moved_inventory.read_bytes() == inventory_bytes
    assert inventory_metadata_path(moved_inventory).read_bytes() == inventory_metadata_bytes
    assert load_inventory_summary(moved_manifest, moved_inventory)["schema_version"] == 2
    assert verify_reference_inventory_full(moved_manifest, moved_inventory)[
        "verification"
    ] == "full_content_verified"


def test_reference_inventory_v1_absolute_schema_remains_readable(
    training_fixture, tmp_path
):
    manifest = training_fixture["manifest"].resolve()
    inventory = tmp_path / "legacy.references.jsonl"
    build_reference_inventory(manifest, inventory)

    records = []
    for line in inventory.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record["path"] = str((manifest.parent / record["path"]).resolve())
        records.append(record)
    inventory.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    metadata_path = inventory_metadata_path(inventory)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = sha256_file(inventory)
    metadata["schema_version"] = 1
    metadata["manifest"]["path"] = str(manifest)
    metadata["inventory"].update(
        path=str(inventory.resolve()),
        sha256=digest,
        size=inventory.stat().st_size,
    )
    metadata["aggregate"] = {
        "algorithm": "sha256(canonical-jsonl-v1)",
        "sha256": digest,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    assert load_inventory_summary(manifest, inventory)["schema_version"] == 1
    assert verify_reference_inventory_full(manifest, inventory)["verification"] == (
        "full_content_verified"
    )


def _identity_args(root: Path) -> argparse.Namespace:
    for name, payload in (
        ("foundation", b"base"),
        ("mntp", b"mntp"),
        ("supervised", b"supervised"),
    ):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "weights.bin").write_bytes(payload)
    lock = root / "models.lock.json"
    lock.write_text('{"models":"same"}\n', encoding="utf-8")
    return argparse.Namespace(
        provider="local",
        foundation_model=str(root / "foundation"),
        foundation_repo_id="NousResearch/Meta-Llama-3-8B-Instruct",
        foundation_revision="foundation-sha",
        mntp_model=str(root / "mntp"),
        mntp_repo_id="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        mntp_revision="mntp-sha",
        supervised_model=str(root / "supervised"),
        supervised_repo_id=(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
        ),
        supervised_revision="supervised-sha",
        model_lock=str(lock),
    )


def test_text_cache_functional_identity_ignores_server_paths_and_git_commit(tmp_path):
    first = _identity_args(tmp_path / "server-a")
    second = _identity_args(tmp_path / "other" / "server-b")

    def identity(args):
        provenance = text_cache_cli._cache_provenance(args)
        provenance["repo_git_commit"] = str(args.foundation_model)
        return text_cache_cli._bind_identity(
            text_cache_cli._functional_encoder_identity(args),
            text_cache_cli._encoder_artifacts(args),
            provenance,
        )

    assert identity(first) == identity(second)
    key_a = text_cache_cli._cache_key("A person walks.", identity(first))
    key_b = text_cache_cli._cache_key("A person walks.", identity(second))
    assert key_a == key_b

    (Path(second.foundation_model) / "README.md").write_text(
        "storage-local documentation", encoding="utf-8"
    )
    (Path(second.foundation_model) / ".gitattributes").write_text(
        "*.bin filter=lfs", encoding="utf-8"
    )
    assert identity(first) == identity(second)

    (Path(second.foundation_model) / "weights.bin").write_bytes(b"changed")
    assert identity(first) != identity(second)


def test_text_cache_metadata_reads_legacy_and_portable_schemas(tmp_path):
    metadata = tmp_path / "cached.jsonl.metadata.json"
    for schema in (3, 4, 5):
        metadata.write_text(json.dumps({"schema_version": schema}), encoding="utf-8")
        assert text_cache_cli.load_text_cache_metadata(metadata)["schema_version"] == schema
    metadata.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported text-cache"):
        text_cache_cli.load_text_cache_metadata(metadata)


def test_atomic_text_cache_artifacts_are_shared_storage_readable(tmp_path, monkeypatch):
    monkeypatch.delenv("KIMODO_DERIVED_FILE_MODE", raising=False)
    embedding = tmp_path / "embedding.npy"
    text_cache_cli._atomic_save_embedding(
        embedding, np.zeros((1, 4096), dtype=np.float32)
    )
    assert stat.S_IMODE(embedding.stat().st_mode) == 0o664


def test_paths_overlay_and_training_overlay_have_strict_precedence(tmp_path):
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        """
schema_version: 1
data:
  manifest: /assets/train.jsonl
  reference_inventory: /assets/train.references.jsonl
model:
  stats_path: /assets/stats
runtime:
  output_dir: /runs/base
""".lstrip(),
        encoding="utf-8",
    )
    overlay = tmp_path / "hardware.yaml"
    overlay.write_text(
        """
runtime:
  batch_size: 7
  output_dir: /runs/overlay
""".lstrip(),
        encoding="utf-8",
    )
    config = load_training_config(
        PROJECT_ROOT / "configs/training/kimodo_tiny_smoke.yaml",
        ["runtime.batch_size=9", "runtime.dry_run=true"],
        paths=paths,
        overlays=[overlay],
    )
    assert config.data.manifest == "/assets/train.jsonl"
    assert config.model.stats_path == "/assets/stats"
    assert config.runtime.output_dir == "/runs/overlay"
    assert config.runtime.batch_size == 9


def test_paths_yaml_rejects_hyperparameters_and_overlay_rejects_unknown_keys(tmp_path):
    invalid_paths = tmp_path / "invalid-paths.yaml"
    invalid_paths.write_text(
        "schema_version: 1\nruntime:\n  batch_size: 4\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="disallowed fields: runtime.batch_size"):
        load_training_config(
            PROJECT_ROOT / "configs/training/kimodo_tiny_smoke.yaml",
            ["runtime.dry_run=true"],
            paths=invalid_paths,
        )

    invalid_overlay = tmp_path / "invalid-overlay.yaml"
    invalid_overlay.write_text("runtime:\n  made_up: true\n", encoding="utf-8")
    with pytest.raises(ConfigKeyError):
        load_training_config(
            PROJECT_ROOT / "configs/training/kimodo_tiny_smoke.yaml",
            ["runtime.dry_run=true"],
            overlays=[invalid_overlay],
        )
