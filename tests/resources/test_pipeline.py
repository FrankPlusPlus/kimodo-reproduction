from __future__ import annotations

import hashlib
import io
import json
import stat
import tarfile
from pathlib import Path

import numpy as np
import pytest

from kimodo.resources.adoption import adopt_legacy_bundle, bind_prepared_bundle
from kimodo.resources.config import ResourceConfigError, load_catalog, load_paths
from kimodo.resources.pipeline import (
    PipelineError,
    _atomic_yaml,
    _safe_extract,
    _validate_conversion_inventory,
    _validate_stats_bundle,
    plan_pipeline,
)
from kimodo.sanitize import sanitize_texts


def _public_catalog() -> Path:
    return Path(__file__).resolve().parents[2] / "resources" / "catalog.public.yaml"


def _paths(path: Path, *, text_device: str = "cpu") -> Path:
    path.write_text(
        f"""
schema_version: 1
resources:
  bones_seed: {{destination: raw, existing_path: null}}
  kimodo_benchmark: {{destination: benchmark, existing_path: null}}
  llm2vec_foundation: {{destination: foundation, existing_path: null}}
  llm2vec_mntp_adapter: {{destination: mntp, existing_path: null}}
  llm2vec_supervised_adapter: {{destination: supervised, existing_path: null}}
pipeline:
  dataset_root: expanded
  prepared_root: prepared
  run_root: runs
  repro_paths_yaml: generated/repro.paths.yaml
  text_device: {text_device}
  motion_workers: 2
  threads_per_worker: 1
  stats_workers: 3
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_pipeline_paths_resolve_from_yaml_and_plan_is_read_only(tmp_path):
    catalog = load_catalog(_public_catalog())
    paths = load_paths(_paths(tmp_path / "paths.yaml"), catalog)
    assert paths.pipeline is not None
    assert paths.pipeline.prepared_root == tmp_path / "prepared"
    assert paths.pipeline.repro_paths_yaml == tmp_path / "generated" / "repro.paths.yaml"
    assert paths.pipeline.motion_workers == 2
    plan = plan_pipeline(paths)
    assert plan["dataset_extract"] == "extract"
    assert plan["raw_manifest"] == "build"
    assert not (tmp_path / "prepared").exists()


def test_pipeline_paths_are_strict(tmp_path):
    catalog = load_catalog(_public_catalog())
    path = _paths(tmp_path / "paths.yaml")
    path.write_text(
        path.read_text(encoding="utf-8") + "  shell_command: curl bad.example\n",
        encoding="utf-8",
    )
    with pytest.raises(ResourceConfigError, match="unknown paths.pipeline keys"):
        load_paths(path, catalog)


def test_pipeline_refuses_to_report_legacy_cached_tree_as_portable(tmp_path):
    catalog = load_catalog(_public_catalog())
    paths = load_paths(_paths(tmp_path / "paths.yaml"), catalog)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    cached = prepared / "train.cached.jsonl"
    cached.write_text("", encoding="utf-8")
    cached.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps({"schema_version": 3, "output": {"sha256": "legacy"}}),
        encoding="utf-8",
    )
    with pytest.raises(PipelineError, match="portable rebuild requires schema 5"):
        plan_pipeline(paths)


def test_pipeline_refuses_legacy_absolute_reference_inventory(tmp_path):
    catalog = load_catalog(_public_catalog())
    paths = load_paths(_paths(tmp_path / "paths.yaml"), catalog)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    inventory = prepared / "train.cached.references.jsonl"
    inventory.write_text("{}\n", encoding="utf-8")
    inventory.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )
    with pytest.raises(PipelineError, match="portable rebuild requires schema 2"):
        plan_pipeline(paths)


def test_safe_extract_rejects_traversal_and_publishes_atomically(tmp_path):
    unsafe = tmp_path / "unsafe.tar.gz"
    with tarfile.open(unsafe, "w:gz") as archive:
        value = b"bad"
        member = tarfile.TarInfo("../escape")
        member.size = len(value)
        archive.addfile(member, io.BytesIO(value))
    with pytest.raises(PipelineError, match="unsafe archive member"):
        _safe_extract(unsafe, tmp_path / "unsafe-output")
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "unsafe-output").exists()

    safe = tmp_path / "safe.tar.gz"
    with tarfile.open(safe, "w:gz") as archive:
        value = b"HIERARCHY\n"
        member = tarfile.TarInfo("soma_uniform/bvh/example.bvh")
        member.size = len(value)
        archive.addfile(member, io.BytesIO(value))
    destination = tmp_path / "expanded"
    assert _safe_extract(safe, destination) == "extract"
    assert (destination / "soma_uniform" / "bvh" / "example.bvh").read_bytes() == value
    assert _safe_extract(safe, destination) == "reuse"


def test_safe_extract_rejects_special_files(tmp_path):
    archive_path = tmp_path / "special.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("soma_uniform/bvh/pipe")
        member.type = tarfile.FIFOTYPE
        archive.addfile(member)
    with pytest.raises(PipelineError, match="unsafe archive member"):
        _safe_extract(archive_path, tmp_path / "expanded")


def test_conversion_inventory_requires_every_declared_source_and_cache(tmp_path):
    dataset = tmp_path / "dataset"
    motions = tmp_path / "motions"
    source = dataset / "a.bvh"
    cached = motions / "a.npz"
    source.parent.mkdir()
    cached.parent.mkdir()
    source.write_bytes(b"source")
    cached.write_bytes(b"cached")
    inventory = tmp_path / "conversion.jsonl"
    inventory.write_text(
        json.dumps(
            {
                "source": "a.bvh",
                "source_sha256": hashlib.sha256(b"source").hexdigest(),
                "cached": "a.npz",
                "cached_sha256": hashlib.sha256(b"cached").hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    inventory.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps({"effective_entries": 1}), encoding="utf-8"
    )
    _validate_conversion_inventory(inventory, dataset_root=dataset, motion_root=motions)
    cached.unlink()
    with pytest.raises(PipelineError, match="conversion output is missing"):
        _validate_conversion_inventory(inventory, dataset_root=dataset, motion_root=motions)


def test_stats_bundle_hashes_shapes_and_generated_paths_are_fail_closed(
    tmp_path, monkeypatch
):
    stats = tmp_path / "stats"
    files = {}
    for group, dimension in (("global_root", 5), ("local_root", 4), ("body", 364)):
        folder = stats / group
        folder.mkdir(parents=True)
        for filename in ("mean.npy", "std.npy"):
            path = folder / filename
            np.save(path, np.ones(dimension, dtype=np.float32), allow_pickle=False)
            files[f"{group}/{filename}"] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
                "dtype": "float32",
                "shape": [dimension],
            }
    (stats / "stats.metadata.json").write_text(
        json.dumps({"schema_version": 3, "files": files}), encoding="utf-8"
    )
    _validate_stats_bundle(stats)
    (stats / "global_root/mean.npy").write_bytes(b"corrupt")
    with pytest.raises(PipelineError, match="stats array is unreadable"):
        _validate_stats_bundle(stats)

    paths = tmp_path / "generated.yaml"
    monkeypatch.delenv("KIMODO_DERIVED_FILE_MODE", raising=False)
    _atomic_yaml(paths, {"schema_version": 1, "value": "generated"})
    assert stat.S_IMODE(paths.stat().st_mode) == 0o664
    _atomic_yaml(paths, {"schema_version": 1, "value": "generated"})
    with pytest.raises(PipelineError, match="refusing to overwrite"):
        _atomic_yaml(paths, {"schema_version": 1, "value": "edited"})


def test_legacy_adoption_is_portable_and_never_reencodes(tmp_path):
    legacy = tmp_path / "legacy"
    motion = legacy / "motions/soma30-30fps/fixture/motion.npz"
    embedding_key = "a" * 64
    embedding = legacy / f"text-cache/{embedding_key}.npy"
    motion.parent.mkdir(parents=True)
    embedding.parent.mkdir(parents=True)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (8, 30, 3, 3)).copy()
    np.savez(
        motion,
        local_rot_mats=rotations,
        root_positions=np.zeros((8, 3), dtype=np.float32),
    )
    np.save(embedding, np.zeros((1, 4096), dtype=np.float32), allow_pickle=False)
    text = "A person walks."
    normalized = sanitize_texts([text])[0]
    content_sha = hashlib.sha256(embedding.read_bytes()).hexdigest()
    provider = "llm2vec:legacy-fixture"
    embedding.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "cache_key": embedding_key,
                "normalized_text_sha256": hashlib.sha256(
                    normalized.encode("utf-8")
                ).hexdigest(),
                "provider_identity": provider,
                "dtype": "float32",
                "shape": [1, 4096],
                "sha256": content_sha,
            }
        ),
        encoding="utf-8",
    )
    row = {
        "id": "fixture",
        "motion": str(motion),
        "text": text,
        "split": "train",
        "source_fps": 30,
        "sample_kind": "full",
    }
    (legacy / "train.raw.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    cached_row = {
        **row,
        "text_cache_key": embedding_key,
        "text_embedding": str(embedding),
    }
    (legacy / "train.cached.jsonl").write_text(
        json.dumps(cached_row) + "\n", encoding="utf-8"
    )
    (legacy / "train.raw.jsonl.metadata.json").write_text(
        json.dumps({"schema_version": 1, "paper_parity_gate": {"eligible": False}}),
        encoding="utf-8",
    )
    (legacy / "train.cached.jsonl.metadata.json").write_text(
        json.dumps({"schema_version": 3, "encoder": provider}), encoding="utf-8"
    )
    legacy_stats = legacy / "stats/repro-soma30-30fps"
    for group, dimension in (("global_root", 5), ("local_root", 4), ("body", 364)):
        folder = legacy_stats / group
        folder.mkdir(parents=True)
        np.save(folder / "mean.npy", np.zeros(dimension, dtype=np.float32))
        np.save(folder / "std.npy", np.ones(dimension, dtype=np.float32))
    (legacy_stats / "stats.metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "preprocessing": {},
                "frame_counts": {
                    "global_root": 8,
                    "local_root": 8,
                    "body": 8,
                },
            }
        ),
        encoding="utf-8",
    )

    adopted = tmp_path / "adopted"
    paths = tmp_path / "config/repro.paths.yaml"
    receipt = adopt_legacy_bundle(
        legacy_root=legacy,
        output_root=adopted,
        run_root=tmp_path / "runs",
        repro_paths_yaml=paths,
        asset_mode="copy",
    )
    assert receipt["mode"] == "verified_legacy_no_reencode"
    assert receipt["full_manifest_entries"] == 1
    assert receipt["data_preflight"]["motion_shape"] == [1, 8, 369]
    adopted_row = json.loads(
        (adopted / "train.cached.jsonl").read_text(encoding="utf-8")
    )
    assert not Path(adopted_row["motion"]).is_absolute()
    assert not Path(adopted_row["text_embedding"]).is_absolute()
    assert adopted_row["frame_count"] == 8
    assert (adopted / adopted_row["text_embedding_metadata"]).is_file()
    assert paths.is_file()

    rebound_paths = tmp_path / "relocated/repro.paths.yaml"
    rebound = bind_prepared_bundle(
        prepared_root=adopted,
        run_root=tmp_path / "new-runs",
        repro_paths_yaml=rebound_paths,
    )
    assert rebound["status"] == "prepared_bundle_bound"
    assert rebound["reference_verification"] == "full_content_verified"
    assert rebound["data_preflight"]["text_shape"] == [1, 1, 4096]
    assert rebound_paths.is_file()

    reused = adopt_legacy_bundle(
        legacy_root=legacy,
        output_root=adopted,
        run_root=tmp_path / "runs",
        repro_paths_yaml=paths,
        asset_mode="copy",
    )
    assert reused["status"] == "repro_train_ready_reused"
    assert reused["reference_verification"] == "full_content_verified"
