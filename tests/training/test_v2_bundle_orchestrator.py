from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kimodo.data_pipeline.reference_inventory import build_reference_inventory
from kimodo.data_pipeline.v2.response_selection_cli import select as select_response
from kimodo.data_pipeline.v2.v2_bundle_publish_cli import (
    _validate_preflight,
    _validate_stats,
    publish,
)
from kimodo.data_pipeline.v2.v2_lineage_cli import validate_lineage
from kimodo.data_pipeline.v2.v2_resource_state_cli import verify_resource_state

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/internal/build_v2_bundle.sh"
DELIVERY_WATCHER = PROJECT_ROOT / "scripts/internal/watch_v2_delivery.sh"
PACKAGE = PROJECT_ROOT / "scripts/internal/package_v2_bundle.sh"


def test_delivery_package_has_exclusive_atomic_publication():
    source = PACKAGE.read_text(encoding="utf-8")
    assert 'exec 7>"${delivery}.lock"' in source
    assert "flock 7" in source
    assert 'mv -T -- "${staging}" "${delivery}"' in source


def test_resource_state_verifies_every_named_output(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"verified")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "resource-state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "v2_train_ready",
                "outputs": {"artifact_sha256": digest},
                "output_paths": {"artifact_sha256": artifact.name},
            }
        ),
        encoding="utf-8",
    )
    assert verify_resource_state(tmp_path)["verified_outputs"]["artifact_sha256"][
        "sha256"
    ] == digest
    artifact.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_resource_state(tmp_path)


def test_resource_state_detects_finite_stats_mutation(tmp_path):
    stats = tmp_path / "stats" / "body" / "mean.npy"
    stats.parent.mkdir(parents=True)
    np.save(stats, np.ones(364, dtype=np.float32))
    digest = hashlib.sha256(stats.read_bytes()).hexdigest()
    (tmp_path / "resource-state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "v2_train_ready",
                "outputs": {"stats_body_mean_sha256": digest},
                "output_paths": {
                    "stats_body_mean_sha256": "stats/body/mean.npy"
                },
            }
        ),
        encoding="utf-8",
    )
    verify_resource_state(tmp_path)
    np.save(stats, np.full(364, 2.0, dtype=np.float32))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_resource_state(tmp_path)


def test_lineage_rejects_cached_manifest_from_different_raw(tmp_path):
    root = tmp_path / "v2.building"
    provenance = root / "provenance"
    provenance.mkdir(parents=True)
    responses = provenance / "responses.jsonl"
    responses.write_text('{}\n', encoding="utf-8")
    response_sha = hashlib.sha256(responses.read_bytes()).hexdigest()
    response_metadata = responses.with_suffix(responses.suffix + ".metadata.json")
    response_metadata.write_text(
        json.dumps(
            {
                "producer_identity_sha256": "producer",
                "requests": {"sha256": "requests"},
            }
        ),
        encoding="utf-8",
    )
    response_metadata_sha = hashlib.sha256(response_metadata.read_bytes()).hexdigest()

    def pair(name, payload, metadata):
        path = root / name
        path.write_text(payload, encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = {**metadata, "output": {"sha256": digest, "entries": 1}}
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        sidecar.write_text(json.dumps(metadata), encoding="utf-8")
        return path, digest, hashlib.sha256(sidecar.read_bytes()).hexdigest()

    _raw, raw_sha, raw_meta_sha = pair(
        "train.raw.jsonl",
        '{}\n',
        {
            "sources": {
                "llm_responses": [
                    {
                        "sha256": response_sha,
                        "metadata_sha256": response_metadata_sha,
                        "producer_identity_sha256": "producer",
                        "requests_sha256": "requests",
                    }
                ]
            }
        },
    )
    _, llm_raw_sha, llm_raw_meta_sha = pair(
        "train.llm.raw.jsonl",
        '{}\n',
        {"source_v2_raw": {"sha256": raw_sha, "metadata_sha256": raw_meta_sha}},
    )
    _, llm_cached_sha, llm_cached_meta_sha = pair(
        "train.llm.cached.jsonl",
        '{}\n',
        {
            "source_manifest_sha256": llm_raw_sha,
            "source_manifest_metadata_sha256": llm_raw_meta_sha,
        },
    )
    pair(
        "train.cached.jsonl",
        '{}\n',
        {
            "sources": {
                "v2_raw": {"sha256": "wrong"},
                "llm_cached": {
                    "sha256": llm_cached_sha,
                    "metadata_sha256": llm_cached_meta_sha,
                },
            }
        },
    )
    with pytest.raises(ValueError, match="stale selected-response lineage"):
        validate_lineage(root, responses, "cached")


def test_lineage_rejects_same_response_content_with_stale_metadata(tmp_path):
    root = tmp_path / "v2.building"
    provenance = root / "provenance"
    provenance.mkdir(parents=True)
    responses = provenance / "responses.jsonl"
    responses.write_text('{}\n', encoding="utf-8")
    response_sha = hashlib.sha256(responses.read_bytes()).hexdigest()
    metadata = responses.with_suffix(responses.suffix + ".metadata.json")
    metadata.write_text(
        json.dumps(
            {
                "producer_identity_sha256": "current-producer",
                "requests": {"sha256": "current-requests"},
            }
        ),
        encoding="utf-8",
    )
    raw = root / "train.raw.jsonl"
    raw.write_text('{}\n', encoding="utf-8")
    raw.with_suffix(raw.suffix + ".metadata.json").write_text(
        json.dumps(
            {
                "output": {
                    "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                    "entries": 1,
                },
                "sources": {
                    "llm_responses": [
                        {
                            "sha256": response_sha,
                            "metadata_sha256": "stale-metadata",
                            "producer_identity_sha256": "old-producer",
                            "requests_sha256": "old-requests",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stale selected-response metadata"):
        validate_lineage(root, responses, "raw")


def test_v2_bundle_script_has_a_complete_ordered_plan():
    result = subprocess.run(
        [str(SCRIPT), "plan"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == [
        "require-selected-final-response",
        "audit-quality-gate",
        "independent-expert-review-gate",
        "build-raw-manifest",
        "extract-llm-lane",
        "wait-for-text-gpu",
        "verify-v1-v2-embedding-canary",
        "cache-llm2vec",
        "compose-cached-manifest",
        "fit-v2-stats",
        "build-and-verify-reference-inventory",
        "real-batch-preflight",
        "validate-and-atomic-publish",
    ]


def test_v2_bundle_script_rejects_a_nonbuilding_root(tmp_path):
    result = subprocess.run(
        [str(SCRIPT), "status"],
        cwd=PROJECT_ROOT,
        env={"PATH": "/usr/bin:/bin", "KIMODO_V2_ROOT": str(tmp_path / "v2")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "must end in .building" in result.stderr


def test_delivery_watcher_recognizes_an_existing_verified_delivery(tmp_path):
    final = tmp_path / "v2"
    final.mkdir()
    resource_state = final / "resource-state.json"
    resource_state.write_text('{"status":"v2_train_ready"}\n', encoding="utf-8")
    resource_state_sha = hashlib.sha256(resource_state.read_bytes()).hexdigest()
    delivery = tmp_path / "v2.delivery"
    delivery.mkdir()
    archive = delivery / "v2.tar.zst"
    archive.write_bytes(b"portable-v2")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = delivery / "v2.tar.zst.sha256"
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    (delivery / "v2.tar.zst.metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "v2_delivery_archive_verified_after_relocation",
                "bundle_name": final.name,
                "bundle_resource_state_sha256": resource_state_sha,
                "archive": {
                    "path": archive.name,
                    "sha256": digest,
                    "size": archive.stat().st_size,
                },
                "checksum_file": checksum.name,
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [str(DELIVERY_WATCHER)],
        cwd=PROJECT_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "KIMODO_PYTHON": sys.executable,
            "KIMODO_V2_FINAL_ROOT": str(final),
            "KIMODO_V2_DELIVERY_DIR": str(delivery),
            "KIMODO_V2_DELIVERY_LOG": str(tmp_path / "watcher.log"),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "verified delivery already exists" in result.stdout


def test_delivery_watcher_fails_if_builder_disappears_before_publish(tmp_path):
    result = subprocess.run(
        [str(DELIVERY_WATCHER)],
        cwd=PROJECT_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "KIMODO_PYTHON": sys.executable,
            "KIMODO_V2_BUILDER_PID": "99999999",
            "KIMODO_V2_FINAL_ROOT": str(tmp_path / "missing-v2"),
            "KIMODO_V2_DELIVERY_DIR": str(tmp_path / "missing-v2.delivery"),
            "KIMODO_V2_DELIVERY_LOG": str(tmp_path / "watcher.log"),
            "KIMODO_V2_DELIVERY_POLL_SECONDS": "1",
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert "exited before publishing" in result.stdout


def test_v2_publish_validators_bind_stats_and_real_batch(tmp_path):
    manifest = tmp_path / "train.cached.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    stats = tmp_path / "stats"
    records = {}
    for group, width in (("global_root", 5), ("local_root", 4), ("body", 364)):
        folder = stats / group
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("mean.npy", "std.npy"):
            path = folder / name
            np.save(path, np.ones(width, dtype=np.float32))
            records[f"{group}/{name}"] = {
                "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
                "dtype": "float32",
                "shape": [width],
            }
    manifest_sha = __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()
    (stats / "stats.metadata.json").write_text(
        json.dumps({"schema_version": 3, "manifest_sha256": manifest_sha, "files": records}),
        encoding="utf-8",
    )
    assert _validate_stats(stats, manifest_sha)["schema_version"] == 3

    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
                {
                    "event": "kimodo_full_data_preflight_passed",
                    "manifest_entries_validated": 1,
                    "dataset_entries": 1,
                    "excluded_short_entries": 0,
                    "motion_shape": [1, 300, 369],
                    "text_shape": [1, 1, 4096],
                    "bindings": {
                        "manifest_sha256": "a",
                        "inventory_sha256": "b",
                        "inventory_metadata_sha256": "c",
                        "stats_metadata_sha256": "d",
                    },
                }
            ),
            encoding="utf-8",
        )
    assert _validate_preflight(
        preflight,
        1,
        manifest_sha256="a",
        inventory_sha256="b",
        inventory_metadata_sha256="c",
        stats_metadata_sha256="d",
    )["event"] == "kimodo_full_data_preflight_passed"


def test_v2_publisher_full_verifies_and_atomically_renames(tmp_path):
    building = tmp_path / "v2.building"
    final = tmp_path / "v2"
    motion = building / "motions" / "clip.npz"
    embedding = building / "text-cache-v2-llm" / "text.npy"
    embedding_metadata = embedding.with_suffix(embedding.suffix + ".metadata.json")
    motion.parent.mkdir(parents=True)
    embedding.parent.mkdir(parents=True)
    motion.write_bytes(b"motion")
    np.save(embedding, np.ones((1, 4096), dtype=np.float32))
    embedding_sha = hashlib.sha256(embedding.read_bytes()).hexdigest()
    embedding_metadata.write_text(
        json.dumps(
            {
                "sha256": embedding_sha,
                "size": embedding.stat().st_size,
                "dtype": "float32",
                "shape": [1, 4096],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = building / "train.cached.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample",
                "motion": "motions/clip.npz",
                "text_embedding": "text-cache-v2-llm/text.npy",
                    "text_embedding_metadata": "text-cache-v2-llm/text.npy.metadata.json",
                    "text_embedding_sha256": embedding_sha,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.with_suffix(manifest.suffix + ".metadata.json").write_text(
        json.dumps(
            {
                "output": {"sha256": manifest_sha, "entries": 1},
                "paper_parity_gate": {"eligible": False},
                "leakage_gate": {"eligible": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stats = building / "stats" / "repro-soma30-30fps"
    stats_records = {}
    for group, width in (("global_root", 5), ("local_root", 4), ("body", 364)):
        folder = stats / group
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("mean.npy", "std.npy"):
            path = folder / name
            np.save(path, np.ones(width, dtype=np.float32))
            stats_records[f"{group}/{name}"] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
                "dtype": "float32",
                "shape": [width],
            }
    (stats / "stats.metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "manifest_sha256": manifest_sha,
                "files": stats_records,
                "processed_spans": 1,
                "frame_counts": {"global_root": 2, "local_root": 2, "body": 2},
            }
        ),
        encoding="utf-8",
    )

    responses = building / "provenance" / "responses.jsonl"
    responses.parent.mkdir(parents=True)
    responses.write_text('{"request_id":"one"}\n', encoding="utf-8")
    responses_sha = hashlib.sha256(responses.read_bytes()).hexdigest()
    requests = responses.parent / "requests.jsonl"
    requests.write_text('{"request_id":"one"}\n', encoding="utf-8")
    requests_sha = hashlib.sha256(requests.read_bytes()).hexdigest()
    receipts = responses.with_suffix(responses.suffix + ".api-receipts.jsonl")
    receipts.write_text('{"response_id":"one"}\n', encoding="utf-8")
    receipts_sha = hashlib.sha256(receipts.read_bytes()).hexdigest()
    producer_identity = {"kind": "test"}
    producer_sha = hashlib.sha256(
        json.dumps(producer_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    responses_metadata_path = responses.with_suffix(
        responses.suffix + ".metadata.json"
    )
    responses_metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "generator": "kimodo.training.semantic_response_finalize_cli",
                "output": {"sha256": responses_sha, "entries": 1},
                "requests": {"sha256": requests_sha},
                "producer_identity": producer_identity,
                "producer_identity_sha256": producer_sha,
                "api_receipts": {"sha256": receipts_sha},
                "quality": {"invalid_published": 0},
                "semantic_finalization": {
                    "remaining_expert_required_count_facts": 0,
                    "requests_sha256": requests_sha,
                },
            }
        ),
        encoding="utf-8",
    )
    responses_metadata_sha = hashlib.sha256(
        responses_metadata_path.read_bytes()
    ).hexdigest()
    selection = responses.parent / "selected.json"
    select_response(
        argparse.Namespace(
            requests=str(requests), responses=str(responses), output=str(selection)
        )
    )

    raw = building / "train.raw.jsonl"
    raw.write_text('{"id":"raw"}\n', encoding="utf-8")
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    raw_sidecar = raw.with_suffix(raw.suffix + ".metadata.json")
    raw_sidecar.write_text(
        json.dumps(
            {
                "output": {"sha256": raw_sha, "entries": 1},
                "sources": {
                    "llm_responses": [
                        {
                            "sha256": responses_sha,
                            "metadata_sha256": responses_metadata_sha,
                            "producer_identity_sha256": producer_sha,
                            "requests_sha256": requests_sha,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    raw_meta_sha = hashlib.sha256(raw_sidecar.read_bytes()).hexdigest()
    llm_raw = building / "train.llm.raw.jsonl"
    llm_raw.write_text('{"id":"llm-raw"}\n', encoding="utf-8")
    llm_raw_sha = hashlib.sha256(llm_raw.read_bytes()).hexdigest()
    llm_raw_sidecar = llm_raw.with_suffix(llm_raw.suffix + ".metadata.json")
    llm_raw_sidecar.write_text(
        json.dumps(
            {
                "output": {"sha256": llm_raw_sha, "entries": 1},
                "source_v2_raw": {
                    "sha256": raw_sha,
                    "metadata_sha256": raw_meta_sha,
                },
            }
        ),
        encoding="utf-8",
    )
    llm_raw_meta_sha = hashlib.sha256(llm_raw_sidecar.read_bytes()).hexdigest()
    llm_cached = building / "train.llm.cached.jsonl"
    llm_cached.write_text('{"id":"llm-cached"}\n', encoding="utf-8")
    llm_cached_sha = hashlib.sha256(llm_cached.read_bytes()).hexdigest()
    llm_cached_sidecar = llm_cached.with_suffix(
        llm_cached.suffix + ".metadata.json"
    )
    llm_cached_sidecar.write_text(
        json.dumps(
            {
                "output": {"sha256": llm_cached_sha, "entries": 1},
                "source_manifest_sha256": llm_raw_sha,
                "source_manifest_metadata_sha256": llm_raw_meta_sha,
            }
        ),
        encoding="utf-8",
    )
    llm_cached_meta_sha = hashlib.sha256(llm_cached_sidecar.read_bytes()).hexdigest()
    manifest_metadata_path = manifest.with_suffix(manifest.suffix + ".metadata.json")
    manifest_metadata = json.loads(manifest_metadata_path.read_text(encoding="utf-8"))
    manifest_metadata["sources"] = {
        "v2_raw": {"sha256": raw_sha, "metadata_sha256": raw_meta_sha},
        "llm_cached": {
            "sha256": llm_cached_sha,
            "metadata_sha256": llm_cached_meta_sha,
        },
    }
    manifest_metadata_path.write_text(json.dumps(manifest_metadata), encoding="utf-8")
    review_sample = building / "provenance" / "review.jsonl"
    review_sample.write_text('{"request_id":"one"}\n', encoding="utf-8")
    review_sample_sha = hashlib.sha256(review_sample.read_bytes()).hexdigest()
    quality = building / "provenance" / "quality.json"
    quality.parent.mkdir(parents=True, exist_ok=True)
    quality.write_text(
        json.dumps(
            {
                "quality_gate": {"eligible": True},
                "coverage": {"requests": 1, "responses": 1, "missing": 0, "unexpected": 0},
                "sources": {
                    "requests": {"sha256": requests_sha},
                    "responses": [
                        {
                            "sha256": responses_sha,
                            "metadata_sha256": responses_metadata_sha,
                            "producer_identity_sha256": producer_sha,
                            "requests_sha256": requests_sha,
                        }
                    ],
                },
                "review_sample": {"path": str(review_sample), "sha256": review_sample_sha},
            }
        ),
        encoding="utf-8",
    )
    quality_sha = hashlib.sha256(quality.read_bytes()).hexdigest()
    expert_verdicts = building / "provenance" / "expert-verdicts.jsonl"
    expert_verdicts.write_text('{"request_id":"one","verdict":"pass"}\n', encoding="utf-8")
    expert_verdicts_sha = hashlib.sha256(expert_verdicts.read_bytes()).hexdigest()
    expert = building / "provenance" / "expert.json"
    expert.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "approved",
                "bindings": {
                    "responses_sha256": responses_sha,
                    "quality_report_sha256": quality_sha,
                    "review_sample_sha256": review_sample_sha,
                    "verdicts_sha256": expert_verdicts_sha,
                },
                "review": {
                    "reviewed_unique_requests": 1200,
                    "unresolved_critical_errors": 0,
                    "major_semantic_error_rate": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    canary = building / "provenance" / "canary.json"
    canary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "passed": True,
                "sample_count": 16,
                "observed": {"max_abs_error": 0.0, "min_cosine_similarity": 1.0},
            }
        ),
        encoding="utf-8",
    )
    inventory = building / "train.cached.references.jsonl"
    build_reference_inventory(manifest, inventory)
    inventory_sha = hashlib.sha256(inventory.read_bytes()).hexdigest()
    inventory_metadata = inventory.with_suffix(inventory.suffix + ".metadata.json")
    inventory_metadata_sha = hashlib.sha256(inventory_metadata.read_bytes()).hexdigest()
    stats_metadata_sha = hashlib.sha256((stats / "stats.metadata.json").read_bytes()).hexdigest()
    preflight = building / "provenance" / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "event": "kimodo_full_data_preflight_passed",
                "manifest_entries_validated": 1,
                "dataset_entries": 1,
                "excluded_short_entries": 0,
                "motion_shape": [1, 300, 369],
                "text_shape": [1, 1, 4096],
                "bindings": {
                    "manifest_sha256": manifest_sha,
                    "inventory_sha256": inventory_sha,
                    "inventory_metadata_sha256": inventory_metadata_sha,
                    "stats_metadata_sha256": stats_metadata_sha,
                },
            }
        ),
        encoding="utf-8",
    )

    result = publish(
        argparse.Namespace(
            building_root=str(building),
            final_root=str(final),
            manifest=str(manifest),
            inventory=str(inventory),
            stats=str(stats),
            quality_report=str(quality),
            responses=str(responses),
            response_selection=str(selection),
            expert_review=str(expert),
            expert_verdicts=str(expert_verdicts),
            embedding_canary=str(canary),
            preflight_report=str(preflight),
            expected_entries=1,
            paths_name="repro.paths.yaml",
        )
    )

    assert result["status"] == "v2_train_ready"
    assert not building.exists()
    assert (final / "resource-state.json").is_file()
    receipt = json.loads((final / "resource-state.json").read_text(encoding="utf-8"))
    assert receipt["reference_verification"]["path"] == "train.cached.references.jsonl"
    paths_text = (final / "repro.paths.yaml").read_text(encoding="utf-8")
    assert "${oc.env:KIMODO_DATA_ROOT}/train.cached.jsonl" in paths_text
    assert "${oc.env:KIMODO_RUN_ROOT}/v2-1m-production" in paths_text
