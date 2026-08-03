from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import kimodo.training.engine as training_engine
from kimodo.training.config import TrainingConfig, load_training_config
from kimodo.training.data import validate_paper_data_parity_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_production_profile_enforces_paper_method_defaults():
    config = load_training_config(
        PROJECT_ROOT / "configs/training/kimodo_soma_seed_reproduction.yaml",
        ["runtime.dry_run=true"],
    )
    assert config.paper_method_strict is True
    assert config.model.detach_root_for_body is False
    assert config.data.require_paper_data_parity is True

    config.model.detach_root_for_body = True
    with pytest.raises(ValueError, match="paper_method_strict rejects"):
        config.validate(require_paths=False)


def test_strict_profile_rejects_heading_step_and_runtime_scale_deviations():
    config = load_training_config(
        PROJECT_ROOT / "configs/training/kimodo_soma_seed_reproduction.yaml",
        ["runtime.dry_run=true"],
    )
    config.model.input_first_heading_angle = False
    with pytest.raises(ValueError, match="input_first_heading_angle"):
        config.validate(require_paths=False)

    config.model.input_first_heading_angle = True
    config.runtime.max_steps_override = 10
    with pytest.raises(ValueError, match="max_steps_override"):
        config.validate(require_paths=False)

    config.runtime.max_steps_override = None
    with pytest.raises(RuntimeError, match="world_size=1"):
        training_engine.validate_paper_runtime_scale(config, SimpleNamespace(world_size=1))
    training_engine.validate_paper_runtime_scale(config, SimpleNamespace(world_size=16))


def test_engine_forwards_strict_paper_data_policy(monkeypatch):
    config = TrainingConfig()
    config.data.require_paper_data_parity = True
    captured = {}

    def fake_dataset(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(training_engine, "MotionManifestDataset", fake_dataset)
    assert training_engine.build_training_dataset(config, object()) is not None
    assert captured["require_paper_data_parity"] is True


def _paper_parity_fixture(tmp_path: Path) -> tuple[Path, dict]:
    manifest = tmp_path / "paper.jsonl"
    rows = [
        {
            "id": "para",
            "motion": "base.npz",
            "text": "A person walks.",
            "split": "train",
            "sample_kind": "llm_paraphrase",
            "source_text_id": "original:1",
            "text_generator_model": "Qwen/Qwen3-32B",
            "text_generator_prompt_sha256": "1" * 64,
        },
        {
            "id": "stitch",
            "motion": "stitched.npz",
            "text": "A person walks and then jumps.",
            "split": "train",
            "sample_kind": "stitched_transition",
            "source_motion_ids": ["motion-a", "motion-b"],
            "source_time_ranges": [[0.0, 1.0], [2.0, 3.0]],
            "transition_model_sha256": "2" * 64,
            "transition_frame_range": [30, 45],
        },
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metadata = {
        "output": {
            "path": str(manifest.resolve()),
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
        "paper_parity_gate": {"eligible": True, "blockers": []},
    }
    sidecar = manifest.with_suffix(".jsonl.metadata.json")
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    return manifest, metadata


def test_paper_data_gate_validates_manifest_fingerprint_and_row_provenance(tmp_path):
    manifest, metadata = _paper_parity_fixture(tmp_path)
    assert validate_paper_data_parity_manifest(manifest)["paper_parity_gate"]["eligible"]

    metadata["output"]["sha256"] = "0" * 64
    manifest.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="path/hash does not match"):
        validate_paper_data_parity_manifest(manifest)
