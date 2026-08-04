from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kimodo.training.config import TrainingConfig
from kimodo.training.engine import KimodoTrainer


def _config(fixture, output: Path, steps: int, accumulation: int = 1) -> TrainingConfig:
    config = TrainingConfig()
    config.data.manifest = str(fixture["manifest"])
    config.data.max_seconds = 1.0
    config.data.num_workers = 0
    config.data.pin_memory = False
    config.data.persistent_workers = False
    config.model.stats_path = str(fixture["stats"])
    config.model.llm_dim = 16
    config.model.num_text_tokens_override = 2
    config.model.latent_dim = 16
    config.model.ff_size = 32
    config.model.num_layers = 1
    config.model.num_heads = 4
    config.curriculum.phase1_steps = 1
    config.curriculum.phase2_steps = 1
    config.curriculum.sparse_keyframes_max = 2
    config.runtime.output_dir = str(output)
    config.runtime.device = "cpu"
    config.runtime.precision = "fp32"
    config.runtime.batch_size = 2
    config.runtime.gradient_accumulation_steps = accumulation
    config.runtime.log_every = 1
    config.runtime.checkpoint_every = 1
    config.runtime.max_steps_override = steps
    config.ema.update_every = 1
    config.validate()
    return config


def _state(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def test_two_phase_training_checkpoint_exact_resume(training_fixture, tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted = _config(training_fixture, uninterrupted_dir, 2)
    KimodoTrainer(uninterrupted, project_root).train()

    resumed_dir = tmp_path / "resumed"
    first_leg = _config(training_fixture, resumed_dir, 1)
    KimodoTrainer(first_leg, project_root).train()
    resume_path = resumed_dir / "checkpoints" / "step-000000001.pt"
    second_leg = _config(training_fixture, resumed_dir, 2)
    second_leg.runtime.resume = str(resume_path)
    KimodoTrainer(second_leg, project_root).train()

    expected = _state(uninterrupted_dir / "checkpoints" / "step-000000002.pt")
    actual = _state(resumed_dir / "checkpoints" / "step-000000002.pt")
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    assert (resumed_dir / "exports" / "step-000000002" / "config.yaml").is_file()
    assert (resumed_dir / "exports" / "step-000000002" / "model.pt").is_file()


def test_epoch_boundary_resume_with_accumulation(training_fixture, tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    uninterrupted_dir = tmp_path / "uninterrupted-accum"
    KimodoTrainer(_config(training_fixture, uninterrupted_dir, 2, accumulation=2), project_root).train()

    resumed_dir = tmp_path / "resumed-accum"
    KimodoTrainer(_config(training_fixture, resumed_dir, 1, accumulation=2), project_root).train()
    first_checkpoint = resumed_dir / "checkpoints" / "step-000000001.pt"
    saved = torch.load(first_checkpoint, map_location="cpu", weights_only=False)
    assert saved["batch_in_epoch"] == 2
    second_leg = _config(training_fixture, resumed_dir, 2, accumulation=2)
    second_leg.runtime.resume = str(first_checkpoint)
    KimodoTrainer(second_leg, project_root).train()

    expected = _state(uninterrupted_dir / "checkpoints" / "step-000000002.pt")
    actual = _state(resumed_dir / "checkpoints" / "step-000000002.pt")
    assert all(torch.equal(expected[key], actual[key]) for key in expected)


def test_fresh_run_rejects_nonempty_output_directory(training_fixture, tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "belongs-to-another-run"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output_dir is not empty"):
        KimodoTrainer(_config(training_fixture, output, 1), project_root)

    assert marker.read_text(encoding="utf-8") == "preserve me"


def test_resume_rejects_recipe_or_referenced_data_changes(training_fixture, tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "guarded"
    KimodoTrainer(_config(training_fixture, output, 1), project_root).train()
    checkpoint = output / "checkpoints" / "step-000000001.pt"

    changed_recipe = _config(training_fixture, output, 2)
    changed_recipe.runtime.resume = str(checkpoint)
    changed_recipe.loss.root_position = 11.0
    with pytest.raises(ValueError, match="training-critical config"):
        KimodoTrainer(changed_recipe, project_root)

    embedding = np.load(training_fixture["embedding"])
    np.save(training_fixture["embedding"], embedding + 1.0)
    changed_data = _config(training_fixture, output, 2)
    changed_data.runtime.resume = str(checkpoint)
    with pytest.raises(ValueError, match="provenance mismatch"):
        KimodoTrainer(changed_data, project_root)


def test_resume_lineage_rejects_foreign_output_and_allows_explicit_fork(
    training_fixture, tmp_path
):
    project_root = Path(__file__).resolve().parents[2]
    parent = tmp_path / "parent"
    KimodoTrainer(_config(training_fixture, parent, 1), project_root).train()
    checkpoint = parent / "checkpoints" / "step-000000001.pt"

    child = tmp_path / "child"
    accidental = _config(training_fixture, child, 2)
    accidental.runtime.resume = str(checkpoint)
    with pytest.raises(ValueError, match="in-place resume checkpoint must belong"):
        KimodoTrainer(accidental, project_root)

    forked = _config(training_fixture, child, 2)
    forked.runtime.resume = str(checkpoint)
    forked.runtime.resume_mode = "fork"
    KimodoTrainer(forked, project_root).train()
    lineage = json.loads((child / "provenance.json").read_text(encoding="utf-8"))[
        "resume_lineage"
    ]
    assert lineage["mode"] == "fork"
    assert len(lineage["parent_checkpoint_sha256"]) == 64


def test_resume_rejects_total_steps_behind_checkpoint(training_fixture, tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    parent = tmp_path / "parent-behind"
    KimodoTrainer(_config(training_fixture, parent, 2), project_root).train()
    checkpoint = parent / "checkpoints/step-000000002.pt"
    child = tmp_path / "child-behind"
    config = _config(training_fixture, child, 1)
    config.runtime.resume = str(checkpoint)
    config.runtime.resume_mode = "fork"
    with pytest.raises(ValueError, match="exceeds configured total_steps"):
        KimodoTrainer(config, project_root)
