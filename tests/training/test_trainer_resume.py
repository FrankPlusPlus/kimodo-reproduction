from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kimodo.training.config import TrainingConfig
from kimodo.training.constraints import ConstraintCurriculumSampler
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


def test_phase2_benchmark_lane_trains_and_logs_static_pattern_schema(
    training_fixture, tmp_path
):
    project_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "benchmark-lane"
    config = _config(training_fixture, output, 1, accumulation=2)
    config.curriculum.phase1_steps = 0
    config.curriculum.phase2_steps = 1
    config.curriculum.no_constraint_probability = 0.0
    config.curriculum.mix_two_probability = 0.0
    config.curriculum.benchmark_coverage_probability = 1.0
    config.validate()

    KimodoTrainer(config, project_root).train()

    records = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["phase"] == "phase2"
    assert record["benchmark_lane_fraction"] == 1.0
    assert record["paper_two_pattern_fraction"] == 0.0
    assert record["two_pattern_fraction"] == 0.0
    assert record["system/world_size"] == 1
    assert record["system/per_rank_batch"] == 2
    assert record["system/gradient_accumulation_steps"] == 2
    assert record["system/effective_global_batch"] == 4
    assert record["system/optimizer_steps_per_second"] > 0
    assert record["system/samples_per_second"] > 0
    assert record["optimizer/learning_rate"] == config.optimizer.learning_rate
    assert record["maximum_sparse_keyframes"] == 1
    assert record["scheduled_sparse_keyframes"] == pytest.approx(1.0)
    assert record["sampled_sparse_keyframe_cap_mean"] == pytest.approx(1.0)
    assert record["optimizer/gradient_norm_before_clip"] >= 0
    assert 0 <= record["optimizer/gradient_clip_fraction"] <= 1
    assert record["optimizer/skipped_step_fraction"] == 0
    assert record["ema/num_updates"] == 1
    for name in (
        "root_position",
        "root_heading",
        "joint_position",
        "joint_velocity",
        "joint_rotation",
        "foot_contact",
        "forward_kinematics",
    ):
        assert record[f"loss_weighted/{name}"] == pytest.approx(
            record[f"loss/{name}"] * getattr(config.loss, name)
        )
    assert (
        record["exact_two_component_fraction"]
        == record["benchmark_two_component_fraction"]
    )
    assert (
        record["benchmark_atomic_within_benchmark"]
        + record["benchmark_two_component_within_benchmark"]
        + record["benchmark_three_component_within_benchmark"]
        == 1.0
    )
    assert (
        record["benchmark_with_text_within_benchmark"]
        + record["benchmark_without_text_within_benchmark"]
        == 1.0
    )
    assert (
        record["benchmark_duration_lt_3s_within_benchmark"]
        + record["benchmark_duration_3_to_10s_within_benchmark"]
        + record["benchmark_duration_gt_10s_within_benchmark"]
        == 1.0
    )
    benchmark_keys = [
        f"conditioning/{name}_per_sample"
        for name in ConstraintCurriculumSampler.BENCHMARK_PATTERNS
    ]
    assert all(key in record for key in benchmark_keys)
    assert sum(record[key] for key in benchmark_keys) == 1.0


def test_nonfinal_milestone_exports_ema_bundle(training_fixture, tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "milestone-export"
    config = _config(training_fixture, output, 2)
    config.runtime.milestone_every = 1

    KimodoTrainer(config, project_root).train()

    assert (output / "exports/step-000000001/model.pt").is_file()
    assert (output / "exports/step-000000001/config.yaml").is_file()
    assert (output / "exports/step-000000002/model.pt").is_file()


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


def test_fork_resume_applies_yaml_learning_rate_after_loading_optimizer(
    training_fixture, tmp_path
):
    project_root = Path(__file__).resolve().parents[2]
    parent = tmp_path / "parent-lr"
    KimodoTrainer(_config(training_fixture, parent, 1), project_root).train()
    checkpoint = parent / "checkpoints/step-000000001.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["config"]["optimizer"]["learning_rate"] == pytest.approx(2.0e-5)
    state["config"]["optimizer"]["learning_rate"] = 1.0e-5
    torch.save(state, checkpoint)

    child = tmp_path / "child-lr"
    config = _config(training_fixture, child, 2)
    config.optimizer.learning_rate = 1.0e-5
    config.runtime.resume = str(checkpoint)
    config.runtime.resume_mode = "fork"
    trainer = KimodoTrainer(config, project_root)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-5)
    trainer.train()
    records = [
        json.loads(line)
        for line in (child / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["optimizer/learning_rate"] == pytest.approx(1.0e-5)


def test_fork_resume_allows_overlay_fields_without_rewriting_checkpoint(
    training_fixture, tmp_path
):
    project_root = Path(__file__).resolve().parents[2]
    parent = tmp_path / "parent-overlay"
    KimodoTrainer(_config(training_fixture, parent, 1), project_root).train()
    checkpoint = parent / "checkpoints" / "step-000000001.pt"

    child = tmp_path / "child-overlay"
    config = _config(training_fixture, child, 2)
    config.optimizer.learning_rate = 1.0e-5
    config.optimizer.skip_gradient_norm = 5.0
    config.curriculum.sparse_keyframe_cap_mode = "adjacent_mix"
    config.runtime.resume = str(checkpoint)
    config.runtime.resume_mode = "fork"
    trainer = KimodoTrainer(config, project_root)
    assert trainer.global_step == 1
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-5)


def test_fork_reset_optimizer_drops_moments_and_keeps_ema(training_fixture, tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    parent = tmp_path / "parent-reset"
    KimodoTrainer(_config(training_fixture, parent, 1), project_root).train()
    checkpoint = parent / "checkpoints" / "step-000000001.pt"
    parent_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert parent_state["optimizer"]["state"]
    assert parent_state["ema"] is not None

    child = tmp_path / "child-reset"
    config = _config(training_fixture, child, 2)
    config.optimizer.weight_decay = 0.3
    config.optimizer.learning_rate = 1.0e-5
    config.optimizer.warmup_steps = 2
    config.optimizer.warmup_start_lr = 1.0e-6
    config.optimizer.lr_schedule_start_step = 1
    config.runtime.resume = str(checkpoint)
    config.runtime.resume_mode = "fork"
    config.runtime.reset_optimizer = True
    trainer = KimodoTrainer(config, project_root)
    assert trainer.optimizer.state == {}
    assert trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.3)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-6)
    assert trainer.ema is not None
    for name, value in parent_state["ema"]["shadow"].items():
        assert torch.equal(trainer.ema.shadow[name].cpu(), value.cpu())


def test_fork_last_layer_weight_decay_survives_scheduled_hyperparams(
    training_fixture, tmp_path
):
    project_root = Path(__file__).resolve().parents[2]
    parent = tmp_path / "parent-lastwd"
    parent_config = _config(training_fixture, parent, 1)
    parent_config.model.num_layers = 2
    KimodoTrainer(parent_config, project_root).train()
    checkpoint = parent / "checkpoints" / "step-000000001.pt"

    child = tmp_path / "child-lastwd"
    config = _config(training_fixture, child, 2)
    config.model.num_layers = 2
    config.optimizer.weight_decay = 0.3
    config.optimizer.last_layer_weight_decay = 1.0
    config.optimizer.learning_rate = 1.0e-5
    config.optimizer.lr_schedule_start_step = 1
    config.runtime.resume = str(checkpoint)
    config.runtime.resume_mode = "fork"
    config.runtime.reset_optimizer = True
    trainer = KimodoTrainer(config, project_root)
    by_name = {group["name"]: group for group in trainer.optimizer.param_groups}
    assert by_name["rest"]["weight_decay"] == pytest.approx(0.3)
    assert by_name["last_layer"]["weight_decay"] == pytest.approx(1.0)
    trainer._apply_scheduled_optimizer_hyperparams()
    assert by_name["rest"]["weight_decay"] == pytest.approx(0.3)
    assert by_name["last_layer"]["weight_decay"] == pytest.approx(1.0)


def test_scheduled_learning_rate_warms_then_decays():
    from kimodo.training.optim import scheduled_learning_rate

    assert scheduled_learning_rate(
        650_000,
        peak_lr=1e-5,
        total_steps=1_000_000,
        warmup_steps=2_000,
        warmup_start_lr=1e-6,
        lr_end=2e-6,
        schedule_start_step=650_000,
    ) == pytest.approx(1e-6)
    assert scheduled_learning_rate(
        652_000,
        peak_lr=1e-5,
        total_steps=1_000_000,
        warmup_steps=2_000,
        warmup_start_lr=1e-6,
        lr_end=2e-6,
        schedule_start_step=650_000,
    ) == pytest.approx(1e-5)
    assert scheduled_learning_rate(
        1_000_000,
        peak_lr=1e-5,
        total_steps=1_000_000,
        warmup_steps=2_000,
        warmup_start_lr=1e-6,
        lr_end=2e-6,
        schedule_start_step=650_000,
    ) == pytest.approx(2e-6)

