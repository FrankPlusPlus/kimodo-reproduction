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
    assert config.model.detach_root_for_body is True
    assert config.model.llm_tokens == 1
    assert config.data.require_paper_data_parity is True
    assert config.data.reference_verification == "inventory"
    assert config.runtime.enforce_paper_scale is True

    # The paper does not specify cross-bridge autograd, so strict paper values
    # do not reject the explicit gradient-coupled ablation.
    config.model.detach_root_for_body = False
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
    config.model.llm_tokens = 2
    with pytest.raises(ValueError, match="model.llm_tokens"):
        config.validate(require_paths=False)

    config.model.llm_tokens = 1
    config.runtime.max_steps_override = 10
    with pytest.raises(ValueError, match="max_steps_override"):
        config.validate(require_paths=False)

    config.runtime.max_steps_override = None
    config.runtime.initial_global_step = 999_999
    with pytest.raises(ValueError, match="initial_global_step"):
        config.validate(require_paths=False)
    config.runtime.initial_global_step = 0
    with pytest.raises(RuntimeError, match="world_size=1"):
        training_engine.validate_paper_runtime_scale(config, SimpleNamespace(world_size=1))
    training_engine.validate_paper_runtime_scale(config, SimpleNamespace(world_size=16))

    # Hardware scale is an explicit, independently recorded exception. It
    # must not disable any of the method-level strict validation above.
    config.runtime.enforce_paper_scale = False
    training_engine.validate_paper_runtime_scale(config, SimpleNamespace(world_size=2))
    config.model.input_first_heading_angle = False
    with pytest.raises(ValueError, match="input_first_heading_angle"):
        config.validate(require_paths=False)


def test_two_gpu_profile_only_relaxes_runtime_scale():
    config = load_training_config(
        PROJECT_ROOT / "configs/training/kimodo_soma_seed_reproduction.yaml",
        ["runtime.dry_run=true"],
        overlays=[PROJECT_ROOT / "configs/overlays/two_h200_gb2048.yaml"],
    )
    assert config.paper_method_strict is True
    assert config.runtime.enforce_paper_scale is False
    assert config.runtime.batch_size == 128
    assert config.runtime.gradient_accumulation_steps == 8
    assert config.data.require_paper_data_parity is True
    assert config.data.reference_verification == "inventory"
    assert config.model.detach_root_for_body is True
    training_engine.validate_paper_runtime_scale(config, SimpleNamespace(world_size=2))


def test_v2_30k_profile_uses_normalized_direct_loss_and_preserves_other_loss_terms():
    config = load_training_config(
        PROJECT_ROOT / "configs/training/kimodo_soma_seed_v2_30k.yaml",
        ["runtime.dry_run=true"],
        overlays=[PROJECT_ROOT / "configs/overlays/two_node_16_h200_gb2048.yaml"],
    )

    assert config.curriculum.phase1_steps == 20_000
    assert config.curriculum.phase2_steps == 10_000
    assert config.curriculum.benchmark_coverage_probability == 0.25
    assert config.loss.direct_feature_domain == "normalized"
    assert config.loss.smooth_l1_beta == 1.0
    assert {
        "root_position": config.loss.root_position,
        "root_heading": config.loss.root_heading,
        "joint_position": config.loss.joint_position,
        "joint_velocity": config.loss.joint_velocity,
        "joint_rotation": config.loss.joint_rotation,
        "foot_contact": config.loss.foot_contact,
        "forward_kinematics": config.loss.forward_kinematics,
    } == {
        "root_position": 10.0,
        "root_heading": 2.0,
        "joint_position": 10.0,
        "joint_velocity": 3.0,
        "joint_rotation": 10.0,
        "foot_contact": 4.0,
        "forward_kinematics": 5.0,
    }
    assert config.optimizer.name == "adam_atan2"
    assert config.optimizer.learning_rate == 1.0e-5
    assert config.model.detach_root_for_body is True
    assert config.runtime.batch_size * 16 * config.runtime.gradient_accumulation_steps == 2048


def test_company_v2_1m_profile_is_complete_and_does_not_need_a_hardware_overlay():
    config = load_training_config(
        PROJECT_ROOT / "configs/training/kimodo_soma_seed_v2_1m_16h200.yaml",
        ["runtime.dry_run=true"],
    )

    assert config.paper_method_strict is False
    assert config.curriculum.phase1_steps == 500_000
    assert config.curriculum.phase2_steps == 500_000
    assert config.curriculum.benchmark_coverage_probability == 0.25
    assert config.loss.direct_feature_domain == "normalized"
    assert config.loss.forward_kinematics == 5.0
    assert config.optimizer.name == "adam_atan2"
    assert config.optimizer.learning_rate == 2.0e-5
    assert config.runtime.batch_size == 128
    assert config.runtime.gradient_accumulation_steps == 1
    assert config.runtime.batch_size * 16 == 2048
    assert config.runtime.expected_world_size == 16
    assert config.runtime.expected_global_batch == 2048
    assert config.runtime.checkpoint_every == 10_000
    assert config.runtime.milestone_every == 100_000
    assert config.runtime.enforce_paper_scale is True
    assert config.data.persistent_workers is True
    training_engine.validate_paper_runtime_scale(config, SimpleNamespace(world_size=16))
    with pytest.raises(RuntimeError, match="deployment contract"):
        training_engine.validate_paper_runtime_scale(config, SimpleNamespace(world_size=8))


def test_v2_1m_600k_fork_overlay_halves_lr_and_keeps_more_checkpoints():
    config = load_training_config(
        PROJECT_ROOT / "configs/training/kimodo_soma_seed_v2_1m_16h200.yaml",
        ["runtime.dry_run=true"],
        overlays=[PROJECT_ROOT / "configs/overlays/v2_1m_from600k_kf_smooth.yaml"],
    )

    assert config.curriculum.sparse_keyframes_max == 20
    assert config.curriculum.sparse_keyframe_cap_mode == "adjacent_mix"
    assert config.curriculum.benchmark_coverage_probability == 0.25
    assert config.optimizer.learning_rate == 1.0e-5
    assert config.optimizer.gradient_clip_norm == 1.0
    assert config.optimizer.skip_gradient_norm == 2.0
    assert config.runtime.checkpoint_every == 5_000
    assert config.runtime.keep_last_checkpoints == 20
    assert config.runtime.milestone_every == 100_000
    assert config.model.detach_root_for_body is True
    assert config.curriculum.sparse_load_baseline == 7
    assert config.curriculum.sparse_tail_power == 2.0
    assert config.curriculum.sparse_channel_budget == 800
    assert config.curriculum.sparse_same_frame_overlap_probability == 0.0


def test_v2_1m_k7_fork_overlay_freezes_sampled_cap_and_keeps_paper_max():
    config = load_training_config(
        PROJECT_ROOT / "configs/training/kimodo_soma_seed_v2_1m_16h200.yaml",
        ["runtime.dry_run=true"],
        overlays=[PROJECT_ROOT / "configs/overlays/v2_1m_k7_from695k.yaml"],
    )

    assert config.curriculum.sparse_keyframes_max == 20
    assert config.curriculum.sparse_keyframes_hard_cap == 7
    assert config.curriculum.sparse_keyframe_cap_mode == "adjacent_mix"
    assert config.curriculum.sparse_channel_budget == 0
    assert config.optimizer.learning_rate == 1.0e-5
    assert config.optimizer.skip_gradient_norm == 5.0
    assert config.runtime.checkpoint_every == 1_000
    assert config.runtime.keep_last_checkpoints == 20


def test_v2_1m_wd03_from650k_overlay_keeps_k20_and_sets_stability_knobs():
    config = load_training_config(
        PROJECT_ROOT / "configs/training/kimodo_soma_seed_v2_1m_16h200.yaml",
        ["runtime.dry_run=true"],
        overlays=[PROJECT_ROOT / "configs/overlays/v2_1m_wd03_from650k.yaml"],
    )

    assert config.curriculum.sparse_keyframes_max == 20
    assert config.curriculum.sparse_keyframes_hard_cap == 0
    assert config.curriculum.sparse_keyframe_cap_mode == "adjacent_mix"
    assert config.curriculum.sparse_channel_budget == 0
    assert config.optimizer.learning_rate == 1.0e-5
    assert config.optimizer.weight_decay == pytest.approx(0.3)
    assert config.optimizer.warmup_steps == 2000
    assert config.optimizer.warmup_start_lr == pytest.approx(1.0e-6)
    assert config.optimizer.lr_end == pytest.approx(2.0e-6)
    assert config.optimizer.lr_schedule_start_step == 650_000
    assert config.optimizer.skip_gradient_norm is None
    assert config.runtime.reset_optimizer is True
    assert config.runtime.checkpoint_every == 5_000
    assert config.runtime.milestone_every == 50_000
    assert config.runtime.keep_last_checkpoints == 20


def test_v2_1m_lastwd1_from750k_overlay_keeps_parent_lr_and_splits_wd():
    config = load_training_config(
        PROJECT_ROOT / "configs/training/kimodo_soma_seed_v2_1m_16h200.yaml",
        ["runtime.dry_run=true"],
        overlays=[PROJECT_ROOT / "configs/overlays/v2_1m_lastwd1_from750k.yaml"],
    )

    assert config.curriculum.sparse_keyframes_max == 20
    assert config.curriculum.sparse_keyframes_hard_cap == 0
    assert config.optimizer.learning_rate == pytest.approx(1.0e-5)
    assert config.optimizer.weight_decay == pytest.approx(0.3)
    assert config.optimizer.last_layer_weight_decay == pytest.approx(1.0)
    assert config.optimizer.warmup_steps == 2000
    assert config.optimizer.lr_end == pytest.approx(2.0e-6)
    assert config.optimizer.lr_schedule_start_step == 650_000
    assert config.optimizer.skip_gradient_norm is None
    assert config.runtime.reset_optimizer is True


def test_v2_1m_wd03_from780k_lr3e6_overlay_drops_peak_lr_and_keeps_wd():
    config = load_training_config(
        PROJECT_ROOT / "configs/training/kimodo_soma_seed_v2_1m_16h200.yaml",
        ["runtime.dry_run=true"],
        overlays=[PROJECT_ROOT / "configs/overlays/v2_1m_wd03_from780k_lr3e6.yaml"],
    )

    assert config.curriculum.sparse_keyframes_max == 20
    assert config.curriculum.sparse_keyframes_hard_cap == 0
    assert config.curriculum.sparse_keyframe_cap_mode == "adjacent_mix"
    assert config.optimizer.learning_rate == pytest.approx(3.0e-6)
    assert config.optimizer.weight_decay == pytest.approx(0.3)
    assert config.optimizer.warmup_steps == 2000
    assert config.optimizer.lr_end == pytest.approx(1.5e-6)
    assert config.optimizer.lr_schedule_start_step == 780_000
    assert config.optimizer.skip_gradient_norm is None
    assert config.runtime.reset_optimizer is True
    assert config.data.num_workers == 16
    assert config.runtime.log_every == 20


def test_v2_1m_k7_reseed696k_overlay_changes_seed_and_stops_at_698k():
    config = load_training_config(
        PROJECT_ROOT / "configs/training/kimodo_soma_seed_v2_1m_16h200.yaml",
        ["runtime.dry_run=true"],
        overlays=[PROJECT_ROOT / "configs/overlays/v2_1m_k7_reseed696k.yaml"],
    )

    assert config.runtime.seed == 4321
    assert config.runtime.max_steps_override == 698_000
    assert config.total_steps == 698_000
    assert config.curriculum.sparse_keyframes_hard_cap == 7
    assert config.curriculum.sparse_keyframes_max == 20
    assert config.optimizer.learning_rate == 1.0e-5


def test_public_profile_plus_hardware_overlay_differs_only_in_unavailable_data_claims():
    strict = load_training_config(
        PROJECT_ROOT / "configs/training/kimodo_soma_seed_reproduction.yaml",
        ["runtime.dry_run=true"],
        overlays=[PROJECT_ROOT / "configs/overlays/two_h200_gb2048.yaml"],
    )
    public = load_training_config(
        PROJECT_ROOT / "configs/training/kimodo_soma_seed_public.yaml",
        ["runtime.dry_run=true"],
        overlays=[PROJECT_ROOT / "configs/overlays/two_h200_gb2048.yaml"],
    )
    assert public.paper_method_strict is False
    assert public.data.require_paper_data_parity is False
    assert public.runtime.enforce_paper_scale is False
    assert public.runtime.batch_size * 2 * public.runtime.gradient_accumulation_steps == 2048

    def flatten(value, prefix=""):
        result = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                result.update(flatten(item, name))
            else:
                result[name] = item
        return result

    strict_values = flatten(strict.to_dict())
    public_values = flatten(public.to_dict())
    differences = {
        key for key in strict_values if strict_values[key] != public_values[key]
    }
    assert differences == {
        "paper_method_strict",
        "data.require_paper_data_parity",
    }


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
