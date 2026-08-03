from __future__ import annotations

import copy
import argparse
import csv
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from kimodo.constraints import FootContactConstraintSet
from kimodo.model.diffusion import Diffusion
from kimodo.model.loading import instantiate_from_dict
from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton
from kimodo.training.config import CurriculumConfig, LossConfig, ModelConfig, TrainingConfig
from kimodo.training.constraints import ConstraintCurriculumSampler
from kimodo.training.data import MotionManifestDataset, collate_motion_batch
from kimodo.training.ema import ExponentialMovingAverage
from kimodo.training.losses import KimodoLoss
from kimodo.training.manifest_cli import build_manifest
from kimodo.training.modeling import build_trainable_denoiser, validate_model_contract
from kimodo.training.optim import AdamAtan2
from kimodo.training.provenance import code_fingerprints
from kimodo.training.stats_cli import compute_stats
from kimodo.training.smoke_fixture_cli import create_smoke_fixture


def _motion_rep(training_fixture):
    return KimodoMotionRep(build_skeleton(30), fps=30, stats_path=str(training_fixture["stats"]))


def test_data_representation_and_all_constraint_families(training_fixture):
    rep = _motion_rep(training_fixture)
    dataset = MotionManifestDataset(
        training_fixture["manifest"], "train", rep, max_seconds=1.0, min_frames=2, seed=7
    )
    batch = collate_motion_batch([dataset[0], dataset[1]])
    assert batch["clean_motion"].shape == (2, 8, 369)
    assert batch["valid_frames"].all()
    assert batch["text_features"].shape == (2, 2, 16)

    curriculum = CurriculumConfig(phase1_steps=1, phase2_steps=2, sparse_keyframes_max=3)
    sampler = ConstraintCurriculumSampler(rep, curriculum)
    for pattern in sampler.PATTERNS:
        mask = torch.zeros(8, rep.motion_rep_dim, dtype=torch.bool)
        sampler._apply_pattern(pattern, mask, 8, 3, torch.Generator().manual_seed(4))
        assert mask.any(), pattern

    deterministic = CurriculumConfig(
        phase1_steps=1,
        phase2_steps=2,
        sparse_keyframes_min=2,
        sparse_keyframes_max=2,
        root_heading_probability=1.0,
    )
    sampler = ConstraintCurriculumSampler(rep, deterministic)
    full = torch.zeros(8, rep.motion_rep_dim, dtype=torch.bool)
    sampler._full_body_sparse(full, 8, 2, torch.Generator().manual_seed(1))
    assert full[:, rep.slice_dict["smooth_root_pos"]].sum() == 6
    assert full[:, rep.slice_dict["global_root_heading"]].sum() == 4
    assert full[:, rep.slice_dict["local_joints_positions"]].sum() == 180
    assert not full[:, rep.slice_dict["global_rot_data"]].any()

    root = torch.zeros_like(full)
    sampler._root_sparse(root, 8, 2, torch.Generator().manual_seed(1))
    assert root.sum() == 8  # two frames × (root XZ + heading cos/sin)
    assert not root[:, rep.slice_dict["foot_contacts"]].any()

    contacts = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    constraint = FootContactConstraintSet(rep.skeleton, torch.tensor([1, 5]), contacts)
    observed, mask = rep.create_conditions_from_constraints(
        [constraint], length=8, to_normalize=False, device="cpu"
    )
    foot_slice = rep.slice_dict["foot_contacts"]
    assert torch.equal(observed[[1, 5], foot_slice], contacts)
    assert mask[[1, 5], foot_slice].all()


def test_bones_manifest_full_event_and_combined_crops(training_fixture, tmp_path):
    metadata = tmp_path / "metadata.csv"
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
                "move_soma_uniform_path": str(training_fixture["motion"]),
                "content_natural_desc_1": "A person walks.",
            }
        )
    split = tmp_path / "split.txt"
    split.write_text("230101/motion\n", encoding="utf-8")
    temporal = tmp_path / "timeline.jsonl"
    temporal.write_text(
        json.dumps(
            {
                "filename": "motion",
                "events": [
                    {"start_time": 0.0, "end_time": 0.1, "description": "A person starts."},
                    {"start_time": 0.1, "end_time": 0.2, "description": "A person stops."},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    args = argparse.Namespace(
        metadata=str(metadata),
        temporal_labels=str(temporal),
        split_file=str(split),
        dataset_root=str(tmp_path),
        skeleton="soma_uniform",
        output=str(manifest),
        split_name="train",
        source_fps=30.0,
        full_repeats=1,
        event_repeats=1,
        combined_event_repeats=1,
        allow_missing=False,
    )
    counts = build_manifest(args)
    assert counts == {"motions": 1, "full": 1, "event": 2, "combined": 1, "missing": 0}
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert {row["sample_kind"] for row in rows} == {"full", "event", "combined_events"}
    source_record = json.loads(
        manifest.with_suffix(".jsonl.metadata.json").read_text(encoding="utf-8")
    )
    assert source_record["sources"]["metadata"]["path"] == str(metadata)
    assert source_record["sources"]["split_file"]["sha256"]
    assert source_record["sources"]["temporal_labels"]["sha256"]

    dataset = MotionManifestDataset(
        manifest,
        "train",
        _motion_rep(training_fixture),
        max_seconds=None,
        min_frames=2,
        require_cached_text=False,
        normalize=False,
        augment=False,
    )
    lengths_by_kind = {}
    for index, entry in enumerate(dataset.entries):
        lengths_by_kind.setdefault(entry.sample_kind, []).append(dataset[index]["length"])
    assert lengths_by_kind == {"full": [8], "event": [3, 3], "combined_events": [6]}


def test_diffusion_loss_and_adam_atan2_contract(training_fixture):
    rep = _motion_rep(training_fixture)
    target = torch.zeros(2, 8, rep.motion_rep_dim)
    valid = torch.ones(2, 8, dtype=torch.bool)
    criterion = KimodoLoss(rep, LossConfig())
    zero = criterion(target.clone(), target, valid)
    assert all(torch.equal(value, torch.zeros_like(value)) for value in zero.values())

    padded_prediction = target.clone()
    padded_prediction[:, 6:] = 100.0
    valid[:, 6:] = False
    padded = criterion(padded_prediction, target, valid)
    assert padded["total"].item() == 0.0

    physical = KimodoLoss(rep, LossConfig(direct_feature_domain="physical"))
    changed = target.clone()
    changed[..., 0] = 1.0
    assert physical(changed, target, torch.ones_like(valid))["root_position"].item() > 0

    diffusion = Diffusion(1000)
    schedule, mapping = diffusion.space_timesteps(100)
    assert len(schedule) == len(mapping) == 100
    assert schedule[0] == 0 and schedule[-1] == 999
    noise = torch.ones_like(target)
    timesteps = torch.tensor([0, 999])
    first = diffusion.q_sample(target, timesteps, noise)
    second = diffusion.q_sample(target, timesteps, noise)
    assert torch.equal(first, second)

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = AdamAtan2([parameter], lr=0.1, betas=(0.0, 0.0), atan2_lambda=8.0)
    parameter.grad = torch.tensor([2.0])
    optimizer.step()
    expected_update = (4.0 / torch.pi) * 8.0 * torch.atan2(torch.tensor(2.0), torch.tensor(16.0))
    expected = 1.0 - 0.1 * expected_update
    assert torch.allclose(parameter.detach(), expected[None])


def test_unequal_length_frame_accumulation_matches_global_batch(training_fixture):
    rep = _motion_rep(training_fixture)
    criterion = KimodoLoss(rep, LossConfig())

    separate_parameter = torch.nn.Parameter(torch.tensor(0.2))
    separate_sums = []
    for length in (3, 7):
        prediction = separate_parameter * torch.ones(1, length, rep.motion_rep_dim)
        target = torch.zeros_like(prediction)
        output = criterion(prediction, target, torch.ones(1, length, dtype=torch.bool))
        separate_sums.append(output.frame_sums["total"])
    (sum(separate_sums) / 10.0).backward()

    global_parameter = torch.nn.Parameter(torch.tensor(0.2))
    prediction = global_parameter * torch.ones(2, 7, rep.motion_rep_dim)
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[True, True, True, False, False, False, False], [True] * 7])
    criterion(prediction, target, valid)["total"].backward()
    assert torch.allclose(separate_parameter.grad, global_parameter.grad, rtol=1e-5, atol=1e-6)


def test_root_body_detach_policy_changes_gradient(training_fixture):
    base = ModelConfig(
        skeleton_joints=30,
        stats_path=str(training_fixture["stats"]),
        llm_dim=16,
        num_text_tokens_override=2,
        latent_dim=16,
        ff_size=32,
        num_layers=1,
        num_heads=4,
    )
    curriculum = CurriculumConfig(phase1_steps=1, phase2_steps=1)

    def root_gradient(detach: bool):
        config = copy.deepcopy(base)
        config.detach_root_for_body = detach
        model = build_trainable_denoiser(config, curriculum, torch.device("cpu"))
        model.train()
        motion = torch.randn(1, 8, model.motion_rep.motion_rep_dim)
        body = model(
            motion,
            torch.ones(1, 8, dtype=torch.bool),
            torch.randn(1, 2, 16),
            torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([12]),
            first_heading_angle=torch.zeros(1),
        )[..., model.motion_rep.body_slice]
        body.square().mean().backward()
        gradients = [parameter.grad for parameter in model.root_model.parameters()]
        return gradients

    assert all(gradient is None or gradient.abs().sum() == 0 for gradient in root_gradient(True))
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in root_gradient(False))


def test_exported_bundle_strictly_instantiates(training_fixture, tmp_path):
    from kimodo.training.checkpoint import export_inference_bundle
    from kimodo.training.config import TrainingConfig

    config = TrainingConfig()
    config.model = ModelConfig(
        skeleton_joints=30,
        stats_path=str(training_fixture["stats"]),
        llm_dim=16,
        num_text_tokens_override=2,
        latent_dim=16,
        ff_size=32,
        num_layers=1,
        num_heads=4,
    )
    config.curriculum = CurriculumConfig(phase1_steps=1, phase2_steps=1)
    model = build_trainable_denoiser(config.model, config.curriculum, torch.device("cpu"))
    bundle = export_inference_bundle(model, None, tmp_path, 2, config)
    loaded = OmegaConf.load(bundle / "config.yaml")
    resolved = OmegaConf.to_container(
        OmegaConf.merge(loaded, OmegaConf.create({"checkpoint_dir": str(bundle)})), resolve=True
    )
    resolved.pop("checkpoint_dir", None)
    wrapper = instantiate_from_dict(resolved, overrides={"device": "cpu"})
    reloaded = wrapper.denoiser.model
    for key, value in model.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[key])


def test_ema_roundtrip_and_persistent_worker_guard():
    model = torch.nn.Linear(2, 1)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.add_(2.0)
        model.bias.add_(1.0)
    ema.update(model)
    assert torch.allclose(ema.shadow["weight"], initial["weight"] + 1.0)
    assert torch.allclose(ema.shadow["bias"], initial["bias"] + 0.5)

    restored = ExponentialMovingAverage(model, decay=0.9)
    restored.load_state_dict(ema.state_dict())
    restored.to("cpu")
    clone = torch.nn.Linear(2, 1)
    restored.copy_to(clone)
    assert restored.num_updates == 1 and restored.decay == 0.5
    assert all(
        torch.equal(value, clone.state_dict()[name])
        for name, value in restored.shadow.items()
    )

    config = TrainingConfig()
    config.data.persistent_workers = True
    with pytest.raises(ValueError, match="persistent_workers is disabled"):
        config.validate(require_paths=False)


def test_official_mode_cannot_hide_skeleton_mismatch(training_fixture):
    config = ModelConfig(
        skeleton_joints=30,
        stats_path=str(training_fixture["stats"]),
        llm_dim=16,
        num_text_tokens_override=2,
        latent_dim=16,
        ff_size=32,
        num_layers=1,
        num_heads=4,
    )
    model = build_trainable_denoiser(
        config, CurriculumConfig(phase1_steps=1, phase2_steps=1), torch.device("cpu")
    )
    config.checkpoint_dir = "/an/official/bundle"
    config.skeleton_joints = 34
    with pytest.raises(ValueError, match="skeleton joint count"):
        validate_model_contract(model, config)


def test_stats_metadata_and_training_code_snapshot(training_fixture, tmp_path):
    output = tmp_path / "computed-stats"
    compute_stats(
        argparse.Namespace(
            manifest=str(training_fixture["manifest"]),
            output=str(output),
            split="train",
            skeleton_joints=30,
            fps=30,
            seed=1234,
        )
    )
    metadata = json.loads((output / "stats.metadata.json").read_text(encoding="utf-8"))
    assert metadata["unique_clips"] == 1
    assert metadata["frame_counts"] == {"global_root": 8, "local_root": 8, "body": 8}
    assert metadata["heading_augmentation"] == "deterministic_uniform"

    project_root = Path(__file__).resolve().parents[2]
    snapshot = code_fingerprints(project_root)
    for expected in (
        "kimodo/model/backbone.py",
        "kimodo/motion_rep/reps/kimodo_motionrep.py",
        "kimodo/skeleton/kinematics.py",
        "kimodo/exports/bvh.py",
        "kimodo/exports/motion_formats.py",
        "kimodo/assets.py",
        "configs/training/kimodo_soma_seed_reproduction.yaml",
        "docker_requirements.txt",
    ):
        assert expected in snapshot


def test_tiny_smoke_fixture_is_self_generated(tmp_path):
    fixture = create_smoke_fixture(tmp_path / "smoke")
    assert fixture["manifest"].is_file()
    dataset = MotionManifestDataset(
        fixture["manifest"],
        "train",
        KimodoMotionRep(build_skeleton(30), fps=30, stats_path=str(fixture["stats"])),
        max_seconds=1.0,
        min_frames=2,
    )
    assert len(dataset) == 4
    assert dataset[0]["clean_motion"].shape == (8, 369)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_smoke_fixture(fixture["root"])
