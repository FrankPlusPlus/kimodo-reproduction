from __future__ import annotations

import argparse
import csv
import json
import shutil

import numpy as np
import pytest
import torch

from kimodo.data_pipeline.manifest_cli import build_manifest
from kimodo.data_pipeline.stats_cli import _covering_windows, compute_stats
from kimodo.exports.motion_io import resample_motion_dict_to_kimodo_fps
from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton
from kimodo.training.config import CurriculumConfig
from kimodo.training.constraints import ConstraintCurriculumSampler
from kimodo.training.data import MotionManifestDataset


def _identity_motion(frames: int, joints: int = 30) -> dict[str, torch.Tensor]:
    rotations = torch.eye(3).expand(frames, joints, 3, 3).clone()
    roots = torch.zeros(frames, 3)
    roots[:, 0] = torch.arange(frames, dtype=torch.float32)
    roots[:, 1] = 1.0
    return {"local_rot_mats": rotations, "root_positions": roots}


def test_resampling_uses_target_time_grid_and_exact_output_count():
    skeleton = build_skeleton(30)

    integer, changed = resample_motion_dict_to_kimodo_fps(
        _identity_motion(10), skeleton, source_fps=120.0, target_fps=30.0
    )
    assert changed
    assert len(integer["root_positions"]) == 3
    assert torch.equal(integer["root_positions"][:, 0], torch.tensor([0.0, 4.0, 8.0]))

    fractional, _ = resample_motion_dict_to_kimodo_fps(
        _identity_motion(6), skeleton, source_fps=60.0, target_fps=24.0
    )
    assert len(fractional["root_positions"]) == 3
    assert torch.allclose(
        fractional["root_positions"][:, 0], torch.tensor([0.0, 2.5, 5.0]), atol=1e-5
    )

    with pytest.raises(ValueError, match="target_fps must be positive"):
        resample_motion_dict_to_kimodo_fps(
            _identity_motion(2), skeleton, source_fps=30.0, target_fps=0.0
        )


def test_training_preprocessing_enforces_paper_cap_origin_and_heading(training_fixture):
    rep = KimodoMotionRep(build_skeleton(30), fps=30, stats_path=None)
    with pytest.raises(ValueError, match="fewer than min_frames"):
        MotionManifestDataset(
            training_fixture["manifest"],
            "train",
            rep,
            max_seconds=0.01,
            min_frames=2,
            require_cached_text=False,
        )
    dataset = MotionManifestDataset(
        training_fixture["manifest"],
        "train",
        rep,
        max_seconds=0.1,
        min_frames=2,
        require_cached_text=False,
        normalize=False,
        augment=True,
        seed=17,
    )
    sample = dataset[0]
    assert sample["length"] == 3
    root = sample["clean_motion"][:, rep.slice_dict["smooth_root_pos"]]
    assert torch.allclose(root[0, [0, 2]], torch.zeros(2), atol=1e-6)
    heading = rep.get_root_heading_angle(sample["clean_motion"].unsqueeze(0))[0, 0]
    assert torch.allclose(
        torch.stack([heading.cos(), heading.sin()]),
        torch.stack([sample["first_heading_angle"].cos(), sample["first_heading_angle"].sin()]),
        atol=1e-5,
    )


def test_stats_reuses_training_cap_and_records_unknown_official_policy(training_fixture, tmp_path):
    assert _covering_windows(601, 300) == [(0, 300), (300, 599), (599, 601)]
    output = tmp_path / "stats"
    args = argparse.Namespace(
        manifest=str(training_fixture["manifest"]),
        output=str(output),
        split="train",
        skeleton_joints=30,
        fps=30,
        seed=1234,
        max_seconds=0.1,
    )
    compute_stats(args)
    metadata = json.loads((output / "stats.metadata.json").read_text(encoding="utf-8"))
    # This fixture has one distinct manifest span.  That span is partitioned as
    # [3, 3, 2], instead of fitting statistics to a single random 3-frame crop.
    # Real full/event/combined spans can overlap and intentionally add weight.
    assert metadata["frame_counts"] == {"global_root": 8, "local_root": 8, "body": 8}
    assert metadata["preprocessing"]["maximum_seconds"] == 0.1
    assert metadata["preprocessing"]["stats_window_count"] == 3
    assert metadata["preprocessing"]["stats_window_frames_min"] == 2
    assert metadata["preprocessing"]["stats_window_frames_max"] == 3
    assert (
        metadata["paper_disclosure"]["normalization_statistics_fitting_procedure"]
        == "not_disclosed_reconstruction_assumption"
    )

    # Stable per-(motion/span,window) seeds make a repeated stats build exactly
    # reproducible, independent of process RNG state.
    second = tmp_path / "stats-second"
    args.output = str(second)
    compute_stats(args)
    for component in ("global_root", "local_root", "body"):
        for filename in ("mean.npy", "std.npy"):
            assert np.array_equal(
                np.load(output / component / filename), np.load(second / component / filename)
            )


def test_constraint_curriculum_matches_disclosed_phase_and_mix_rates():
    rep = KimodoMotionRep(build_skeleton(30), fps=30, stats_path=None)
    config = CurriculumConfig(
        phase1_steps=500_000,
        phase2_steps=500_000,
        no_constraint_probability=0.10,
        mix_two_probability=0.25,
        sparse_keyframes_min=1,
        sparse_keyframes_max=20,
    )
    sampler = ConstraintCurriculumSampler(rep, config)
    assert sampler.maximum_sparse_keyframes(500_000) == 1
    assert sampler.maximum_sparse_keyframes(999_999) == 20

    batch_size = 4_000
    clean = torch.zeros(batch_size, 2, rep.motion_rep_dim)
    lengths = torch.full((batch_size,), 2, dtype=torch.long)
    phase1 = sampler.sample(clean[:16], lengths[:16], 499_999, torch.Generator().manual_seed(1))
    assert not phase1.motion_mask.any()
    assert all(not names for names in phase1.pattern_names)

    phase2 = sampler.sample(clean, lengths, 500_000, torch.Generator().manual_seed(123))
    counts = torch.bincount(
        torch.tensor([len(names) for names in phase2.pattern_names]), minlength=3
    ).float()
    fractions = counts / batch_size
    assert fractions[0].item() == pytest.approx(0.10, abs=0.025)
    assert fractions[2].item() == pytest.approx(0.25, abs=0.03)


def test_manifest_sidecar_does_not_claim_unavailable_paper_augmentations(
    training_fixture, tmp_path
):
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
    manifest = tmp_path / "manifest.jsonl"
    build_manifest(
        argparse.Namespace(
            metadata=str(metadata),
            temporal_labels=None,
            split_file=str(split),
            dataset_root=str(tmp_path),
            skeleton="soma_uniform",
            output=str(manifest),
            split_name="train",
            source_fps=30.0,
            full_repeats=1,
            event_repeats=0,
            combined_event_repeats=0,
            allow_missing=False,
        )
    )
    row = json.loads(manifest.read_text(encoding="utf-8"))
    assert row["text_source"] == "bones_seed_metadata:content_natural_desc_1"
    assert row["augmentation_provenance"] == "dataset_annotation"
    sidecar = json.loads(
        manifest.with_suffix(".jsonl.metadata.json").read_text(encoding="utf-8")
    )
    recipe = sidecar["paper_data_recipe"]
    assert recipe["qwen3_32b_paraphrases"] == "not_generated_external_asset_required"
    assert recipe["random_cross_motion_stitching"] == "not_generated_external_asset_required"
    assert recipe["official_mixture_distribution"].startswith("not_disclosed")
    gate = sidecar["paper_parity_gate"]
    assert gate["eligible"] is False
    assert "qwen3_32b_paraphrases" in gate["blockers"]
    rep = KimodoMotionRep(build_skeleton(30), fps=30, stats_path=None)
    with pytest.raises(RuntimeError, match="not eligible for paper-data parity"):
        MotionManifestDataset(
            manifest,
            "train",
            rep,
            require_cached_text=False,
            normalize=False,
            require_paper_data_parity=True,
        )


def test_manifest_uses_canonical_30fps_for_offline_motion_cache(
    training_fixture, tmp_path
):
    metadata = tmp_path / "cache-metadata.csv"
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
                "move_soma_uniform_path": "soma_uniform/bvh/230101/motion.bvh",
                "content_natural_desc_1": "A person walks.",
            }
        )
    split = tmp_path / "cache-split.txt"
    split.write_text("230101/motion\n", encoding="utf-8")
    cache = tmp_path / "cache" / "230101"
    cache.mkdir(parents=True)
    shutil.copy2(training_fixture["motion"], cache / "motion.npz")

    def args(fps):
        return argparse.Namespace(
            metadata=str(metadata),
            temporal_labels=None,
            split_file=str(split),
            dataset_root=str(tmp_path),
            motion_cache_root=str(cache.parent),
            motion_cache_fps=fps,
            skeleton="soma_uniform",
            output=str(tmp_path / f"cached-{fps}.jsonl"),
            split_name="train",
            source_fps=120.0,
            full_repeats=1,
            event_repeats=0,
            combined_event_repeats=0,
            allow_missing=False,
        )

    build_manifest(args(30.0))
    manifest = tmp_path / "cached-30.0.jsonl"
    assert json.loads(manifest.read_text(encoding="utf-8"))["source_fps"] == 30.0
    dataset = MotionManifestDataset(
        manifest,
        "train",
        KimodoMotionRep(build_skeleton(30), fps=30, stats_path=None),
        min_frames=2,
        require_cached_text=False,
        normalize=False,
        augment=False,
    )
    assert dataset[0]["length"] == 8
    with pytest.raises(ValueError, match="fixed to 30 fps"):
        build_manifest(args(120.0))
