from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton
from kimodo.training.data import MotionManifestDataset
from kimodo.training.feature_cache import (
    build_cache_fingerprint,
    materialize_entry_features,
)
from kimodo.training.feature_cache_cli import build_feature_cache, build_parser


def _rep(fixture, *, with_stats: bool = True):
    stats = str(fixture["stats"]) if with_stats else None
    return KimodoMotionRep(build_skeleton(30), fps=30, stats_path=stats)


def _build_cache(fixture, output: Path, *, overwrite: bool = True):
    args = build_parser().parse_args(
        [
            "--manifest",
            str(fixture["manifest"]),
            "--output",
            str(output),
            "--stats-path",
            str(fixture["stats"]),
            "--num-workers",
            "1",
            "--verify-sample",
            "2",
            "--min-frames",
            "2",
        ]
        + (["--overwrite"] if overwrite else [])
    )
    return build_feature_cache(args)


def test_feature_cache_cli_and_short_clip_parity(training_fixture, tmp_path):
    cache_dir = tmp_path / "feature-cache"
    _build_cache(training_fixture, cache_dir)

    rep = _rep(training_fixture)
    live = MotionManifestDataset(
        training_fixture["manifest"],
        "train",
        rep,
        max_seconds=10.0,
        min_frames=2,
        seed=11,
        augment=False,
        normalize=False,
    )
    cached = MotionManifestDataset(
        training_fixture["manifest"],
        "train",
        rep,
        max_seconds=10.0,
        min_frames=2,
        seed=11,
        augment=False,
        normalize=False,
        feature_cache_dir=cache_dir,
        stats_path=training_fixture["stats"],
    )
    for index in range(len(live)):
        live_feat = live[index]["clean_motion"].numpy()
        cache_feat = cached[index]["clean_motion"].numpy()
        live_f16 = live_feat.astype(np.float16).astype(np.float32)
        assert cache_feat.shape == live_feat.shape
        assert np.allclose(cache_feat, live_f16, rtol=0.0, atol=2e-3)


def test_feature_cache_rng_order_matches_live_window_then_rotate(tmp_path):
    root = tmp_path / "long"
    root.mkdir()
    stats = root / "stats"
    for name, width in (("global_root", 5), ("local_root", 4), ("body", 364)):
        folder = stats / name
        folder.mkdir(parents=True)
        np.save(folder / "mean.npy", np.zeros(width, dtype=np.float32))
        np.save(folder / "std.npy", np.ones(width, dtype=np.float32))

    frames = 40  # > max_frames=10 at 30fps with max_seconds≈0.333 → use max_seconds=0.5 → 15
    joints = 30
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (frames, joints, 3, 3)).copy()
    roots = np.zeros((frames, 3), dtype=np.float32)
    roots[:, 0] = np.linspace(0.0, 0.8, frames, dtype=np.float32)
    roots[:, 1] = 1.0
    motion = root / "motion.npz"
    np.savez(motion, local_rot_mats=rotations, root_positions=roots)
    embedding = root / "embedding.npy"
    np.save(embedding, np.zeros((1, 16), dtype=np.float32))
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "long-0",
                "motion": motion.name,
                "text": "long",
                "split": "train",
                "source_fps": 30,
                "text_embedding": embedding.name,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fixture = {"root": root, "stats": stats, "manifest": manifest}
    cache_dir = tmp_path / "cache"
    _build_cache(fixture, cache_dir)

    rep = _rep(fixture)
    max_seconds = 0.5  # 15 frames at 30 fps
    seed = 99
    live = MotionManifestDataset(
        manifest, "train", rep, max_seconds=max_seconds, min_frames=2, seed=seed, augment=True
    )
    cached = MotionManifestDataset(
        manifest,
        "train",
        rep,
        max_seconds=max_seconds,
        min_frames=2,
        seed=seed,
        augment=True,
        feature_cache_dir=cache_dir,
        stats_path=stats,
    )

    # Probe RNG stream: window start then heading angle.
    gen_live = live._generator(0)
    gen_cache = cached._generator(0)
    length = 40
    max_frames = round(max_seconds * 30)
    live_start = int(torch.randint(length - max_frames + 1, (), generator=gen_live).item())
    cache_start = int(torch.randint(length - max_frames + 1, (), generator=gen_cache).item())
    assert live_start == cache_start
    live_heading = torch.rand((1,), generator=gen_live) * (2.0 * torch.pi)
    cache_heading = torch.rand((1,), generator=gen_cache) * (2.0 * torch.pi)
    assert torch.allclose(live_heading, cache_heading)

    live_sample = live[0]
    cache_sample = cached[0]
    assert live_sample["length"] == cache_sample["length"] == max_frames
    assert torch.allclose(
        live_sample["first_heading_angle"], cache_sample["first_heading_angle"], atol=1e-6
    )


def test_feature_cache_fail_closed_on_missing_row(training_fixture, tmp_path):
    cache_dir = tmp_path / "feature-cache"
    _build_cache(training_fixture, cache_dir)
    index_path = cache_dir / "index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line]
    # Drop one sample from the index.
    index_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows[:-1]), encoding="utf-8"
    )
    rep = _rep(training_fixture)
    with pytest.raises(FileNotFoundError, match="cache rows are missing"):
        MotionManifestDataset(
            training_fixture["manifest"],
            "train",
            rep,
            feature_cache_dir=cache_dir,
            stats_path=training_fixture["stats"],
        )


def test_feature_cache_dir_is_resume_whitelisted():
    # Avoid importing kimodo.training.checkpoint (pulls model/hydra/safetensors).
    source = Path("kimodo/training/checkpoint.py").read_text(encoding="utf-8")
    assert '"feature_cache_dir"' in source
    assert "def _resume_critical_config" in source


def test_materialize_matches_dataset_without_window(training_fixture):
    rep = _rep(training_fixture, with_stats=False)
    dataset = MotionManifestDataset(
        training_fixture["manifest"],
        "train",
        rep,
        max_seconds=10.0,
        min_frames=2,
        seed=3,
        augment=False,
        normalize=False,
    )
    entry = dataset.entries[0]
    materialised = materialize_entry_features(entry, rep, min_frames=2)
    sample = dataset[0]["clean_motion"]
    assert torch.allclose(materialised, sample, atol=1e-5)


def test_cache_fingerprint_binds_stats(training_fixture, tmp_path):
    other_stats = tmp_path / "other-stats"
    for name, width in (("global_root", 5), ("local_root", 4), ("body", 364)):
        folder = other_stats / name
        folder.mkdir(parents=True)
        np.save(folder / "mean.npy", np.ones(width, dtype=np.float32))
        np.save(folder / "std.npy", np.ones(width, dtype=np.float32))
    left = build_cache_fingerprint(
        fps=30, feature_dim=369, skeleton_joints=30, stats_path=training_fixture["stats"]
    )
    right = build_cache_fingerprint(
        fps=30, feature_dim=369, skeleton_joints=30, stats_path=other_stats
    )
    assert left["stats_sha256"] != right["stats_sha256"]
