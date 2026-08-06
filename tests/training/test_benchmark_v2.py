from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton
from kimodo.training.config import CurriculumConfig
from kimodo.training.constraints import ConstraintCurriculumSampler
from kimodo.training.data import load_manifest
from kimodo.training.qwen_augmentation_cli import (
    JUDGE_PROMPT,
    _parse_judge,
    _source_preserving_fallback,
)
from kimodo.training.timeline_multi_cli import SYSTEM_PROMPT, prepare, validate_description
from kimodo.training.v2_cached_manifest_cli import compose
from kimodo.training.v2_manifest_cli import build


def _write_v1_fixture(tmp_path):
    motion = tmp_path / "motions" / "240101" / "walk.npz"
    motion.parent.mkdir(parents=True)
    motion.write_bytes(b"fixture")
    manifest = tmp_path / "train.raw.jsonl"
    rows = [
        {
            "id": "walk:full:0:0", "motion": "motions/240101/walk.npz",
            "text": "A person moves.", "split": "train", "source_fps": 30.0,
            "frame_count": 300, "sample_kind": "full",
        }
    ]
    for index, text in enumerate(("walks left", "waves right", "steps forward", "turns backward", "stops still")):
        rows.append(
            {
                "id": f"walk:event:{index}:0", "motion": "motions/240101/walk.npz",
                "text": text, "split": "train", "source_fps": 30.0,
                "frame_count": 300, "sample_kind": "event",
                "start_time": index, "end_time": index + 0.8,
            }
        )
    rows.append(
        {
            "id": "walk:combined2:0:0", "motion": "motions/240101/walk.npz",
            "text": "walks left Then, waves right", "split": "train", "source_fps": 30.0,
            "frame_count": 300, "sample_kind": "combined_events",
            "start_time": 0.0, "end_time": 1.8,
        }
    )
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    split = tmp_path / "train_split_paths.txt"
    split.write_text("240101/walk\n", encoding="utf-8")
    return manifest, split


def test_v2_plan_is_train_only_frame_bounded_and_deterministic(tmp_path):
    manifest, split = _write_v1_fixture(tmp_path)
    plan = tmp_path / "selected.jsonl"
    requests = tmp_path / "requests.jsonl"
    metadata = prepare(
        argparse.Namespace(
            source_manifest=str(manifest), train_split=str(split),
            output_plan=str(plan), output_requests=str(requests), fps=30,
            max_seconds=10.0, max_gap_seconds=1.5,
        )
    )
    assert metadata["counts"]["candidates"] == {"2": 4, "3": 3, "4": 2, "5": 1}
    rows = [json.loads(line) for line in plan.read_text(encoding="utf-8").splitlines()]
    assert all(row["motion_key"] == "240101/walk" for row in rows)
    assert all(row["end_frame"] - row["start_frame"] <= 300 for row in rows)
    assert len({row["id"] for row in rows}) == len(rows)


def test_v2_manifest_replaces_mechanical_combined_and_keeps_gate_honest(tmp_path):
    manifest, split = _write_v1_fixture(tmp_path)
    plan = tmp_path / "selected.jsonl"
    requests = tmp_path / "requests.jsonl"
    prepare(
        argparse.Namespace(
            source_manifest=str(manifest), train_split=str(split),
            output_plan=str(plan), output_requests=str(requests), fps=30,
            max_seconds=10.0, max_gap_seconds=1.5,
        )
    )
    response = tmp_path / "responses.jsonl"
    with requests.open(encoding="utf-8") as source, response.open("w", encoding="utf-8") as output:
        for line in source:
            request = json.loads(line)
            description = "The person " + ", and then ".join(request["source_texts"]) + "."
            output.write(json.dumps({
                "request_id": request["request_id"], "description": description,
                "error": None, "model": "Qwen/Qwen3-32B", "revision": "pinned",
                "semantic_judge": {"accepted": True, "reason": "all actions retained"},
            }) + "\n")
    response_sha = hashlib.sha256(response.read_bytes()).hexdigest()
    requests_sha = hashlib.sha256(requests.read_bytes()).hexdigest()
    response.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3-32B",
                "revision": "pinned",
                "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest(),
                "requests": {"sha256": requests_sha},
                "local_model_snapshot": {"aggregate_sha256": "fixture-model"},
                "output": {"sha256": response_sha},
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "v2" / "train.raw.jsonl"
    bundled_motion = destination.parent / "motions" / "240101" / "walk.npz"
    bundled_motion.parent.mkdir(parents=True)
    os.link(tmp_path / "motions" / "240101" / "walk.npz", bundled_motion)
    metadata = build(
        argparse.Namespace(
            source_manifest=str(manifest), plan=str(plan), responses=[str(response)],
            train_split=str(split), output=str(destination),
            expected_model="Qwen/Qwen3-32B", expected_revision="pinned",
        )
    )
    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert not any(row["sample_kind"] == "combined_events" for row in rows)
    assert any(row["sample_kind"] == "timeline_multi_qwen" for row in rows)
    assert metadata["paper_parity_gate"]["eligible"] is False
    assert metadata["leakage_gate"]["out_of_train_rows"] == 0
    assert all(
        row.get("text_generator_prompt_sha256")
        == hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
        for row in rows if row["sample_kind"] == "timeline_multi_qwen"
    )
    assert destination.stat().st_mode & 0o004


def test_qwen_semantic_judge_schema_is_fail_closed():
    assert _parse_judge('{"accepted": true, "reason": "complete"}')["accepted"]
    try:
        _parse_judge('{"accepted": true, "reason": "complete", "score": 1}')
    except ValueError:
        pass
    else:
        raise AssertionError("judge accepted an unrecognized schema")
    sources = ["The person steps left.", "The person moves forward with the right hand raised."]
    fallback = _source_preserving_fallback(sources)
    validate_description(sources, fallback)
    assert "left" in fallback and "forward" in fallback and "right" in fallback


def test_benchmark_constraint_patterns_have_exact_endpoint_and_path_shapes():
    rep = KimodoMotionRep(build_skeleton(30), fps=30, stats_path=None)
    sampler = ConstraintCurriculumSampler(
        rep,
        CurriculumConfig(
            phase1_steps=1, phase2_steps=2, benchmark_coverage_probability=1.0,
            sparse_keyframes_max=20, benchmark_sparse_keyframes_max=9,
        ),
    )
    generator = torch.Generator().manual_seed(4)
    endpoint = torch.zeros(12, rep.motion_rep_dim, dtype=torch.bool)
    sampler._benchmark_full_body_inbetweening(endpoint, 12, 9, generator)
    assert endpoint.any(dim=1).nonzero().flatten().tolist() == [0, 11]

    position = torch.zeros_like(endpoint)
    sampler._benchmark_root_path_2dpos(position, 12, 9, generator)
    root = rep.slice_dict["smooth_root_pos"]
    heading = rep.slice_dict["global_root_heading"]
    assert position[:, root.start].all() and position[:, root.start + 2].all()
    assert not position[:, heading].any()

    position_heading = torch.zeros_like(endpoint)
    sampler._benchmark_root_path_2dposrot(position_heading, 12, 9, generator)
    assert position_heading[:, heading].all()

    mixed = torch.zeros_like(endpoint)
    sampler._benchmark_mix_root_ee_hands_feet_posrot_fullbody(mixed, 12, 9, generator)
    # This three-way leaf uses a full root path, irrespective of sparse EE/body frames.
    assert mixed[:, root.start].all() and mixed[:, root.start + 2].all()


def test_paper_strict_rejects_the_engineering_benchmark_lane():
    from kimodo.training.config import TrainingConfig

    config = TrainingConfig(paper_method_strict=True)
    config.data.require_paper_data_parity = True
    config.curriculum.benchmark_coverage_probability = 0.25
    try:
        config.validate(require_paths=False)
    except ValueError as error:
        assert "benchmark_coverage_probability" in str(error)
    else:
        raise AssertionError("paper_method_strict accepted the V2 engineering constraint lane")


def test_v2_cached_composer_reuses_base_and_qwen_producers(tmp_path):
    v1_raw, _ = _write_v1_fixture(tmp_path / "source")
    v1_rows = [json.loads(line) for line in v1_raw.read_text(encoding="utf-8").splitlines()]
    bundle = tmp_path / "bundle"
    bundled_motion = bundle / "motions" / "240101" / "walk.npz"
    bundled_motion.parent.mkdir(parents=True)
    os.link(tmp_path / "source" / "motions" / "240101" / "walk.npz", bundled_motion)

    def embedding(cache_dir, name, key):
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / name
        np.save(path, np.zeros((1, 16), dtype=np.float32), allow_pickle=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = path.with_suffix(".npy.metadata.json")
        metadata.write_text(
            json.dumps(
                {"schema_version": 1, "cache_key": key, "sha256": digest,
                 "dtype": "float32", "shape": [1, 16]}
            ),
            encoding="utf-8",
        )
        return digest

    base_cache = bundle / "text-cache-v1"
    qwen_cache = bundle / "text-cache-v2-qwen"
    base_sha = embedding(base_cache, "base.npy", "base-key")
    qwen_sha = embedding(qwen_cache, "qwen.npy", "qwen-key")

    base_cached = tmp_path / "source" / "train.cached.jsonl"
    base_output = []
    for row in (v1_rows[0], v1_rows[1], v1_rows[-1]):
        cached = dict(row)
        cached.update(
            text_embedding="old-cache/base.npy",
            text_embedding_metadata="old-cache/base.npy.metadata.json",
            text_embedding_sha256=base_sha,
            text_cache_key="base-key",
        )
        base_output.append(cached)
    base_cached.write_text("".join(json.dumps(row) + "\n" for row in base_output), encoding="utf-8")

    qwen_row = dict(v1_rows[2])
    qwen_row.update(
        id="v2multi:fixture", sample_kind="timeline_multi_qwen",
        text="The person first walks left and then waves with the right hand.",
        text_embedding="text-cache-v2-qwen/qwen.npy",
        text_embedding_metadata="text-cache-v2-qwen/qwen.npy.metadata.json",
        text_embedding_sha256=qwen_sha, text_cache_key="qwen-key",
    )
    qwen_cached = bundle / "train.qwen.cached.jsonl"
    qwen_cached.write_text(json.dumps(qwen_row) + "\n", encoding="utf-8")
    gate = {"eligible": False, "blockers": ["transitions"]}

    v2_raw = bundle / "train.raw.jsonl"
    raw_rows = [v1_rows[0], v1_rows[1], {k: v for k, v in qwen_row.items() if not k.startswith("text_") or k == "text"}]
    v2_raw.write_text("".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8")
    v2_raw.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps(
            {
                "sources": {"v1_raw_manifest": {"sha256": hashlib.sha256(v1_raw.read_bytes()).hexdigest()}},
                "paper_parity_gate": gate,
                "output": {"sha256": hashlib.sha256(v2_raw.read_bytes()).hexdigest(), "entries": 3},
            }
        ),
        encoding="utf-8",
    )
    base_cached.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps(
            {"encoder": "v1", "source_manifest_sha256": hashlib.sha256(v1_raw.read_bytes()).hexdigest(),
             "output": {"sha256": hashlib.sha256(base_cached.read_bytes()).hexdigest()}}
        ),
        encoding="utf-8",
    )
    qwen_cached.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps(
            {"encoder": "v2", "paper_parity_gate": gate,
             "output": {"sha256": hashlib.sha256(qwen_cached.read_bytes()).hexdigest()}}
        ),
        encoding="utf-8",
    )

    destination = bundle / "train.cached.jsonl"
    metadata = compose(
        argparse.Namespace(
            v2_raw_manifest=str(v2_raw), v1_cached_manifest=str(base_cached),
            qwen_cached_manifest=str(qwen_cached), output=str(destination),
            base_cache_dir="text-cache-v1", qwen_cache_dir="text-cache-v2-qwen",
        )
    )
    assert metadata["encoder_identities"] == {"v1_base": "v1", "v2_qwen_multi": "v2"}
    assert len(load_manifest(destination, "train")) == 3
