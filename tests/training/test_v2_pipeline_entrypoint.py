from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = PROJECT_ROOT / "scripts/v2_pipeline.sh"


def test_v2_pipeline_plan_exposes_the_manual_review_gate():
    result = subprocess.run(
        [str(PIPELINE), "plan"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    lines = result.stdout.splitlines()
    assert lines[0].startswith("prepare")
    assert any(line.startswith("REVIEW-GATE") for line in lines)
    assert lines[-1].startswith("verify")


def test_v2_pipeline_does_not_automate_the_semantic_review_gate():
    result = subprocess.run(
        [str(PIPELINE), "review-gate"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 3
    assert "intentionally not automatic" in result.stderr


def test_v2_pipeline_prepare_builds_and_reuses_plan_and_requests(tmp_path):
    source = tmp_path / "v1" / "train.raw.jsonl"
    source.parent.mkdir(parents=True)
    rows = []
    for index, text in enumerate(("walks left", "waves the right hand")):
        rows.append(
            {
                "id": f"walk:event:{index}:0",
                "motion": "motions/240101/walk.npz",
                "text": text,
                "split": "train",
                "source_fps": 30.0,
                "frame_count": 300,
                "sample_kind": "event",
                "start_time": float(index),
                "end_time": float(index) + 0.8,
            }
        )
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    split = tmp_path / "train_split_paths.txt"
    split.write_text("240101/walk\n", encoding="utf-8")
    root = tmp_path / "v2.building"
    requests = root / "provenance" / "llm.requests.v2.2.jsonl"
    environment = dict(os.environ)
    environment.update(
        {
            "KIMODO_PYTHON": sys.executable,
            "KIMODO_V1_RAW_MANIFEST": str(source),
            "KIMODO_V2_ROOT": str(root),
            "KIMODO_TRAIN_SPLIT": str(split),
            "KIMODO_LLM_REQUESTS": str(requests),
        }
    )

    first = subprocess.run(
        [str(PIPELINE), "prepare"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert first.returncode == 0
    plan = root / "provenance" / "timeline.selected.v2.2.jsonl"
    for path in (plan, requests):
        assert path.is_file()
        assert path.with_suffix(path.suffix + ".metadata.json").is_file()

    second = subprocess.run(
        [str(PIPELINE), "prepare"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "already exist" in second.stdout
