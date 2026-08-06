from __future__ import annotations

import json
from pathlib import Path

from kimodo.training.eval_monitor_cli import (
    benchmark_inventory_sha256,
    discover_exports,
    resolve_benchmark_inventory_sha256,
    trend_alerts,
)


def _complete_export(root: Path, step: int) -> Path:
    export = root / "exports" / f"step-{step:09d}"
    (export / "stats").mkdir(parents=True)
    (export / "model.pt").write_bytes(b"weights")
    (export / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    return export


def test_discover_exports_accepts_only_atomically_complete_bundles(tmp_path):
    first = _complete_export(tmp_path, 100_000)
    incomplete = tmp_path / "exports" / "step-000200000"
    incomplete.mkdir()
    (incomplete / "model.pt").touch()
    (tmp_path / "exports" / "step-000300000.tmp").mkdir()

    assert discover_exports(tmp_path) == [(100_000, first)]
    assert discover_exports(tmp_path, minimum_step=150_000) == []


def test_benchmark_inventory_hash_binds_metadata_constraints_and_ground_truth(tmp_path):
    sample = tmp_path / "content" / "text2motion" / "overview" / "case" / "0000"
    sample.mkdir(parents=True)
    (sample / "meta.json").write_text(json.dumps({"text": "walk"}), encoding="utf-8")
    (sample / "constraints.json").write_text("{}\n", encoding="utf-8")
    ground_truth = sample / "gt_motion.npz"
    ground_truth.write_bytes(b"ground-truth-a")
    first = benchmark_inventory_sha256(tmp_path)
    assert first == benchmark_inventory_sha256(tmp_path)

    ground_truth.write_bytes(b"ground-truth-b")
    assert benchmark_inventory_sha256(tmp_path) != first


def test_benchmark_inventory_full_hash_is_cached_and_mutation_is_rejected(tmp_path):
    benchmark = tmp_path / "benchmark"
    sample = benchmark / "content" / "case" / "0000"
    sample.mkdir(parents=True)
    (sample / "meta.json").write_text("{}\n", encoding="utf-8")
    (sample / "gt_motion.npz").write_bytes(b"fixed")
    output = tmp_path / "evaluation"
    output.mkdir()

    expected = resolve_benchmark_inventory_sha256(benchmark, output)
    assert resolve_benchmark_inventory_sha256(benchmark, output) == expected
    (sample / "gt_motion.npz").write_bytes(b"mutated")
    try:
        resolve_benchmark_inventory_sha256(benchmark, output)
    except RuntimeError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("mutated benchmark proxy was accepted")


def test_trend_alert_requires_two_consecutive_significant_regressions():
    def record(step: int, r_at_3: float, position_cm: float) -> dict:
        return {
            "step": step,
            "summary": {
                "tables": {
                    "content": {
                        "R@3 (gen)": r_at_3,
                        "End-Effector Pos (gen, cm)": position_cm,
                    }
                }
            },
        }

    history = [
        record(100_000, 80.0, 2.0),
        record(200_000, 76.0, 2.7),
        record(300_000, 72.0, 3.6),
    ]
    alerts = trend_alerts(history)
    assert {alert["metric"].split("/")[-1] for alert in alerts} == {
        "R@3 (gen)",
        "End-Effector Pos (gen, cm)",
    }
    assert trend_alerts(history[:2]) == []
