from __future__ import annotations

import sys
import types

import pytest

from kimodo.monitoring import WandbMonitor
from kimodo.training.engine import JsonlLogger


class _Summary(dict):
    def update(self, values):
        super().update(values)


class _Run:
    def __init__(self):
        self.summary = _Summary()
        self.metrics = []
        self.records = []
        self.exit_code = None

    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def log(self, record):
        self.records.append(record)

    def finish(self, *, exit_code=0):
        self.exit_code = exit_code


def test_wandb_is_not_imported_when_monitoring_is_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("KIMODO_WANDB_ENABLED", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    monitor = WandbMonitor.from_env("train", output_dir=tmp_path)
    assert not monitor.enabled
    assert "wandb" not in sys.modules


def test_wandb_key_enables_monitoring_and_derives_stable_run_identity(monkeypatch, tmp_path):
    run = _Run()
    captured = {}

    def init(**kwargs):
        captured.update(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=init))
    monkeypatch.delenv("KIMODO_WANDB_ENABLED", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "injected-secret")
    monitor = WandbMonitor.from_env(
        "benchmark",
        output_dir=tmp_path / "eval/.wandb",
        identity_root=tmp_path / "runs/v2-1m-production",
    )
    assert monitor.enabled
    assert captured["project"] == "kimodo-reproduction"
    assert captured["group"].startswith("v2-1m-production-")
    assert captured["id"] == captured["name"]
    assert captured["id"].endswith("-benchmark")


def test_wandb_monitor_uses_scoped_identity_and_redacts_secrets(monkeypatch, tmp_path):
    run = _Run()
    captured = {}

    def init(**kwargs):
        captured.update(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=init))
    monkeypatch.setenv("KIMODO_WANDB_ENABLED", "1")
    monkeypatch.setenv("WANDB_PROJECT", "kimodo-production")
    monkeypatch.setenv("KIMODO_WANDB_GROUP", "run-001")
    monkeypatch.setenv("KIMODO_WANDB_TRAIN_RUN_ID", "train-001")
    monkeypatch.setenv("KIMODO_WANDB_TRAIN_RUN_NAME", "train")
    monitor = WandbMonitor.from_env(
        "train",
        output_dir=tmp_path,
        config={"runtime": {"batch_size": 128}, "api_key": "must-not-leak"},
        metadata={"kimodo/world_size": 16},
    )

    assert monitor.enabled
    assert captured["project"] == "kimodo-production"
    assert captured["group"] == "run-001"
    assert captured["id"] == "train-001"
    assert captured["resume"] == "allow"
    assert captured["config"]["api_key"] == "<redacted>"
    assert run.summary["kimodo/world_size"] == 16

    monitor.log({"loss/total": 2.5}, step=100)
    assert run.records == [{"loss/total": 2.5, "global_step": 100}]
    monitor.finish(exit_code=0)
    assert run.exit_code == 0


def test_jsonl_logger_preserves_local_log_and_forwards_rank_zero_metrics(tmp_path):
    run = _Run()
    monitor = WandbMonitor(run)
    path = tmp_path / "train.jsonl"
    logger = JsonlLogger(path, True, monitor)
    logger.write({"global_step": 10, "loss/total": 1.5})
    assert '"loss/total": 1.5' in path.read_text(encoding="utf-8")
    assert run.records[-1]["global_step"] == 10


def test_required_wandb_initialization_failure_is_fatal(monkeypatch, tmp_path):
    def init(**_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=init))
    monkeypatch.setenv("KIMODO_WANDB_ENABLED", "1")
    monkeypatch.setenv("KIMODO_WANDB_REQUIRED", "1")
    with pytest.raises(RuntimeError, match="required W&B initialization failed"):
        WandbMonitor.from_env("train", output_dir=tmp_path)
