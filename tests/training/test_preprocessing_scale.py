from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
import torch

from kimodo.training import stats_cli, text_cache_cli


class _FakeEncoder:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.calls: list[str] = []
        self.fail_after = fail_after

    def __call__(self, texts: list[str]):
        assert len(texts) == 1
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("injected encoder failure")
        self.calls.append(texts[0])
        value = float(len(self.calls))
        return torch.full((1, 1, 4096), value), [1]


def _text_cache_args(source, destination, cache_dir):
    return argparse.Namespace(
        manifest=str(source),
        output_manifest=str(destination),
        cache_dir=str(cache_dir),
        provider="api",
    )


def test_text_cache_streams_equivalent_rows_and_reuses_cache(monkeypatch, tmp_path):
    source = tmp_path / "source.jsonl"
    records = [
        {"id": "a", "text": "A person walks."},
        {"id": "b", "text": "A person jumps."},
        {"id": "c", "text": "A person walks."},
    ]
    source.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    destination = tmp_path / "derived" / "cached.jsonl"
    cache_dir = tmp_path / "cache"
    encoder = _FakeEncoder()
    identity = "fake-encoder:revision"
    monkeypatch.setattr(text_cache_cli, "_build_encoder", lambda args: (encoder, identity))

    text_cache_cli.run(_text_cache_args(source, destination, cache_dir))

    assert encoder.calls == ["A person walks.", "A person jumps."]
    actual = destination.read_text(encoding="utf-8").splitlines()
    expected = []
    for record in records:
        key = text_cache_cli._cache_key(record["text"], identity)
        expected_record = {
            **record,
            "text_embedding": str((cache_dir / f"{key}.npy").resolve()),
            "text_cache_key": key,
        }
        expected.append(json.dumps(expected_record, ensure_ascii=False, sort_keys=True))
        cached = np.load(cache_dir / f"{key}.npy", allow_pickle=False)
        assert cached.shape == (1, 4096)
        assert cached.dtype == np.float32
    assert actual == expected
    assert len(list(cache_dir.glob("*.npy"))) == 2
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))
    metadata = json.loads(
        destination.with_suffix(".jsonl.metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["internal_batch_size"] == 1


def test_text_cache_failure_does_not_publish_partial_manifest(monkeypatch, tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"id": "a", "text": "A person walks."})
        + "\n"
        + json.dumps({"id": "b", "text": "A person jumps."})
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "derived" / "cached.jsonl"
    encoder = _FakeEncoder(fail_after=1)
    monkeypatch.setattr(
        text_cache_cli,
        "_build_encoder",
        lambda args: (encoder, "fake-encoder:revision"),
    )

    with pytest.raises(RuntimeError, match="injected encoder failure"):
        text_cache_cli.run(_text_cache_args(source, destination, tmp_path / "cache"))

    assert not destination.exists()
    assert not destination.with_suffix(".jsonl.metadata.json").exists()
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_stats_loads_each_motion_once_without_dropping_distinct_spans(
    monkeypatch, training_fixture, tmp_path
):
    manifest = tmp_path / "spans.jsonl"
    records = [
        {
            "id": "full",
            "motion": str(training_fixture["motion"]),
            "text": "full",
            "split": "train",
            "source_fps": 30,
        },
        {
            "id": "early",
            "motion": str(training_fixture["motion"]),
            "text": "early",
            "split": "train",
            "source_fps": 30,
            "start_time": 0.0,
            "end_time": 0.1,
        },
        {
            "id": "late",
            "motion": str(training_fixture["motion"]),
            "text": "late",
            "split": "train",
            "source_fps": 30,
            "start_time": 0.1,
            "end_time": 0.2,
        },
        # An exact span duplicate represents another caption and must not alter stats.
        {
            "id": "late-caption-2",
            "motion": str(training_fixture["motion"]),
            "text": "late again",
            "split": "train",
            "source_fps": 30,
            "start_time": 0.1,
            "end_time": 0.2,
        },
    ]
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    original_loader = stats_cli.load_motion_file
    calls = []

    def counting_loader(*args, **kwargs):
        calls.append((args, kwargs))
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(stats_cli, "load_motion_file", counting_loader)
    output = tmp_path / "stats"
    stats_cli.compute_stats(
        argparse.Namespace(
            manifest=str(manifest),
            output=str(output),
            split="train",
            skeleton_joints=30,
            fps=30,
            seed=1234,
            max_seconds=0.1,
        )
    )

    assert len(calls) == 1
    metadata = json.loads((output / "stats.metadata.json").read_text(encoding="utf-8"))
    assert metadata["unique_clips"] == 3
    assert metadata["preprocessing"]["stats_window_count"] == 5
    # Full clip [3, 3, 2] plus two distinct 3-frame temporal spans.
    assert metadata["frame_counts"] == {"global_root": 14, "local_root": 14, "body": 14}
