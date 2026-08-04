from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from kimodo.model.llm2vec.llm2vec import LLM2Vec
from kimodo.training import stats_cli, text_cache_cli


class _FakeEncoder:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.calls: list[str] = []
        self.fail_after = fail_after

    def __call__(self, texts: list[str]):
        if self.fail_after is not None and len(self.calls) + len(texts) > self.fail_after:
            raise RuntimeError("injected encoder failure")
        values = []
        for text in texts:
            self.calls.append(text)
            values.append(torch.full((1, 4096), float(len(self.calls))))
        return torch.stack(values), [1] * len(values)


def test_llm2vec_explicit_device_never_enters_visible_multi_gpu_pool(monkeypatch):
    encoder = object.__new__(LLM2Vec)
    encoder.eval = lambda: encoder
    encoder.to = lambda device: encoder
    encoder._text_length = lambda unused: 1
    encoder._convert_to_str = lambda instruction, text: text
    encoder._encode = lambda texts, device, convert_to_numpy: torch.tensor(
        [[float(len(text))] for text in texts]
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    result = encoder.encode(
        ["short", "longer"],
        batch_size=1,
        show_progress_bar=False,
        device="cuda:1",
    )

    assert result.tolist() == [[5.0], [6.0]]


def _text_cache_args(source, destination, cache_dir):
    return argparse.Namespace(
        manifest=str(source),
        output_manifest=str(destination),
        cache_dir=str(cache_dir),
        provider="api",
    )


def _hold_output_lock(lock_path, ready, release):
    with text_cache_cli._exclusive_output_lock(Path(lock_path)):
        ready.set()
        if not release.wait(timeout=10):
            raise RuntimeError("timed out waiting to release test lock")


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
    metadata = json.loads(
        destination.with_suffix(".jsonl.metadata.json").read_text(encoding="utf-8")
    )
    bound_identity = metadata["encoder"]
    actual = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    for actual_record, record in zip(actual, records, strict=True):
        key = text_cache_cli._cache_key(record["text"], bound_identity)
        assert actual_record["id"] == record["id"]
        assert actual_record["text"] == record["text"]
        assert actual_record["text_embedding"] == Path(
            os.path.relpath(cache_dir / f"{key}.npy", destination.parent)
        ).as_posix()
        assert actual_record["text_embedding_metadata"] == Path(
            os.path.relpath(
                text_cache_cli.embedding_metadata_path(cache_dir / f"{key}.npy"),
                destination.parent,
            )
        ).as_posix()
        assert actual_record["text_cache_key"] == key
        cached = np.load(cache_dir / f"{key}.npy", allow_pickle=False)
        assert cached.shape == (1, 4096)
        assert cached.dtype == np.float32
        embedding_metadata = json.loads(
            text_cache_cli.embedding_metadata_path(cache_dir / f"{key}.npy").read_text(
                encoding="utf-8"
            )
        )
        assert actual_record["text_embedding_sha256"] == embedding_metadata["sha256"]
    assert len(list(cache_dir.glob("*.npy"))) == 2
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))
    assert metadata["internal_batch_size"] == 1

    corrupt_key = text_cache_cli._cache_key("A person walks.", bound_identity)
    (cache_dir / f"{corrupt_key}.npy").write_bytes(b"truncated")
    second_destination = tmp_path / "derived" / "cached-second.jsonl"
    second_encoder = _FakeEncoder()
    monkeypatch.setattr(
        text_cache_cli,
        "_build_encoder",
        lambda args: (second_encoder, identity),
    )
    text_cache_cli.run(_text_cache_args(source, second_destination, cache_dir))
    assert second_encoder.calls == ["A person walks."]
    assert text_cache_cli._embedding_is_valid(cache_dir / f"{corrupt_key}.npy")

    third_destination = tmp_path / "derived" / "cached-third.jsonl"
    monkeypatch.setattr(
        text_cache_cli,
        "_build_encoder",
        lambda args: pytest.fail("all valid embedding pairs should avoid loading the encoder"),
    )
    text_cache_cli.run(_text_cache_args(source, third_destination, cache_dir))


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


def test_text_cache_rejects_valid_shaped_embedding_swap(monkeypatch, tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"id": "a", "text": "A person walks."})
        + "\n"
        + json.dumps({"id": "b", "text": "A person jumps."})
        + "\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    first = _FakeEncoder()
    monkeypatch.setattr(
        text_cache_cli, "_build_encoder", lambda args: (first, "fake-encoder:revision")
    )
    first_output = tmp_path / "first.jsonl"
    text_cache_cli.run(_text_cache_args(source, first_output, cache))
    rows = [json.loads(line) for line in first_output.read_text(encoding="utf-8").splitlines()]
    first_path = (first_output.parent / rows[0]["text_embedding"]).resolve()
    second_path = (first_output.parent / rows[1]["text_embedding"]).resolve()
    first_bytes = first_path.read_bytes()
    first_path.write_bytes(second_path.read_bytes())
    second_path.write_bytes(first_bytes)

    regenerated = _FakeEncoder()
    monkeypatch.setattr(
        text_cache_cli,
        "_build_encoder",
        lambda args: (regenerated, "fake-encoder:revision"),
    )
    text_cache_cli.run(_text_cache_args(source, tmp_path / "second.jsonl", cache))
    assert regenerated.calls == ["A person walks.", "A person jumps."]


def test_text_cache_requires_one_pooled_token(tmp_path):
    with pytest.raises(ValueError, match="invalid embedding"):
        text_cache_cli._atomic_save_embedding(
            tmp_path / "multi.npy", np.zeros((2, 4096), dtype=np.float32)
        )


def test_text_cache_interrupted_embedding_write_is_not_reused(monkeypatch, tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"id": "a", "text": "A person walks."}) + "\n", encoding="utf-8")
    destination = tmp_path / "cached.jsonl"
    cache_dir = tmp_path / "cache"
    encoder = _FakeEncoder()
    monkeypatch.setattr(
        text_cache_cli,
        "_build_encoder",
        lambda args: (encoder, "fake-encoder:revision"),
    )

    def interrupted_save(output, array, *, allow_pickle):
        output.write(b"partial")
        raise OSError("injected write failure")

    monkeypatch.setattr(text_cache_cli.np, "save", interrupted_save)
    with pytest.raises(OSError, match="injected write failure"):
        text_cache_cli.run(_text_cache_args(source, destination, cache_dir))
    assert not list(cache_dir.glob("*.npy"))
    assert not list(cache_dir.glob(".*.tmp"))
    assert not destination.exists()
    assert not destination.with_suffix(".jsonl.metadata.json").exists()


def test_text_cache_never_publishes_manifest_before_sidecar(monkeypatch, tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"id": "a", "text": "A person walks."}) + "\n", encoding="utf-8")
    destination = tmp_path / "cached.jsonl"
    encoder = _FakeEncoder()
    monkeypatch.setattr(
        text_cache_cli,
        "_build_encoder",
        lambda args: (encoder, "fake-encoder:revision"),
    )
    monkeypatch.setattr(
        text_cache_cli,
        "_atomic_write_json",
        lambda path, payload: (_ for _ in ()).throw(OSError("injected sidecar failure")),
    )

    with pytest.raises(OSError, match="injected sidecar failure"):
        text_cache_cli.run(_text_cache_args(source, destination, tmp_path / "cache"))
    assert not destination.exists()
    assert not destination.with_suffix(".jsonl.metadata.json").exists()


def test_text_cache_rejects_orphaned_sidecar(monkeypatch, tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"id": "a", "text": "A person walks."}) + "\n", encoding="utf-8")
    destination = tmp_path / "cached.jsonl"
    destination.with_suffix(".jsonl.metadata.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="orphaned derived output"):
        text_cache_cli.run(_text_cache_args(source, destination, tmp_path / "cache"))


def test_text_cache_exclusive_lock_keeps_manifest_and_sidecar_consistent(
    monkeypatch, tmp_path
):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"id": "a", "text": "A person walks."}) + "\n", encoding="utf-8")
    destination = tmp_path / "cached.jsonl"
    entered = threading.Event()
    release = threading.Event()
    encoder = _FakeEncoder()

    def blocking_build(args):
        entered.set()
        assert release.wait(timeout=10)
        return encoder, "fake-encoder:revision"

    monkeypatch.setattr(text_cache_cli, "_build_encoder", blocking_build)
    errors = []

    def first_run():
        try:
            text_cache_cli.run(_text_cache_args(source, destination, tmp_path / "cache"))
        except Exception as error:  # noqa: BLE001  # pragma: no cover - asserted below
            errors.append(error)

    thread = threading.Thread(target=first_run)
    thread.start()
    assert entered.wait(timeout=10)
    with pytest.raises(FileExistsError, match="output is locked"):
        text_cache_cli.run(_text_cache_args(source, destination, tmp_path / "cache"))
    release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert not errors

    sidecar_path = destination.with_suffix(".jsonl.metadata.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["output"]["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert not destination.with_suffix(".jsonl.lock").exists()


def test_text_cache_rejects_stale_output_lock(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"id": "a", "text": "A person walks."}) + "\n", encoding="utf-8")
    destination = tmp_path / "cached.jsonl"
    lock = destination.with_suffix(".jsonl.lock")
    lock.write_text('{"hostname":"old-host","pid":123}\n', encoding="utf-8")
    with pytest.raises(FileExistsError, match="Inspect the recorded process"):
        text_cache_cli.run(_text_cache_args(source, destination, tmp_path / "cache"))
    assert lock.exists()


def test_text_cache_output_lock_is_exclusive_across_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    lock = tmp_path / "cached.jsonl.lock"
    process = context.Process(target=_hold_output_lock, args=(str(lock), ready, release))
    process.start()
    assert ready.wait(timeout=10)
    try:
        with (
            pytest.raises(FileExistsError, match="output is locked"),
            text_cache_cli._exclusive_output_lock(lock),
        ):
            pytest.fail("second process unexpectedly acquired the output lock")
    finally:
        release.set()
        process.join(timeout=10)
    assert process.exitcode == 0
    assert not lock.exists()


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
    original_loader = stats_cli._load_training_motion_file
    calls = []

    def counting_loader(*args, **kwargs):
        calls.append((args, kwargs))
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(stats_cli, "_load_training_motion_file", counting_loader)
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


def test_stats_worker_count_preserves_numeric_arrays(training_fixture, tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "clip",
                "motion": str(training_fixture["motion"]),
                "text": "A person moves.",
                "split": "train",
                "source_fps": 30,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    outputs = []
    for workers in (1, 2):
        output = tmp_path / f"stats-{workers}"
        stats_cli.compute_stats(
            argparse.Namespace(
                manifest=str(manifest),
                output=str(output),
                split="train",
                skeleton_joints=30,
                fps=30,
                seed=1234,
                max_seconds=10.0,
                num_workers=workers,
            )
        )
        outputs.append(output)

    for group in ("global_root", "local_root", "body"):
        for filename in ("mean.npy", "std.npy"):
            single = np.load(outputs[0] / group / filename, allow_pickle=False)
            parallel = np.load(outputs[1] / group / filename, allow_pickle=False)
            assert np.array_equal(single, parallel)

    single_metadata = json.loads(
        (outputs[0] / "stats.metadata.json").read_text(encoding="utf-8")
    )
    parallel_metadata = json.loads(
        (outputs[1] / "stats.metadata.json").read_text(encoding="utf-8")
    )
    assert single_metadata["preprocessing"]["worker_processes"] == 1
    assert parallel_metadata["preprocessing"]["worker_processes"] == 2
