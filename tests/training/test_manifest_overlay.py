from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from kimodo.training.data import load_manifest
from kimodo.training.manifest_overlay_cli import build_overlay_manifest
from kimodo.training.reference_inventory import (
    build_reference_inventory,
    verify_reference_inventory_full,
)


def _write_manifest(path: Path, fixture: dict[str, Path], prefix: str, count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": f"{prefix}-{index}",
            "motion": str(fixture["motion"]),
            "text": f"{prefix} sample {index}",
            "split": "train",
            "source_fps": 30,
            "text_embedding": str(fixture["embedding"]),
        }
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_small_dance_overlay_is_deterministic_zero_copy_and_inventory_bound(
    training_fixture, tmp_path
):
    base = _write_manifest(tmp_path / "base" / "cached.jsonl", training_fixture, "base", 9)
    dance = _write_manifest(tmp_path / "dance" / "cached.jsonl", training_fixture, "dance", 2)
    output = tmp_path / "mixed" / "train.cached.jsonl"
    metadata = build_overlay_manifest(
        argparse.Namespace(
            base_manifest=str(base),
            overlay_manifest=str(dance),
            output=str(output),
            overlay_fraction=0.1,
            base_name="base",
            overlay_name="dance",
            split="train",
            seed=7,
        )
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert metadata["base_entries"] == 9
    assert metadata["overlay_entries"] == 1
    assert len(rows) == 10
    assert sum(row["mixture_source"] == "dance" for row in rows) == 1
    assert not list((tmp_path / "mixed").glob("*.npy"))
    assert {entry.mixture_source for entry in load_manifest(output, "train")} == {
        "base",
        "dance",
    }

    inventory = tmp_path / "mixed" / "references.jsonl"
    build_reference_inventory(output, inventory)
    assert verify_reference_inventory_full(output, inventory)["verification"] == (
        "full_content_verified"
    )


def test_overlay_rejects_invalid_fraction_and_overwrite(training_fixture, tmp_path):
    base = _write_manifest(tmp_path / "base.jsonl", training_fixture, "base", 2)
    dance = _write_manifest(tmp_path / "dance.jsonl", training_fixture, "dance", 1)
    output = tmp_path / "mixed.jsonl"
    args = argparse.Namespace(
        base_manifest=str(base),
        overlay_manifest=str(dance),
        output=str(output),
        overlay_fraction=0.2,
        base_name="base",
        overlay_name="dance",
        split="train",
        seed=1,
    )
    build_overlay_manifest(args)
    try:
        build_overlay_manifest(args)
    except FileExistsError:
        pass
    else:  # pragma: no cover
        raise AssertionError("overlay manifest was overwritten")

    args.output = str(tmp_path / "invalid.jsonl")
    args.overlay_fraction = 1.0
    try:
        build_overlay_manifest(args)
    except ValueError as error:
        assert "overlay_fraction" in str(error)
    else:  # pragma: no cover
        raise AssertionError("invalid overlay fraction was accepted")


def test_legacy_key_only_manifest_remains_trainable_but_schema5_fails_closed(
    training_fixture, tmp_path
):
    manifest = _write_manifest(tmp_path / "legacy.jsonl", training_fixture, "legacy", 1)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["text_cache_key"] = "a" * 64
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps({"schema_version": 4}) + "\n", encoding="utf-8"
    )
    assert len(load_manifest(manifest, "train")) == 1

    manifest.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps({"schema_version": 5}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incomplete text-embedding identity"):
        load_manifest(manifest, "train")
