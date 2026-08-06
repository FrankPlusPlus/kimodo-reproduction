from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kimodo.training.core_subset_cli import build_core_subset


def test_core_subset_is_zero_copy_stratified_and_take_grouped(tmp_path: Path):
    source = tmp_path / "source" / "train.cached.jsonl"
    source.parent.mkdir()
    rows = []
    metadata_rows = []
    for index, (take, mirror, movement) in enumerate(
        [
            ("walk_take", False, "walking"),
            ("walk_take", True, "walking"),
            ("dance_take", False, "dancing"),
            ("jump_take", False, "jumping"),
            ("sit_take", False, "sitting"),
        ]
    ):
        name = f"motion_{index}"
        motion = source.parent / f"{name}.npz"
        embedding = source.parent / f"{name}.npy"
        embedding_metadata = source.parent / f"{name}.npy.metadata.json"
        for path in (motion, embedding, embedding_metadata):
            path.write_bytes(name.encode())
        rows.append(
            {
                "id": name,
                "motion": motion.name,
                "frame_count": 120,
                "source_fps": 30,
                "split": "train",
                "text": movement,
                "text_embedding": embedding.name,
                "text_embedding_metadata": embedding_metadata.name,
            }
        )
        metadata_rows.append(
            {
                "move_name": name,
                "filename": name,
                "is_mirror": mirror,
                "take_name": take,
                "package": "Dances" if movement == "dancing" else "Locomotion",
                "category": movement,
                "content_type_of_movement": movement,
                "actor_gender": "F" if index % 2 else "M",
            }
        )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metadata = tmp_path / "metadata.csv"
    fields = ["move_name", "filename", "is_mirror", "take_name", *_metadata_fields()]
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata_rows)

    output = tmp_path / "core" / "manifest.jsonl"
    receipt = build_core_subset(
        argparse.Namespace(
            source_manifest=str(source),
            metadata_csv=str(metadata),
            output=str(output),
            name="test-core",
            target_hours=1.0,
            validation_fraction=0.25,
            fps=30,
            source_split="train",
            seed=7,
        )
    )
    selected = [json.loads(line) for line in output.read_text().splitlines()]
    assert "motion_1" not in {Path(row["motion"]).stem for row in selected}
    assert {row["split"] for row in selected} == {"train", "validation"}
    assert all((output.parent / row["motion"]).resolve().is_file() for row in selected)
    assert receipt["selected"]["motions"] == 4
    assert receipt["output"]["entries"] == 4
    assert output.with_suffix(".jsonl.metadata.json").is_file()


def _metadata_fields() -> list[str]:
    return [
        "package",
        "category",
        "content_all_rigplay_styles",
        "content_uniform_style",
        "content_type_of_movement",
        "content_body_position",
        "content_horizontal_move",
        "content_vertical_move",
        "content_props",
        "content_complex_action",
        "content_repeated_action",
        "actor_gender",
    ]
