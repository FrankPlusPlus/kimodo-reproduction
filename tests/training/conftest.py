from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def build_training_fixture(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    stats = root / "stats"
    for name, width in (("global_root", 5), ("local_root", 4), ("body", 364)):
        folder = stats / name
        folder.mkdir(parents=True)
        np.save(folder / "mean.npy", np.zeros(width, dtype=np.float32))
        np.save(folder / "std.npy", np.ones(width, dtype=np.float32))

    frames = 8
    joints = 30
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (frames, joints, 3, 3)).copy()
    roots = np.zeros((frames, 3), dtype=np.float32)
    roots[:, 0] = np.linspace(0.0, 0.14, frames, dtype=np.float32)
    roots[:, 1] = 1.0
    motion = root / "motion.npz"
    np.savez(motion, local_rot_mats=rotations, root_positions=roots)

    embedding = root / "embedding.npy"
    np.save(embedding, np.linspace(-0.2, 0.2, 32, dtype=np.float32).reshape(2, 16))
    manifest = root / "manifest.jsonl"
    records = [
        {
            "id": f"synthetic-{index}",
            "motion": motion.name,
            "text": f"synthetic walking sample {index}",
            "split": "train",
            "source_fps": 30,
            "text_embedding": embedding.name,
        }
        for index in range(4)
    ]
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return {"root": root, "stats": stats, "motion": motion, "embedding": embedding, "manifest": manifest}


@pytest.fixture
def training_fixture(tmp_path):
    return build_training_fixture(tmp_path / "fixture")
