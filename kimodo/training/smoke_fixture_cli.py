# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create the deterministic synthetic assets used by the tiny training smoke config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def create_smoke_fixture(output: str | Path) -> dict[str, Path]:
    root = Path(output).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty smoke fixture directory: {root}")
    root.mkdir(parents=True, exist_ok=True)

    stats = root / "stats"
    for name, width in (("global_root", 5), ("local_root", 4), ("body", 364)):
        folder = stats / name
        folder.mkdir(parents=True, exist_ok=False)
        np.save(folder / "mean.npy", np.zeros(width, dtype=np.float32))
        np.save(folder / "std.npy", np.ones(width, dtype=np.float32))

    frames, joints = 8, 30
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32), (frames, joints, 3, 3)
    ).copy()
    roots = np.zeros((frames, 3), dtype=np.float32)
    roots[:, 0] = np.linspace(0.0, 0.14, frames, dtype=np.float32)
    roots[:, 1] = 1.0
    motion = root / "motion.npz"
    np.savez(motion, local_rot_mats=rotations, root_positions=roots)

    embedding = root / "embedding.npy"
    np.save(
        embedding,
        np.linspace(-0.2, 0.2, 32, dtype=np.float32).reshape(2, 16),
        allow_pickle=False,
    )
    manifest = root / "manifest.jsonl"
    records = [
        {
            "id": f"synthetic-{index}",
            "motion": motion.name,
            "text": f"synthetic walking sample {index}",
            "split": "train",
            "source_fps": 30,
            "text_embedding": embedding.name,
            "sample_kind": "full",
        }
        for index in range(4)
    ]
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    result = {"root": root, "stats": stats, "motion": motion, "embedding": embedding, "manifest": manifest}
    print(json.dumps({name: str(path) for name, path in result.items()}, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tests/fixtures/training")
    create_smoke_fixture(parser.parse_args().output)


if __name__ == "__main__":
    main()
