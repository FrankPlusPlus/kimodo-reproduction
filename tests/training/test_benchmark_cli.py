from __future__ import annotations

import json

import numpy as np

import pytest

from kimodo.training.benchmark_cli import _count_periodic_updates, create_fixture


def test_benchmark_fixture_has_production_shapes(tmp_path):
    root = tmp_path / "benchmark"
    create_fixture(root, samples=3, frames=300, llm_dim=4096)

    with np.load(root / "motion-300f.npz", allow_pickle=False) as motion:
        assert motion["local_rot_mats"].shape == (300, 30, 3, 3)
        assert motion["root_positions"].shape == (300, 3)
    assert np.load(root / "llm2vec-4096d.npy", allow_pickle=False).shape == (1, 4096)
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    assert len(records) == 3
    assert len({record["id"] for record in records}) == 3


@pytest.mark.parametrize(
    ("start_step", "end_step", "update_every", "expected"),
    [
        (0, 10, 10, 1),
        (1, 9, 10, 0),
        (4, 12, 10, 1),
        (12, 20, 10, 1),
        (10, 20, 10, 1),
        (9, 20, 10, 2),
    ],
)
def test_count_periodic_updates_uses_open_closed_window(
    start_step, end_step, update_every, expected
):
    assert _count_periodic_updates(start_step, end_step, update_every) == expected


@pytest.mark.parametrize("values", [(-1, 1, 1), (2, 1, 1), (0, 1, 0)])
def test_count_periodic_updates_rejects_invalid_intervals(values):
    with pytest.raises(ValueError, match="Invalid periodic-update interval"):
        _count_periodic_updates(*values)
