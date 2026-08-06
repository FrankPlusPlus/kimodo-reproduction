"""Two-rank exact-resume acceptance worker launched by test_ddp_resume.py."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kimodo.training.config import TrainingConfig
from kimodo.training.engine import KimodoTrainer


def _config(fixture: Path, output: Path, steps: int, benchmark_lane: bool) -> TrainingConfig:
    config = TrainingConfig()
    config.data.manifest = str(fixture / "manifest.jsonl")
    config.data.max_seconds = 1.0
    config.data.num_workers = 0
    config.data.pin_memory = False
    config.model.stats_path = str(fixture / "stats")
    config.model.llm_dim = 16
    config.model.num_text_tokens_override = 2
    config.model.latent_dim = 16
    config.model.ff_size = 32
    config.model.num_layers = 1
    config.model.num_heads = 4
    config.curriculum.phase1_steps = 1
    config.curriculum.phase2_steps = 1
    config.curriculum.sparse_keyframes_max = 2
    if benchmark_lane:
        config.curriculum.phase1_steps = 0
        config.curriculum.phase2_steps = 2
        config.curriculum.no_constraint_probability = 0.0
        config.curriculum.mix_two_probability = 0.0
        config.curriculum.benchmark_coverage_probability = 1.0
    config.runtime.output_dir = str(output)
    config.runtime.device = "cpu"
    config.runtime.precision = "fp32"
    config.runtime.batch_size = 1
    config.runtime.log_every = 1
    config.runtime.checkpoint_every = 1
    config.runtime.max_steps_override = steps
    config.ema.update_every = 1
    config.validate()
    return config


def _equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _logs(path: Path) -> list[dict]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for key in (
            "elapsed_seconds",
            "system/interval_seconds",
            "system/optimizer_steps_per_second",
            "system/samples_per_second",
        ):
            record.pop(key, None)
        result.append(record)
    return result


def _rank_main(
    rank: int,
    fixture_value: str,
    root_value: str,
    rendezvous_value: str,
    benchmark_lane: bool,
) -> None:
    fixture, root = Path(fixture_value), Path(root_value)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    dist.init_process_group(
        backend="gloo",
        init_method=Path(rendezvous_value).resolve().as_uri(),
        rank=rank,
        world_size=2,
    )
    project_root = Path(__file__).resolve().parents[2]
    continuous = root / "continuous"
    resumed = root / "resumed"

    KimodoTrainer(_config(fixture, continuous, 2, benchmark_lane), project_root).train()
    KimodoTrainer(_config(fixture, resumed, 1, benchmark_lane), project_root).train()
    checkpoint = resumed / "checkpoints" / "step-000000001.pt"
    second = _config(fixture, resumed, 2, benchmark_lane)
    second.runtime.resume = str(checkpoint)
    KimodoTrainer(second, project_root).train()

    dist.barrier()
    if dist.get_rank() == 0:
        expected = torch.load(
            continuous / "checkpoints" / "step-000000002.pt",
            map_location="cpu",
            weights_only=False,
        )
        actual = torch.load(
            resumed / "checkpoints" / "step-000000002.pt",
            map_location="cpu",
            weights_only=False,
        )
        for key in (
            "model",
            "optimizer",
            "ema",
            "scaler",
            "global_step",
            "epoch",
            "batch_in_epoch",
            "micro_index",
            "rng_by_rank",
        ):
            assert _equal(expected[key], actual[key]), key
        assert len(actual["rng_by_rank"]) == 2
        assert _logs(continuous / "train.jsonl") == _logs(resumed / "train.jsonl")
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    fixture, root = map(Path, sys.argv[1:3])
    benchmark_lane = len(sys.argv) == 4 and sys.argv[3] == "benchmark"
    root.mkdir(parents=True, exist_ok=True)
    rendezvous = root / "file-store"
    mp.spawn(
        _rank_main,
        args=(str(fixture), str(root), str(rendezvous), benchmark_lane),
        nprocs=2,
        join=True,
    )


if __name__ == "__main__":
    main()
