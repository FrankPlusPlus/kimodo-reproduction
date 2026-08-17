#!/usr/bin/env python3
"""Read-only body Jacobian probe across trainer checkpoints.

Loads one frozen batch, then for each checkpoint runs a training-mode forward
and backwards with dropout 0. No optimizer step, no weight write.

Default comparison is kf-smooth 650k / 690k plus the K7 696k takeoff point,
on identical constraints sampled at --constraint-step (default 690000).
Pass two checkpoints and two --constraint-steps for a 2x2 takeoff grid.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from kimodo.model.diffusion import Diffusion
from kimodo.training.body_jacobian_probe import (
    compare_checkpoint_rows,
    median_takeoff_grids,
    probe_forward_backward,
    summarize_takeoff_grid,
)
from kimodo.training.config import load_training_config
from kimodo.training.constraints import ConstraintCurriculumSampler
from kimodo.training.data import MotionManifestDataset, collate_motion_batch
from kimodo.training.losses import KimodoLoss
from kimodo.training.modeling import build_trainable_denoiser, set_model_dropout


def _load_checkpoint_weights(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    weights = state.get("model")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Checkpoint has no model weights: {path}")
    return weights, {"global_step": int(state["global_step"])}


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _constraint_steps(args) -> list[int]:
    steps = [int(step) for step in (args.constraint_steps or [])]
    if steps:
        return steps
    return [int(args.constraint_step)]


def _distributed_context() -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def _shutdown_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _prepare_frozen_inputs(config, args, device):
    model = build_trainable_denoiser(config.model, config.curriculum, torch.device("cpu"))
    motion_rep = copy.deepcopy(model.motion_rep)
    for value in vars(motion_rep).values():
        if isinstance(value, torch.nn.Module):
            value.cpu()
    dataset = MotionManifestDataset(
        Path(config.data.manifest),
        "train",
        motion_rep,
        max_seconds=config.data.max_seconds,
        min_frames=config.data.min_frames,
        seed=args.seed,
        require_cached_text=True,
        require_paper_data_parity=False,
        normalize=True,
        augment=False,
        feature_cache_dir=config.data.feature_cache_dir,
        stats_path=config.model.stats_path,
    )
    sample_count = min(int(args.samples), len(dataset))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[
        :sample_count
    ].tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=sample_count,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_motion_batch,
    )
    batch = _to_device(next(iter(loader)), device)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    diffusion = Diffusion(config.model.num_diffusion_steps).to(device)
    noise = torch.randn(
        batch["clean_motion"].shape,
        dtype=batch["clean_motion"].dtype,
        device=device,
        generator=noise_generator,
    )
    timesteps = torch.randint(
        config.model.num_diffusion_steps,
        (len(batch["clean_motion"]),),
        device=device,
        generator=noise_generator,
    )
    noisy = diffusion.q_sample(batch["clean_motion"], timesteps, noise)
    del model
    return batch, noisy, timesteps, motion_rep


def _sample_constraints(motion_rep, config, batch, constraint_step: int, seed: int):
    sampler = ConstraintCurriculumSampler(motion_rep, config.curriculum)
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    return sampler.sample(
        batch["clean_motion"], batch["lengths"], int(constraint_step), cpu_generator
    )


def _cell_scalars(cell: dict) -> dict:
    probe = cell.get("probe") or {}
    return {
        "weight_step": cell.get("weight_step"),
        "constraint_step": cell.get("constraint_step"),
        "prediction_grad_norm": probe.get("prediction_grad_norm"),
        "body_batch_grad_norm": probe.get("body_batch_grad_norm"),
        "loss_total_mean": probe.get("loss_total_mean"),
    }


def diagnose(args) -> dict:
    rank, world, local_rank = _distributed_context()
    args.seed = int(args.seed) + rank * 10007
    config = load_training_config(Path(args.config).expanduser().resolve())
    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    constraint_steps = _constraint_steps(args)
    batch, noisy, timesteps, motion_rep = _prepare_frozen_inputs(config, args, device)
    model = build_trainable_denoiser(config.model, config.curriculum, device)
    set_model_dropout(model, 0.0)
    model.train()
    loss_function = KimodoLoss(model.motion_rep, config.loss)
    cells = []
    checkpoint_rows_by_clock: dict[int, list[dict]] = {step: [] for step in constraint_steps}
    for checkpoint in args.checkpoints:
        path = Path(checkpoint).expanduser().resolve()
        weights, summary = _load_checkpoint_weights(path)
        model.load_state_dict(weights, strict=True)
        for constraint_step in constraint_steps:
            conditioning = _sample_constraints(
                motion_rep, config, batch, constraint_step, args.seed
            )
            probe = probe_forward_backward(
                model,
                noisy=noisy,
                valid_frames=batch["valid_frames"],
                text_features=batch["text_features"],
                text_pad_mask=batch["text_pad_mask"],
                timesteps=timesteps,
                first_heading_angle=batch["first_heading_angle"],
                motion_mask=conditioning.motion_mask,
                observed_motion=conditioning.observed_motion,
                target=batch["clean_motion"],
                loss_function=loss_function,
                pair_samples=args.pair_samples,
            )
            cell = {
                "checkpoint": str(path),
                "weight_step": summary["global_step"],
                "global_step": summary["global_step"],
                "constraint_step": int(constraint_step),
                "probe": probe,
            }
            cells.append(cell)
            checkpoint_rows_by_clock[int(constraint_step)].append(
                {"checkpoint": str(path), "global_step": summary["global_step"], "probe": probe}
            )
            print(json.dumps(_cell_scalars(cell), sort_keys=True), flush=True)
    payload = {
        "rank": rank,
        "world_size": world,
        "seed": int(args.seed),
        "samples": int(args.samples),
        "pair_samples": int(args.pair_samples),
        "constraint_steps": constraint_steps,
        "device": str(device),
        "cells": cells,
        "takeoff_grid": summarize_takeoff_grid(cells) if len(constraint_steps) >= 2 and len(args.checkpoints) >= 2 else None,
        "comparison_by_clock": {
            str(step): compare_checkpoint_rows(rows) for step, rows in checkpoint_rows_by_clock.items()
        },
    }
    if len(constraint_steps) == 1:
        payload["constraint_step"] = constraint_steps[0]
        payload["checkpoints"] = checkpoint_rows_by_clock[constraint_steps[0]]
        payload["comparison_vs_first"] = payload["comparison_by_clock"][str(constraint_steps[0])]
    if payload["takeoff_grid"] is not None:
        print(json.dumps({"rank": rank, "takeoff_grid": payload["takeoff_grid"]}, indent=2), flush=True)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        rank_path = output_dir / f"rank-{rank:02d}.json"
        rank_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if world > 1:
            import torch.distributed as dist

            dist.barrier()
        if rank == 0:
            rank_payloads = []
            for index in range(world):
                path = output_dir / f"rank-{index:02d}.json"
                rank_payloads.append(json.loads(path.read_text(encoding="utf-8")))
            merged = {
                "world_size": world,
                "constraint_steps": constraint_steps,
                "samples_per_rank": int(args.samples),
                "takeoff_grid": median_takeoff_grids(
                    [item.get("takeoff_grid") or {} for item in rank_payloads]
                ),
                "rank_verdicts": [item.get("takeoff_grid", {}).get("verdict") for item in rank_payloads],
            }
            (output_dir / "verdict.json").write_text(
                json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps({"merged_takeoff_grid": merged["takeoff_grid"]}, indent=2), flush=True)
    elif args.output and rank == 0:
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--pair-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--constraint-step", type=int, default=690000)
    parser.add_argument("--constraint-steps", nargs="+", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    try:
        diagnose(build_parser().parse_args())
    finally:
        _shutdown_distributed()


if __name__ == "__main__":
    main()
