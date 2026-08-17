#!/usr/bin/env python3
"""Read-only body attention probe across trainer checkpoints.

Loads one frozen batch, then for each checkpoint runs a training-mode forward
with dropout 0 and records last-layer (and all-layer) attention entropy plus
mass on constrained motion tokens. No optimizer step, no weight write.

Default comparison is rescue 795k vs 800k weights on constraint clocks
795000 and 800000 (same 2x2 layout as the Jacobian probe).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch

from kimodo.training.body_attention_probe import (
    compare_attention_rows,
    median_pointer_grids,
    probe_body_attention,
    summarize_pointer_grid,
)
from kimodo.training.config import load_training_config
from kimodo.training.modeling import build_trainable_denoiser, set_model_dropout


def _load_jacobian_diagnose():
    path = Path(__file__).resolve().parent / "diagnose_body_jacobian.py"
    spec = importlib.util.spec_from_file_location("diagnose_body_jacobian", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_jacobian = _load_jacobian_diagnose()


def _layer_indices(args) -> list[int] | None:
    if not args.layers:
        return None
    return [int(index) for index in args.layers]


def _cell_scalars(cell: dict) -> dict:
    probe = cell.get("probe") or {}
    layers = {int(layer["index"]): layer for layer in probe.get("layers") or []}
    last = layers[max(layers)] if layers else {}
    unconstrained = last.get("unconstrained") or {}
    return {
        "weight_step": cell.get("weight_step"),
        "constraint_step": cell.get("constraint_step"),
        "last_layer": max(layers) if layers else None,
        "unconstrained_entropy": unconstrained.get("entropy"),
        "unconstrained_normalized_entropy": unconstrained.get("normalized_entropy"),
        "unconstrained_max_prob": unconstrained.get("max_prob"),
        "unconstrained_keyframe_mass": unconstrained.get("keyframe_mass"),
        "unconstrained_keyframe_lift": unconstrained.get("keyframe_lift"),
        "unconstrained_prefix_mass": unconstrained.get("prefix_mass"),
        "keyframe_frame_fraction": probe.get("keyframe_frame_fraction"),
    }


def diagnose(args) -> dict:
    rank, world, local_rank = _jacobian._distributed_context()
    args.seed = int(args.seed) + rank * 10007
    config = load_training_config(Path(args.config).expanduser().resolve())
    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    constraint_steps = _jacobian._constraint_steps(args)
    batch, noisy, timesteps, motion_rep = _jacobian._prepare_frozen_inputs(config, args, device)
    model = build_trainable_denoiser(config.model, config.curriculum, device)
    set_model_dropout(model, 0.0)
    model.train()
    layer_indices = _layer_indices(args)
    cells = []
    checkpoint_rows_by_clock: dict[int, list[dict]] = {step: [] for step in constraint_steps}
    for checkpoint in args.checkpoints:
        path = Path(checkpoint).expanduser().resolve()
        weights, summary = _jacobian._load_checkpoint_weights(path)
        model.load_state_dict(weights, strict=True)
        for constraint_step in constraint_steps:
            conditioning = _jacobian._sample_constraints(
                motion_rep, config, batch, constraint_step, args.seed
            )
            probe = probe_body_attention(
                model,
                noisy=noisy,
                valid_frames=batch["valid_frames"],
                text_features=batch["text_features"],
                text_pad_mask=batch["text_pad_mask"],
                timesteps=timesteps,
                first_heading_angle=batch["first_heading_angle"],
                motion_mask=conditioning.motion_mask,
                observed_motion=conditioning.observed_motion,
                layer_indices=layer_indices,
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
        "constraint_steps": constraint_steps,
        "device": str(device),
        "layers": layer_indices,
        "cells": cells,
        "pointer_grid": (
            summarize_pointer_grid(cells)
            if len(constraint_steps) >= 2 and len(args.checkpoints) >= 2
            else None
        ),
        "comparison_by_clock": {
            str(step): compare_attention_rows(rows) for step, rows in checkpoint_rows_by_clock.items()
        },
    }
    if payload["pointer_grid"] is not None:
        print(json.dumps({"rank": rank, "pointer_grid": payload["pointer_grid"]}, indent=2), flush=True)

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
                "pointer_grid": median_pointer_grids(
                    [item.get("pointer_grid") or {} for item in rank_payloads]
                ),
                "rank_verdicts": [item.get("pointer_grid", {}).get("verdict") for item in rank_payloads],
            }
            (output_dir / "verdict.json").write_text(
                json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps({"merged_pointer_grid": merged["pointer_grid"]}, indent=2), flush=True)
    elif args.output and rank == 0:
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--constraint-step", type=int, default=795000)
    parser.add_argument("--constraint-steps", nargs="+", type=int)
    parser.add_argument("--layers", nargs="+", type=int, help="Body layer indices. Default: all.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    try:
        diagnose(build_parser().parse_args())
    finally:
        _jacobian._shutdown_distributed()


if __name__ == "__main__":
    main()
