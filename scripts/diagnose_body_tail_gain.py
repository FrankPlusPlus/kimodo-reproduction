#!/usr/bin/env python3
"""Read-only tail-gain probe: 795k vs 800k weight spectra + layer I/O rank.

Loads two checkpoints, compares L14/L15 (and L00/L07/L13) matrix singular
values, then runs one frozen-batch forward/backward for activation effective
rank and incoming gradient RMS. No optimizer step, no weight write.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from kimodo.training.body_tail_gain_probe import (
    collect_tail_matrices,
    compare_matrix_spectra,
    compare_tail_io,
    probe_tail_io,
    summarize_tail_gain,
)
from kimodo.training.config import load_training_config
from kimodo.training.losses import KimodoLoss
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


def _layers(args) -> list[int]:
    if args.layers:
        return [int(index) for index in args.layers]
    return [0, 7, 13, 14, 15]


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
    checkpoints = [Path(path).expanduser().resolve() for path in args.checkpoints]
    if len(checkpoints) != 2:
        raise ValueError("pass exactly two checkpoints: healthy then takeoff")
    weights = []
    steps = []
    for path in checkpoints:
        state, summary = _jacobian._load_checkpoint_weights(path)
        weights.append(state)
        steps.append(int(summary["global_step"]))
    layers = _layers(args)
    baseline_mats = collect_tail_matrices(weights[0], layers=layers)
    current_mats = collect_tail_matrices(weights[1], layers=layers)
    spectra = compare_matrix_spectra(current_mats, baseline_mats)
    print(json.dumps({"spectra_hot": {k: v for k, v in spectra.items() if abs((v.get("spectral_norm_ratio") or 1) - 1) >= 0.05}}, indent=2), flush=True)

    constraint_step = int(args.constraint_step)
    batch, noisy, timesteps, motion_rep = _jacobian._prepare_frozen_inputs(config, args, device)
    model = build_trainable_denoiser(config.model, config.curriculum, device)
    set_model_dropout(model, 0.0)
    model.train()
    loss_function = KimodoLoss(model.motion_rep, config.loss)
    io_rows = []
    for state, step in zip(weights, steps):
        model.load_state_dict(state, strict=True)
        conditioning = _jacobian._sample_constraints(motion_rep, config, batch, constraint_step, args.seed)
        probe = probe_tail_io(
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
            layer_indices=layers,
        )
        row = {"checkpoint_step": step, "probe": probe}
        io_rows.append(row)
        print(json.dumps({"io_step": step, "layers": probe["layers"]}, indent=2), flush=True)
    io_ratios = compare_tail_io(io_rows[1]["probe"], io_rows[0]["probe"])
    verdict = summarize_tail_gain(spectra, io_ratios)
    payload = {
        "rank": rank,
        "world_size": world,
        "seed": int(args.seed),
        "samples": int(args.samples),
        "constraint_step": constraint_step,
        "device": str(device),
        "healthy_step": steps[0],
        "takeoff_step": steps[1],
        "layers": layers,
        "spectra": spectra,
        "io": io_rows,
        "io_ratios": io_ratios,
        "verdict": verdict,
    }
    print(json.dumps({"verdict": verdict}, indent=2), flush=True)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"rank-{rank:02d}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if rank == 0:
            (output_dir / "verdict.json").write_text(
                json.dumps({"verdict": verdict, "io_ratios": io_ratios}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    elif args.output and rank == 0:
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs=2, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--constraint-step", type=int, default=795000)
    parser.add_argument("--layers", nargs="+", type=int)
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
