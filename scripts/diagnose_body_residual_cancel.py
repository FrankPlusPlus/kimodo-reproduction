#!/usr/bin/env python3
"""Read-only residual-cancellation probe: 795k vs 800k last-layer x vs attn(x).

Forward only. Photographs whether L15 attention opposes the residual stream,
making x+attn(x) quiet before LayerNorm. No optimizer step, no weight write.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from kimodo.training.body_residual_cancel_probe import (
    compare_residual_rows,
    full_stack_timeline,
    probe_residual_cancel,
    residual_timeline,
    summarize_attn_ffn_asymmetry,
    summarize_residual_cancel,
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


def _layers(args) -> list[int]:
    if getattr(args, "all_layers", False):
        return list(range(16))
    if args.layers:
        return [int(index) for index in args.layers]
    return [0, 7, 13, 14, 15]


def diagnose(args) -> dict:
    rank, world, local_rank = _jacobian._distributed_context()
    args.seed = int(args.seed) + rank * 10007
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "running.json").write_text(
            json.dumps({"status": "starting", "rank": rank}, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "starting", "output_dir": str(output_dir)}), flush=True)
    config = load_training_config(Path(args.config).expanduser().resolve())
    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    print(json.dumps({"status": "loading_batch", "device": str(device)}), flush=True)
    batch, noisy, timesteps, motion_rep = _jacobian._prepare_frozen_inputs(config, args, device)
    print(json.dumps({"status": "batch_ready", "samples": int(args.samples)}), flush=True)
    model = build_trainable_denoiser(config.model, config.curriculum, device)
    set_model_dropout(model, 0.0)
    model.train()
    constraint_steps = _jacobian._constraint_steps(args)
    layers = _layers(args)
    if len(args.checkpoints) < 2:
        raise ValueError("pass at least two checkpoints")
    rows = []
    for checkpoint in args.checkpoints:
        path = Path(checkpoint).expanduser().resolve()
        print(json.dumps({"status": "loading_checkpoint", "path": str(path)}), flush=True)
        weights, summary = _jacobian._load_checkpoint_weights(path)
        model.load_state_dict(weights, strict=True)
        for constraint_step in constraint_steps:
            conditioning = _jacobian._sample_constraints(
                motion_rep, config, batch, int(constraint_step), args.seed
            )
            probe = probe_residual_cancel(
                model,
                noisy=noisy,
                valid_frames=batch["valid_frames"],
                text_features=batch["text_features"],
                text_pad_mask=batch["text_pad_mask"],
                timesteps=timesteps,
                first_heading_angle=batch["first_heading_angle"],
                motion_mask=conditioning.motion_mask,
                observed_motion=conditioning.observed_motion,
                layer_indices=layers,
            )
            row = {
                "checkpoint": str(path),
                "global_step": summary["global_step"],
                "constraint_step": int(constraint_step),
                "probe": probe,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "status": "probed",
                        "global_step": summary["global_step"],
                        "constraint_step": int(constraint_step),
                        "layers": probe["layers"],
                    },
                    indent=2,
                ),
                flush=True,
            )
            if output_dir is not None:
                (output_dir / f"partial-step-{summary['global_step']:09d}-c{int(constraint_step):09d}.json").write_text(
                    json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
    timeline = residual_timeline(rows)
    full_stack = full_stack_timeline(rows, constraint_step=constraint_steps[-1]) if getattr(args, "all_layers", False) else []
    asymmetry = summarize_attn_ffn_asymmetry(full_stack) if full_stack else {}
    same_clock = [row for row in rows if row.get("constraint_step") == constraint_steps[-1]]
    ratios = (
        compare_residual_rows(same_clock[-1]["probe"], same_clock[0]["probe"])
        if len(same_clock) >= 2
        else {}
    )
    verdict = summarize_residual_cancel(ratios) if ratios else {"verdict": "incomplete"}
    payload = {
        "rank": rank,
        "world_size": world,
        "seed": int(args.seed),
        "samples": int(args.samples),
        "constraint_steps": constraint_steps,
        "device": str(device),
        "layers": layers,
        "rows": rows,
        "timeline": timeline,
        "full_stack_timeline": full_stack,
        "asymmetry": asymmetry,
        "ratios": ratios,
        "verdict": verdict,
    }
    print(json.dumps({"timeline": timeline, "asymmetry": asymmetry, "verdict": verdict}, indent=2), flush=True)
    if output_dir is not None:
        (output_dir / f"rank-{rank:02d}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if rank == 0:
            (output_dir / "verdict.json").write_text(
                json.dumps(
                    {"verdict": verdict, "ratios": ratios, "timeline": timeline, "asymmetry": asymmetry},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            running = output_dir / "running.json"
            if running.is_file():
                running.unlink()
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
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--all-layers", action="store_true")
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
