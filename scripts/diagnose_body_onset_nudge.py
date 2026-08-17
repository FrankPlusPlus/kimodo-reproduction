#!/usr/bin/env python3
"""Read-only onset nudge: does the 7-term loss reward last-layer cancel?

Freeze weights. Rotate last-layer attn/FFN cosine by ±delta on the same
frozen batch and compare training losses. No optimizer step.

Primary checkpoint is the *pre-flip* weights (kf-smooth 690k), not 800k.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from kimodo.training.body_onset_nudge_probe import probe_onset_nudge, summarize_onset_nudge
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


def _write_partial(output_dir: Path | None, row: dict) -> None:
    if output_dir is None:
        return
    step = int(row["global_step"])
    delta = float(row.get("cosine_delta") or 0.0)
    targets = "-".join(sorted(row.get("targets") or [])) or "identity"
    name = f"partial-step-{step:09d}-d{delta:+.2f}-{targets}.json"
    (output_dir / name).write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    loss_function = KimodoLoss(motion_rep, config.loss)
    constraint_step = int(args.constraint_step)
    conditioning = _jacobian._sample_constraints(motion_rep, config, batch, constraint_step, args.seed)
    layer_index = int(args.layer)
    delta = abs(float(args.cosine_delta))
    if len(args.checkpoints) < 2:
        raise ValueError("pass at least pre-flip and flipped checkpoints")

    common = dict(
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
        layer_index=layer_index,
    )
    interventions: list[tuple[float, tuple[str, ...]]] = [
        (0.0, ("attn",)),
        (-delta, ("attn",)),
        (delta, ("attn",)),
        (-delta, ("ffn",)),
        (delta, ("ffn",)),
    ]

    rows = []
    for checkpoint in args.checkpoints:
        path = Path(checkpoint).expanduser().resolve()
        print(json.dumps({"status": "loading_checkpoint", "path": str(path)}), flush=True)
        weights, summary = _jacobian._load_checkpoint_weights(path)
        model.load_state_dict(weights, strict=True)
        step = int(summary["global_step"])
        for cosine_delta, targets in interventions:
            probe = probe_onset_nudge(
                model, cosine_delta=cosine_delta, targets=targets, **common
            )
            row = {
                "checkpoint": str(path),
                "global_step": step,
                "cosine_delta": cosine_delta,
                "targets": list(targets),
                "probe": probe,
            }
            rows.append(row)
            _write_partial(output_dir, row)
            attn = (probe.get("nudge") or {}).get("attn") or {}
            ffn = (probe.get("nudge") or {}).get("ffn") or {}
            print(
                json.dumps(
                    {
                        "status": "probed",
                        "global_step": step,
                        "cosine_delta": cosine_delta,
                        "targets": list(targets),
                        "loss": probe.get("loss_total_mean"),
                        "attn_cosine": attn.get("after_cosine"),
                        "ffn_cosine": ffn.get("after_cosine"),
                    },
                    indent=2,
                ),
                flush=True,
            )

    verdict = summarize_onset_nudge(
        rows,
        preflip_step=int(args.preflip_step),
        flipped_step=int(args.flipped_step),
        healthy_step=int(args.healthy_step),
        cosine_delta=delta,
    )
    payload = {
        "rank": rank,
        "world_size": world,
        "seed": int(args.seed),
        "samples": int(args.samples),
        "constraint_step": constraint_step,
        "device": str(device),
        "rows": rows,
        "verdict": verdict,
    }
    print(json.dumps({"verdict": verdict}, indent=2), flush=True)
    if output_dir is not None:
        (output_dir / f"rank-{rank:02d}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if rank == 0:
            (output_dir / "verdict.json").write_text(
                json.dumps({"verdict": verdict}, indent=2, sort_keys=True) + "\n",
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
    parser.add_argument("--constraint-step", type=int, default=695000)
    parser.add_argument("--healthy-step", type=int, default=650000)
    parser.add_argument("--preflip-step", type=int, default=690000)
    parser.add_argument("--flipped-step", type=int, default=695000)
    parser.add_argument("--cosine-delta", type=float, default=0.25)
    parser.add_argument("--layer", type=int, default=15)
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
