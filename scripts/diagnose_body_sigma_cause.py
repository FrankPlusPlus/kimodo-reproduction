#!/usr/bin/env python3
"""Causal σ/L15-grad probe: scale vs direction, 750k vs 790k.

Restores weights after each intervention. No checkpoint write.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from kimodo.training.body_sigma_cause_probe import (
    measure_l15_sigma_and_grad,
    scale_l15_in_proj_,
    snapshot_l15_attn,
    restore_l15_attn,
    summarize_sigma_cause,
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


def _run(model, batch, noisy, timesteps, conditioning, loss_function, **kwargs) -> dict:
    return measure_l15_sigma_and_grad(
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
        **kwargs,
    )


def diagnose(args) -> dict:
    rank, world, local_rank = _jacobian._distributed_context()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "running.json").write_text(json.dumps({"status": "starting"}, indent=2) + "\n")
        print(json.dumps({"status": "starting", "output_dir": str(output_dir)}), flush=True)
    config = load_training_config(Path(args.config).expanduser().resolve())
    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    batch, noisy, timesteps, motion_rep = _jacobian._prepare_frozen_inputs(config, args, device)
    conditioning = _jacobian._sample_constraints(
        motion_rep, config, batch, int(args.constraint_step), args.seed
    )
    model = build_trainable_denoiser(config.model, config.curriculum, device)
    set_model_dropout(model, 0.0)
    model.train()
    loss_function = KimodoLoss(model.motion_rep, config.loss)

    healthy_w, healthy_sum = _jacobian._load_checkpoint_weights(Path(args.healthy_checkpoint).resolve())
    crashed_w, crashed_sum = _jacobian._load_checkpoint_weights(Path(args.crashed_checkpoint).resolve())

    rows: list[dict] = []

    def add(name: str, payload: dict) -> None:
        row = {"name": name, **payload}
        rows.append(row)
        print(json.dumps({"status": "measured", **row}, default=str)[:1200], flush=True)
        if output_dir is not None:
            (output_dir / f"partial-{name}.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    model.load_state_dict(healthy_w, strict=True)
    healthy = _run(model, batch, noisy, timesteps, conditioning, loss_function)
    add("healthy_identity", healthy)

    model.load_state_dict(crashed_w, strict=True)
    crashed = _run(model, batch, noisy, timesteps, conditioning, loss_function)
    add("crashed_identity", crashed)

    attn_scale = float(healthy["attn_rms"]) / float(crashed["attn_rms"])
    attn_scale_up = float(crashed["attn_rms"]) / float(healthy["attn_rms"])
    healthy_cos = float(healthy["mean_token_cosine"])
    crashed_cos = float(crashed["mean_token_cosine"])

    model.load_state_dict(crashed_w, strict=True)
    add(
        "crashed_attn_scale_to_healthy",
        _run(model, batch, noisy, timesteps, conditioning, loss_function, attn_scale=attn_scale),
    )
    add(
        "crashed_cosine_to_healthy",
        _run(
            model,
            batch,
            noisy,
            timesteps,
            conditioning,
            loss_function,
            target_cosine=healthy_cos,
        ),
    )
    snap = snapshot_l15_attn(model)
    in_proj = model.body_model.seqTransEncoder.layers[15].self_attn.in_proj_weight
    crashed_in = float(in_proj.detach().float().pow(2).mean().sqrt())
    model.load_state_dict(healthy_w, strict=True)
    healthy_in = float(
        model.body_model.seqTransEncoder.layers[15].self_attn.in_proj_weight.detach().float().pow(2).mean().sqrt()
    )
    model.load_state_dict(crashed_w, strict=True)
    restore_l15_attn(model, snap)
    scale_l15_in_proj_(model, healthy_in / crashed_in)
    add(
        "crashed_in_proj_rms_to_healthy",
        _run(model, batch, noisy, timesteps, conditioning, loss_function),
    )
    restore_l15_attn(model, snap)

    model.load_state_dict(healthy_w, strict=True)
    add(
        "healthy_attn_scale_to_crashed",
        _run(model, batch, noisy, timesteps, conditioning, loss_function, attn_scale=attn_scale_up),
    )
    add(
        "healthy_cosine_to_crashed",
        _run(
            model,
            batch,
            noisy,
            timesteps,
            conditioning,
            loss_function,
            target_cosine=crashed_cos,
        ),
    )

    verdict = summarize_sigma_cause(rows)
    payload = {
        "healthy_step": healthy_sum["global_step"],
        "crashed_step": crashed_sum["global_step"],
        "attn_scale_down": attn_scale,
        "attn_scale_up": attn_scale_up,
        "in_proj_scale_down": healthy_in / crashed_in,
        "rows": rows,
        "verdict": verdict,
    }
    print(json.dumps({"verdict": verdict}, indent=2), flush=True)
    if output_dir is not None:
        (output_dir / "rank-00.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        (output_dir / "verdict.json").write_text(json.dumps({"verdict": verdict}, indent=2, sort_keys=True) + "\n")
        running = output_dir / "running.json"
        if running.is_file():
            running.unlink()
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--healthy-checkpoint", required=True)
    parser.add_argument("--crashed-checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--constraint-step", type=int, default=750000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    diagnose(build_parser().parse_args())


if __name__ == "__main__":
    main()
