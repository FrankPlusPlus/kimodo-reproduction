#!/usr/bin/env python3
"""First-cause config probe: official L15 geometry + multi-batch 1-step drift.

Photographs official SEED-v1.1 vs our checkpoints on one frozen batch, then
takes one optimizer step per fresh microbatch at a pre-flip checkpoint under
the knobs we filled in (wd, clip, λ, bf16, atan2 vs Adam). Restores weights
after every step. No checkpoint write.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
from pathlib import Path

import torch

from kimodo.training.body_firstcause_config import (
    layer15,
    summarize_multibatch_drift,
    summarize_official_vs_ours,
)
from kimodo.training.body_onset_path_probe import probe_virtual_steps
from kimodo.training.body_residual_cancel_probe import probe_residual_cancel
from kimodo.training.config import load_training_config
from kimodo.training.engine import _autocast_context
from kimodo.training.losses import KimodoLoss
from kimodo.training.modeling import (
    build_trainable_denoiser,
    load_official_trainable_denoiser,
    set_model_dropout,
)


def _load_jacobian_diagnose():
    path = Path(__file__).resolve().parent / "diagnose_body_jacobian.py"
    spec = importlib.util.spec_from_file_location("diagnose_body_jacobian", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_jacobian = _load_jacobian_diagnose()


def _slice_batch(batch: dict, noisy: torch.Tensor, timesteps: torch.Tensor, start: int, end: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[0] >= end:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced, noisy[start:end], timesteps[start:end]


def _photograph(model, batch, noisy, timesteps, conditioning, precision: str, device) -> dict:
    ctx = _autocast_context(device, precision) if precision != "fp32" else contextlib.nullcontext()
    with ctx:
        return probe_residual_cancel(
            model,
            noisy=noisy,
            valid_frames=batch["valid_frames"],
            text_features=batch["text_features"],
            text_pad_mask=batch["text_pad_mask"],
            timesteps=timesteps,
            first_heading_angle=batch["first_heading_angle"],
            motion_mask=conditioning.motion_mask,
            observed_motion=conditioning.observed_motion,
            layer_indices=(0, 14, 15),
        )


def diagnose(args) -> dict:
    rank, world, local_rank = _jacobian._distributed_context()
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
    conditioning = _jacobian._sample_constraints(
        motion_rep, config, batch, int(args.constraint_step), args.seed
    )
    photos = []

    if args.official_dir:
        print(json.dumps({"status": "loading_official", "path": args.official_dir}), flush=True)
        official = load_official_trainable_denoiser(args.official_dir, device)
        set_model_dropout(official, 0.0)
        official.train()
        for precision in args.precisions:
            probe = _photograph(official, batch, noisy, timesteps, conditioning, precision, device)
            row = {
                "label": "official",
                "global_step": None,
                "precision": precision,
                "probe": probe,
            }
            photos.append(row)
            l15 = layer15(probe)
            print(
                json.dumps(
                    {
                        "status": "official",
                        "precision": precision,
                        "l15_cos": l15.get("mean_token_cosine"),
                        "l15_sigma": l15.get("ln_sigma_mean"),
                    }
                ),
                flush=True,
            )
        del official
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model = build_trainable_denoiser(config.model, config.curriculum, device)
    set_model_dropout(model, 0.0)
    model.train()
    loss_function = KimodoLoss(model.motion_rep, config.loss)
    for checkpoint, label in zip(args.checkpoints or [], args.labels or []):
        path = Path(checkpoint).expanduser().resolve()
        weights, summary = _jacobian._load_checkpoint_weights(path)
        model.load_state_dict(weights, strict=True)
        for precision in args.precisions:
            probe = _photograph(model, batch, noisy, timesteps, conditioning, precision, device)
            row = {
                "label": label,
                "global_step": summary["global_step"],
                "precision": precision,
                "probe": probe,
            }
            photos.append(row)
            l15 = layer15(probe)
            print(
                json.dumps(
                    {
                        "status": "photographed",
                        "label": label,
                        "precision": precision,
                        "l15_cos": l15.get("mean_token_cosine"),
                        "l15_sigma": l15.get("ln_sigma_mean"),
                    }
                ),
                flush=True,
            )
        if output_dir is not None:
            (output_dir / f"photo-{label}.json").write_text(
                json.dumps(row, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
            )

    official_report = summarize_official_vs_ours(
        [row for row in photos if row.get("precision") == args.precisions[-1]]
    )

    drift_rows = []
    if args.preflip_checkpoint:
        weights, summary = _jacobian._load_checkpoint_weights(Path(args.preflip_checkpoint).expanduser().resolve())
        model.load_state_dict(weights, strict=True)
        sample_count = int(noisy.shape[0])
        chunk = max(1, int(args.chunk_size))
        variants = list(args.variants or ["atan2", "adam", "atan2_wd03", "atan2_lambda1"])
        for variant in variants:
            for precision in args.precisions:
                clip = float(args.clip_norm) if args.clip_norm is not None else None
                if variant == "atan2_noclip":
                    clip = None
                    opt_variant = "atan2"
                else:
                    opt_variant = variant
                deltas = []
                in_proj_deltas = []
                for start in range(0, sample_count, chunk):
                    end = min(start + chunk, sample_count)
                    micro, micro_noisy, micro_t = _slice_batch(batch, noisy, timesteps, start, end)
                    micro_cond = _jacobian._sample_constraints(
                        motion_rep, config, micro, int(args.constraint_step), args.seed + start
                    )
                    ctx = _autocast_context(device, precision) if precision != "fp32" else contextlib.nullcontext()
                    with ctx:
                        result = probe_virtual_steps(
                            model,
                            noisy=micro_noisy,
                            valid_frames=micro["valid_frames"],
                            text_features=micro["text_features"],
                            text_pad_mask=micro["text_pad_mask"],
                            timesteps=micro_t,
                            first_heading_angle=micro["first_heading_angle"],
                            motion_mask=micro_cond.motion_mask,
                            observed_motion=micro_cond.observed_motion,
                            target=micro["clean_motion"],
                            loss_function=loss_function,
                            optimizer_config=config.optimizer,
                            global_step=int(summary["global_step"]),
                            total_steps=int(config.curriculum.phase1_steps + config.curriculum.phase2_steps),
                            variant=opt_variant,
                            n_steps=1,
                            log_every=1,
                            clip_norm=clip,
                        )
                    deltas.append(result.get("attn_cosine_delta"))
                    trace = result.get("trace") or []
                    if len(trace) >= 2:
                        in_proj_deltas.append(
                            float(trace[-1].get("in_proj_rms") or 0) - float(trace[0].get("in_proj_rms") or 0)
                        )
                row = {
                    "variant": variant,
                    "precision": precision,
                    "clip_norm": clip,
                    "preflip_step": summary["global_step"],
                    "deltas": deltas,
                    "in_proj_deltas": in_proj_deltas,
                }
                drift_rows.append(row)
                finite = [float(x) for x in deltas if x is not None]
                mean = sum(finite) / len(finite) if finite else float("nan")
                print(
                    json.dumps(
                        {
                            "status": "drift",
                            "variant": variant,
                            "precision": precision,
                            "clip": clip,
                            "mean_delta": mean,
                            "n": len(finite),
                        }
                    ),
                    flush=True,
                )
                if output_dir is not None:
                    (output_dir / f"drift-{variant}-{precision}.json").write_text(
                        json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )

    drift_report = summarize_multibatch_drift(drift_rows) if drift_rows else {"verdict": "skipped"}
    payload = {
        "photos": [
            {
                "label": row.get("label"),
                "global_step": row.get("global_step"),
                "precision": row.get("precision"),
                "l15": layer15(row.get("probe") or {}),
            }
            for row in photos
        ],
        "official_vs_ours": official_report,
        "drift": drift_report,
    }
    print(json.dumps(payload, indent=2, default=str)[:4000], flush=True)
    if output_dir is not None:
        (output_dir / "rank-00.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        (output_dir / "verdict.json").write_text(
            json.dumps(
                {"official_vs_ours": official_report, "drift": drift_report},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        running = output_dir / "running.json"
        if running.is_file():
            running.unlink()
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--official-dir")
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--preflip-checkpoint")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--constraint-step", type=int, default=750000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precisions", nargs="*", default=["fp32", "bf16"])
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    diagnose(build_parser().parse_args())


if __name__ == "__main__":
    main()
