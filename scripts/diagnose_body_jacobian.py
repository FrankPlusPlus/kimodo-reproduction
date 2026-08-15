#!/usr/bin/env python3
"""Read-only body Jacobian probe across trainer checkpoints.

Loads one frozen batch, then for each checkpoint runs a training-mode forward
and backwards with dropout 0. No optimizer step, no weight write.

Default comparison is kf-smooth 650k / 690k plus the K7 696k takeoff point,
on identical constraints sampled at --constraint-step (default 690000).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from kimodo.model.diffusion import Diffusion
from kimodo.training.body_jacobian_probe import compare_checkpoint_rows, probe_forward_backward
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


def _build_batch(config, args, device):
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
    cpu_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    sampler = ConstraintCurriculumSampler(model.motion_rep, config.curriculum)
    conditioning = sampler.sample(
        batch["clean_motion"], batch["lengths"], args.constraint_step, cpu_generator
    )
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
    return batch, conditioning, noisy, timesteps


def diagnose(args) -> dict:
    config = load_training_config(Path(args.config).expanduser().resolve())
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    batch, conditioning, noisy, timesteps = _build_batch(config, args, device)
    model = build_trainable_denoiser(config.model, config.curriculum, device)
    set_model_dropout(model, 0.0)
    model.train()
    loss_function = KimodoLoss(model.motion_rep, config.loss)
    results = []
    for checkpoint in args.checkpoints:
        path = Path(checkpoint).expanduser().resolve()
        weights, summary = _load_checkpoint_weights(path)
        model.load_state_dict(weights, strict=True)
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
        row = {
            "checkpoint": str(path),
            "global_step": summary["global_step"],
            "probe": probe,
        }
        results.append(row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)
    payload = {
        "seed": int(args.seed),
        "samples": int(args.samples),
        "pair_samples": int(args.pair_samples),
        "constraint_step": int(args.constraint_step),
        "device": str(device),
        "checkpoints": results,
        "comparison_vs_first": compare_checkpoint_rows(results),
    }
    print(json.dumps({"comparison_vs_first": payload["comparison_vs_first"]}, indent=2), flush=True)
    if args.output:
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    return parser


def main() -> None:
    diagnose(build_parser().parse_args())


if __name__ == "__main__":
    main()
