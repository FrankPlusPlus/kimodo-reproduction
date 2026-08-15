#!/usr/bin/env python3
"""Read-only discriminator for the 6D Gram-Schmidt / FK gradient hypothesis.

Loads a fixed batch once, then for each trainer checkpoint:
  1. Forward the denoiser (no optimizer).
  2. Measure unnormalized 6D column-norm and cross-product tails.
  3. Backward each loss term onto the prediction (not the Transformer).

If FK's dL/d(6D) and the cross-product tail worsen while direct-term
dL/d(pred) stay flat, the singularity is in FK/Gram-Schmidt. If every
term's dL/d(pred) is stable, the training gnorm climb is the body Jacobian.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from kimodo.model.diffusion import Diffusion
from kimodo.training.config import load_training_config
from kimodo.training.constraints import ConstraintCurriculumSampler
from kimodo.training.data import MotionManifestDataset, collate_motion_batch
from kimodo.training.losses import KimodoLoss
from kimodo.training.modeling import build_trainable_denoiser, set_model_dropout


def _load_checkpoint_weights(path: Path, use_ema: bool) -> tuple[dict[str, torch.Tensor], dict]:
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    weights = (state.get("ema") or {}).get("shadow") if use_ema else state.get("model")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Checkpoint has no model weights: {path}")
    return weights, {"global_step": int(state["global_step"])}


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


DIRECT_TERMS = (
    "root_position",
    "root_heading",
    "joint_position",
    "joint_rotation",
    "joint_velocity",
    "foot_contact",
)
ALL_TERMS = DIRECT_TERMS + ("forward_kinematics",)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {"min": float("nan"), "p0.001": float("nan"), "p0.01": float("nan"), "median": float("nan")}
    qs = torch.quantile(flat, torch.tensor([0.001, 0.01, 0.5], device=flat.device))
    return {
        "min": float(flat.min()),
        "p0.001": float(qs[0]),
        "p0.01": float(qs[1]),
        "median": float(qs[2]),
    }


def _sixd_tails(prediction: torch.Tensor, motion_rep, valid_frames: torch.Tensor) -> dict[str, dict[str, float]]:
    raw = motion_rep.unnormalize(prediction.detach().float())
    rot = raw[..., motion_rep.slice_dict["global_rot_data"]]
    rot = rot.reshape(*rot.shape[:2], motion_rep.skeleton.nbjoints, 6)
    x_raw = rot[..., 0:3]
    y_raw = rot[..., 3:6]
    x_norm = x_raw.norm(dim=-1)
    x_hat = x_raw / x_norm.unsqueeze(-1).clamp_min(1e-8)
    cross_norm = torch.cross(x_hat, y_raw, dim=-1).norm(dim=-1)
    valid = valid_frames.unsqueeze(-1).expand_as(x_norm)
    return {
        "x_raw_norm": _quantiles(x_norm[valid]),
        "cross_norm": _quantiles(cross_norm[valid]),
    }


def _term_prediction_grads(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_frames: torch.Tensor,
    loss_function: KimodoLoss,
    motion_rep,
) -> dict[str, dict[str, float]]:
    pred = prediction.detach().float().requires_grad_(True)
    losses = loss_function(pred, target.float(), valid_frames)
    denom = losses.valid_frame_count.to(dtype=pred.dtype).clamp_min(1)
    rot_slice = motion_rep.slice_dict["global_rot_data"]
    root_dim = motion_rep.global_root_dim
    reports: dict[str, dict[str, float]] = {}

    def _record(name: str, tensor: torch.Tensor) -> None:
        if pred.grad is not None:
            pred.grad.zero_()
        tensor.backward(retain_graph=True)
        grad = pred.grad.detach()
        reports[name] = {
            "pred_grad_norm": float(grad.norm()),
            "sixd_grad_norm": float(grad[..., rot_slice].norm()),
            "root_grad_norm": float(grad[..., :root_dim].norm()),
            "body_grad_norm": float(grad[..., root_dim:].norm()),
        }

    for name in ALL_TERMS:
        _record(name, losses.frame_sums[name] / denom)
    direct = sum(
        (float(getattr(loss_function.config, name)) * losses.frame_sums[name] for name in DIRECT_TERMS),
        start=prediction.new_zeros(()),
    )
    _record("direct_without_fk", direct / denom)
    _record("total", losses.frame_sums["total"] / denom)
    reports["loss_means"] = {name: float(losses.means[name].detach()) for name in (*ALL_TERMS, "total")}
    return reports


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
    conditioning = sampler.sample(batch["clean_motion"], batch["lengths"], args.constraint_step, cpu_generator)
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
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    batch, conditioning, noisy, timesteps = _build_batch(config, args, device)
    model = build_trainable_denoiser(config.model, config.curriculum, device)
    set_model_dropout(model, 0.0)
    model.eval()
    loss_function = KimodoLoss(model.motion_rep, config.loss)
    results = []
    for checkpoint in args.checkpoints:
        path = Path(checkpoint).expanduser().resolve()
        weights, summary = _load_checkpoint_weights(path, use_ema=False)
        model.load_state_dict(weights, strict=True)
        with torch.inference_mode():
            prediction = model(
                noisy,
                batch["valid_frames"],
                batch["text_features"],
                batch["text_pad_mask"],
                timesteps,
                first_heading_angle=batch["first_heading_angle"],
                motion_mask=conditioning.motion_mask,
                observed_motion=conditioning.observed_motion,
            )
        prediction = prediction.detach().clone().float()
        target = batch["clean_motion"].float()
        row = {
            "checkpoint": str(path),
            "global_step": summary["global_step"],
            "sixd_tails": _sixd_tails(prediction, model.motion_rep, batch["valid_frames"]),
            "term_grads": _term_prediction_grads(
                prediction, target, batch["valid_frames"], loss_function, model.motion_rep
            ),
        }
        results.append(row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)
    payload = {
        "seed": int(args.seed),
        "samples": int(args.samples),
        "constraint_step": int(args.constraint_step),
        "device": str(device),
        "checkpoints": results,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--constraint-step", type=int, default=696000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    return parser


def main() -> None:
    diagnose(build_parser().parse_args())


if __name__ == "__main__":
    main()
