# SPDX-License-Identifier: Apache-2.0
"""Evaluate deterministic held-out denoising losses for a trainer checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .checkpoint import atomic_text_write
from .config import load_training_config
from .constraints import ConstraintCurriculumSampler
from .data import MotionManifestDataset, collate_motion_batch
from .engine import Diffusion, _autocast_context, _to_device
from .losses import KimodoLoss
from .modeling import build_trainable_denoiser, set_model_dropout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint_weights(path: Path, use_ema: bool) -> tuple[dict[str, torch.Tensor], dict]:
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if use_ema:
        ema = state.get("ema")
        if not isinstance(ema, dict) or not isinstance(ema.get("shadow"), dict):
            raise ValueError("Checkpoint does not contain EMA shadow weights")
        weights = ema["shadow"]
    else:
        weights = state.get("model")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Checkpoint does not contain model weights")
    summary = {
        "global_step": int(state["global_step"]),
        "micro_index": int(state["micro_index"]),
        "ema_updates": int((state.get("ema") or {}).get("num_updates", 0)),
        "resume_exact": bool(state.get("resume_exact", True)),
    }
    return weights, summary


def _parameter_delta(
    current: dict[str, torch.Tensor], baseline: dict[str, torch.Tensor]
) -> dict[str, float | int]:
    if current.keys() != baseline.keys():
        raise ValueError("Current and baseline state dict keys differ")
    delta_sq = 0.0
    baseline_sq = 0.0
    changed = 0
    maximum = 0.0
    for name, value in current.items():
        left = value.detach().float().cpu()
        right = baseline[name].detach().float().cpu()
        difference = left - right
        value = float(difference.square().sum())
        delta_sq += value
        baseline_sq += float(right.square().sum())
        maximum = max(maximum, float(difference.abs().max()))
        changed += int(value > 0.0)
    return {
        "changed_tensors": changed,
        "tensor_count": len(current),
        "relative_l2": (delta_sq / max(baseline_sq, 1e-30)) ** 0.5,
        "maximum_absolute_delta": maximum,
    }


def evaluate(args) -> dict:
    config_path = Path(args.config).expanduser().resolve()
    config = load_training_config(config_path)
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else Path(config.data.manifest)
    config.data.manifest = str(manifest)
    config.data.split = args.split
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_trainable_denoiser(config.model, config.curriculum, device)
    baseline = (
        {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if config.model.checkpoint_dir
        else None
    )
    checkpoint_summary = None
    parameter_delta = None
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        weights, checkpoint_summary = _load_checkpoint_weights(checkpoint_path, args.use_ema)
        model.load_state_dict(weights, strict=True)
        if baseline is not None:
            parameter_delta = _parameter_delta(model.state_dict(), baseline)
    set_model_dropout(model, 0.0)
    model.eval()

    motion_rep = copy.deepcopy(model.motion_rep)
    for value in vars(motion_rep).values():
        if isinstance(value, torch.nn.Module):
            value.cpu()
    dataset = MotionManifestDataset(
        manifest,
        args.split,
        motion_rep,
        max_seconds=config.data.max_seconds,
        min_frames=config.data.min_frames,
        seed=args.seed,
        require_cached_text=True,
        require_paper_data_parity=False,
        normalize=True,
        augment=False,
    )
    sample_count = min(int(args.samples), len(dataset))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[
        :sample_count
    ].tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=collate_motion_batch,
    )
    diffusion = Diffusion(config.model.num_diffusion_steps).to(device)
    loss_function = KimodoLoss(model.motion_rep, config.loss)
    constraint_sampler = ConstraintCurriculumSampler(model.motion_rep, config.curriculum)
    evaluation_step = max(0, config.curriculum.phase1_steps + config.curriculum.phase2_steps - 1)
    sums: dict[str, float] = defaultdict(float)
    denominators: dict[str, int] = defaultdict(int)
    pattern_samples = CounterLike()

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = _to_device(batch, device)
            cpu_generator = torch.Generator(device="cpu").manual_seed(args.seed + batch_index * 2 + 1)
            noise_generator = torch.Generator(device=device).manual_seed(args.seed + batch_index * 2 + 2)
            conditioning = constraint_sampler.sample(
                batch["clean_motion"], batch["lengths"], evaluation_step, cpu_generator
            )
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
            with _autocast_context(device, config.runtime.precision):
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
                losses = loss_function(
                    prediction, batch["clean_motion"], batch["valid_frames"]
                )
            denominator = int(losses.valid_frame_count.item())
            for name, value in losses.frame_sums.items():
                sums[f"all/{name}"] += float(value.float().item())
            denominators["all"] += denominator
            for pattern in constraint_sampler.PATTERNS:
                selected = [
                    index
                    for index, names in enumerate(conditioning.pattern_names)
                    if pattern in names
                ]
                pattern_samples.add(pattern, len(selected))
                if not selected:
                    continue
                chosen = torch.tensor(selected, device=device)
                with _autocast_context(device, config.runtime.precision):
                    family_losses = loss_function(
                        prediction.index_select(0, chosen),
                        batch["clean_motion"].index_select(0, chosen),
                        batch["valid_frames"].index_select(0, chosen),
                    )
                family_denominator = int(family_losses.valid_frame_count.item())
                for name, value in family_losses.frame_sums.items():
                    sums[f"{pattern}/{name}"] += float(value.float().item())
                denominators[pattern] += family_denominator

    metrics = {}
    for key, value in sorted(sums.items()):
        group, name = key.split("/", 1)
        metrics[f"{group}/loss_{name}"] = value / denominators[group]
    result = {
        "event": "kimodo_fixed_denoising_validation",
        "config": str(config_path),
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "split": args.split,
        "seed": int(args.seed),
        "samples": sample_count,
        "batch_size": int(args.batch_size),
        "evaluation_global_step": evaluation_step,
        "maximum_sparse_keyframes": constraint_sampler.maximum_sparse_keyframes(evaluation_step),
        "weights": "ema" if args.checkpoint and args.use_ema else "online_or_official",
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path else None,
        "checkpoint_state": checkpoint_summary,
        "parameter_delta_from_official_initialization": parameter_delta,
        "pattern_samples": dict(sorted(pattern_samples.values.items())),
        "metrics": metrics,
    }
    if args.output:
        atomic_text_write(json.dumps(result, indent=2, sort_keys=True) + "\n", args.output)
    return result


class CounterLike:
    def __init__(self) -> None:
        self.values: dict[str, int] = defaultdict(int)

    def add(self, key: str, value: int) -> None:
        self.values[key] += int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved training YAML")
    parser.add_argument("--checkpoint", help="Full trainer checkpoint; omit for initialization baseline")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--manifest", help="Optional held-out manifest override")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    return parser


def main() -> None:
    print(json.dumps(evaluate(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
