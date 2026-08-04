# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DDP-capable training loop for the paper-aligned Kimodo reconstruction."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from kimodo.model.diffusion import Diffusion

from .checkpoint import (
    CheckpointManager,
    build_training_state,
    capture_rng_state,
    export_inference_bundle,
    load_training_state,
)
from .constraints import ConstraintCurriculumSampler
from .data import MotionManifestDataset, collate_motion_batch
from .ema import ExponentialMovingAverage
from .losses import KimodoLoss
from .modeling import build_trainable_denoiser, set_model_dropout, unwrap_model, validate_model_contract
from .optim import build_optimizer
from .provenance import collect_provenance, save_provenance


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(device_setting: str, mode: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1 if mode == "auto" else mode.lower() in {"1", "true", "yes"}
    if enabled and world_size <= 1:
        raise ValueError("distributed=true requires torchrun environment variables")

    if device_setting == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_setting)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if enabled and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(rank, world_size, local_rank, device)


def seed_everything(seed: int, rank: int) -> None:
    effective = seed + rank
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32" or device.type not in {"cuda", "cpu"}:
        return contextlib.nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 training is only supported on CUDA")
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError("runtime.precision must be fp32, bf16, or fp16")


def _to_device(batch: dict, device: torch.device) -> dict:
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return result


def _drop_text_conditioning(
    text_features: torch.Tensor,
    text_pad_mask: torch.Tensor,
    probability: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dropped = torch.rand((len(text_features),), generator=generator) < probability
    dropped_device = dropped.to(text_features.device)
    text_features = text_features.clone()
    text_pad_mask = text_pad_mask.clone()
    text_features[dropped_device] = 0
    text_pad_mask[dropped_device] = False
    return text_features, text_pad_mask, dropped_device


def _reduce_scalar(value: torch.Tensor, context: DistributedContext) -> float:
    detached = value.detach().float()
    if context.world_size > 1:
        dist.all_reduce(detached, op=dist.ReduceOp.SUM)
        detached /= context.world_size
    return float(detached.item())


class JsonlLogger:
    def __init__(self, path: Path, enabled: bool) -> None:
        self.path = path
        self.enabled = enabled
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        if not self.enabled:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)


def build_training_dataset(config, motion_rep):
    """Build the dataset while preserving strict paper-data parity policy."""
    return MotionManifestDataset(
        config.data.manifest,
        config.data.split,
        motion_rep,
        max_seconds=config.data.max_seconds,
        min_frames=config.data.min_frames,
        seed=config.runtime.seed,
        require_cached_text=config.data.require_cached_text,
        require_paper_data_parity=config.data.require_paper_data_parity,
        normalize=True,
        augment=True,
    )


def validate_paper_runtime_scale(config, context: DistributedContext) -> None:
    """Enforce the paper's disclosed 16-rank/global-batch-2048 training scale."""
    if not config.paper_method_strict or not config.runtime.enforce_paper_scale:
        return
    effective_global_batch = (
        context.world_size
        * config.runtime.batch_size
        * config.runtime.gradient_accumulation_steps
    )
    mismatches = []
    if context.world_size != 16:
        mismatches.append(f"world_size={context.world_size} (paper requires 16)")
    if effective_global_batch != 2048:
        mismatches.append(
            f"effective_global_batch={effective_global_batch} (paper requires 2048)"
        )
    if mismatches:
        raise RuntimeError(
            "paper_method_strict rejects runtime-scale deviations: " + "; ".join(mismatches)
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_output_lineage(config, output_dir: Path) -> tuple[str, str] | None:
    resume_value = config.runtime.resume
    nonempty = output_dir.exists() and (
        not output_dir.is_dir() or next(output_dir.iterdir(), None) is not None
    )
    if not resume_value:
        if nonempty:
            return (
                "file",
                f"fresh training output_dir is not empty: {output_dir}; choose a new directory",
            )
        return None

    resume = Path(resume_value).expanduser().resolve()
    if not resume.is_file():
        return ("value", f"resume checkpoint does not exist: {resume}")
    if config.runtime.resume_mode == "in_place":
        expected_parent = (output_dir / "checkpoints").resolve()
        if resume.parent != expected_parent:
            return (
                "value",
                (
                    "in-place resume checkpoint must belong to output_dir/checkpoints; "
                    "use runtime.resume_mode=fork with a new empty output_dir for an explicit fork"
                ),
            )
        required = (output_dir / "config.resolved.yaml", output_dir / "provenance.json")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            return ("value", f"in-place resume run metadata is incomplete: {missing}")
    elif nonempty:
        return ("file", f"fork resume requires a new empty output_dir: {output_dir}")
    return None


class KimodoTrainer:
    def __init__(self, config, project_root: str | Path, step_observer=None) -> None:
        self.config = config
        self.project_root = Path(project_root)
        # Optional read-only instrumentation hook used by the standalone
        # benchmark harness. Production training passes no observer, so the
        # mathematical path and checkpoint behavior remain unchanged.
        self.step_observer = step_observer
        self.context = initialize_distributed(config.runtime.device, config.runtime.distributed)
        validate_paper_runtime_scale(config, self.context)
        seed_everything(config.runtime.seed, self.context.rank)
        self.output_dir = Path(config.runtime.output_dir).expanduser().resolve()
        output_error = (
            _validate_output_lineage(config, self.output_dir)
            if self.context.is_main
            else None
        )
        if self.context.world_size > 1:
            payload = [output_error]
            dist.broadcast_object_list(payload, src=0)
            output_error = payload[0]
        if output_error is not None:
            error_type, message = output_error
            if error_type == "file":
                raise FileExistsError(message)
            raise ValueError(message)
        # Validate provenance before allocating the 283M-parameter model and
        # optimizer. Rank zero performs the filesystem work once; peers wait
        # for the compact result.
        if self.context.is_main:
            provenance = collect_provenance(config, self.project_root, context=self.context)
        else:
            provenance = None
        if self.context.world_size > 1:
            payload = [provenance]
            dist.broadcast_object_list(payload, src=0)
            provenance = payload[0]
        if config.runtime.resume and config.runtime.resume_mode == "fork":
            provenance = copy.deepcopy(provenance)
            resume_path = Path(config.runtime.resume).expanduser().resolve()
            provenance["resume_lineage"] = {
                "mode": "fork",
                "parent_checkpoint": str(resume_path),
                "parent_checkpoint_sha256": _sha256_file(resume_path),
            }
        self.provenance = provenance
        self.model = build_trainable_denoiser(config.model, config.curriculum, self.context.device)
        validate_model_contract(self.model, config.model)
        self.model.train()
        if self.context.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.context.local_rank] if self.context.device.type == "cuda" else None,
                broadcast_buffers=False,
            )
        bare = unwrap_model(self.model)
        self.motion_rep = bare.motion_rep
        self.diffusion = Diffusion(config.model.num_diffusion_steps).to(self.context.device)
        self.loss = KimodoLoss(self.motion_rep, config.loss)
        self.constraint_sampler = ConstraintCurriculumSampler(self.motion_rep, config.curriculum)
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.ema = ExponentialMovingAverage(bare, config.ema.decay) if config.ema.enabled else None
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.runtime.precision == "fp16")
        protected_steps = {config.curriculum.phase1_steps, config.total_steps}
        if config.runtime.milestone_every:
            protected_steps.update(
                range(config.runtime.milestone_every, config.total_steps + 1, config.runtime.milestone_every)
            )
        self.checkpoints = CheckpointManager(
            self.output_dir,
            config.runtime.keep_last_checkpoints,
            protected_steps=protected_steps,
        )
        self.logger = JsonlLogger(self.output_dir / "train.jsonl", self.context.is_main)
        self.global_step = int(config.runtime.initial_global_step)
        self.epoch = 0
        self.batch_in_epoch = 0
        self.micro_index = self.global_step * config.runtime.gradient_accumulation_steps
        self._last_dropout = None
        self._last_saved_position: tuple[int, int] | None = None

        # The denoiser and loss use the device-resident representation above,
        # while DataLoader workers construct motion features on CPU. Sharing
        # the same instance leaves skeleton index buffers on CUDA and makes
        # CPU forward kinematics fail before the first training step.
        dataset_motion_rep = copy.deepcopy(self.motion_rep)
        for value in vars(dataset_motion_rep).values():
            if isinstance(value, torch.nn.Module):
                value.cpu()
        dataset = build_training_dataset(config, dataset_motion_rep)
        self.dataset = dataset
        # Use an epoch-addressable sampler even on one process. RandomSampler
        # creates its permutation from ambient RNG state when iteration starts,
        # which makes an exact mid-epoch resume impossible.
        self.distributed_sampler = DistributedSampler(
            dataset,
            num_replicas=self.context.world_size,
            rank=self.context.rank,
            shuffle=True,
            seed=config.runtime.seed,
            drop_last=True,
        )
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": config.runtime.batch_size,
            "shuffle": False,
            "sampler": self.distributed_sampler,
            "num_workers": config.data.num_workers,
            "pin_memory": config.data.pin_memory,
            "drop_last": True,
            "collate_fn": collate_motion_batch,
            # Keep DataLoader worker/base-seed draws isolated from the global
            # model RNG restored by checkpoints.
            "generator": torch.Generator(device="cpu").manual_seed(
                config.runtime.seed + self.context.rank
            ),
        }
        if config.data.num_workers > 0:
            loader_kwargs.update(
                prefetch_factor=config.data.prefetch_factor,
                persistent_workers=config.data.persistent_workers,
            )
        self.loader = DataLoader(**loader_kwargs)
        if len(self.loader) == 0:
            raise ValueError("Training DataLoader is empty; lower batch_size or add data")

        if config.runtime.resume:
            state = load_training_state(
                config.runtime.resume,
                model=self.model,
                optimizer=self.optimizer,
                ema=self.ema,
                scaler=self.scaler,
                expected_provenance=self.provenance,
                current_config=config.to_dict(),
                rank=self.context.rank,
                world_size=self.context.world_size,
            )
            self.global_step = int(state["global_step"])
            self.epoch = int(state["epoch"])
            self.batch_in_epoch = int(state["batch_in_epoch"])
            self.micro_index = int(state["micro_index"])
            self._last_saved_position = (self.global_step, self.micro_index)

        if self.context.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            from .config import save_resolved_config

            save_resolved_config(config, self.output_dir / "config.resolved.yaml")
            save_provenance(self.provenance, self.output_dir / "provenance.json")

    def _observed_section(self, name: str):
        """Return an optional benchmark-only profiling range."""
        if self.step_observer is None or not hasattr(self.step_observer, "section"):
            return contextlib.nullcontext()
        return self.step_observer.section(name)

    def _phase_dropout(self) -> float:
        if self.global_step < self.config.curriculum.phase1_steps:
            return self.config.curriculum.phase1_dropout
        return self.config.curriculum.phase2_dropout

    def _apply_phase(self) -> str:
        probability = self._phase_dropout()
        if probability != self._last_dropout:
            set_model_dropout(self.model, probability)
            self._last_dropout = probability
        return "phase1" if self.global_step < self.config.curriculum.phase1_steps else "phase2"

    def _step_generators(self, micro_index: int):
        unique = (
            self.config.runtime.seed
            + self.context.rank * 1_000_000_000
            + micro_index
        )
        cpu = torch.Generator(device="cpu").manual_seed(unique)
        noise_device = self.context.device if self.context.device.type == "cuda" else torch.device("cpu")
        noise = torch.Generator(device=noise_device).manual_seed(unique + 17)
        return cpu, noise

    def _save(self, *, diagnostic_reason: str | None = None) -> Path | None:
        position = (self.global_step, self.micro_index)
        if diagnostic_reason is None and self._last_saved_position == position:
            return self.checkpoints.directory / f"step-{self.global_step:09d}.pt"
        local_rng = capture_rng_state()
        if self.context.world_size > 1:
            rng_by_rank = [None] * self.context.world_size if self.context.is_main else None
            dist.gather_object(local_rng, rng_by_rank, dst=0)
        else:
            rng_by_rank = [local_rng]
        if not self.context.is_main:
            if diagnostic_reason is None:
                self._last_saved_position = position
            return None
        state = build_training_state(
            model=self.model,
            optimizer=self.optimizer,
            ema=self.ema,
            scaler=self.scaler,
            global_step=self.global_step,
            epoch=self.epoch,
            batch_in_epoch=self.batch_in_epoch,
            micro_index=self.micro_index,
            config=self.config.to_dict(),
            provenance=self.provenance,
            rng_by_rank=rng_by_rank,
            resume_exact=diagnostic_reason is None,
            diagnostic_reason=diagnostic_reason,
        )
        if diagnostic_reason is not None:
            return self.checkpoints.save_diagnostic(state, diagnostic_reason)
        saved = self.checkpoints.save(state)
        self._last_saved_position = position
        return saved

    def train(self) -> None:
        accumulation = self.config.runtime.gradient_accumulation_steps
        self.optimizer.zero_grad(set_to_none=True)
        skip_batches = self.batch_in_epoch
        if skip_batches >= len(self.loader):
            completed_epochs, skip_batches = divmod(skip_batches, len(self.loader))
            self.epoch += completed_epochs
            self.batch_in_epoch = skip_batches
        started = time.time()
        accumulated_valid_frames = 0
        accumulated_loss_sums: dict[str, torch.Tensor] = {}
        curriculum_counts = {
            "samples": 0.0,
            "text_dropped": 0.0,
            "constrained": 0.0,
            "two_patterns": 0.0,
            "joint": 0.0,
            "constraint_only": 0.0,
            "text_only": 0.0,
            "unconditional": 0.0,
            **{f"pattern/{name}": 0.0 for name in self.constraint_sampler.PATTERNS},
            **{f"data_source/{name}": 0.0 for name in self.dataset.mixture_sources},
        }

        while self.global_step < self.config.total_steps:
            self.dataset.set_epoch(self.epoch)
            self.distributed_sampler.set_epoch(self.epoch)
            for batch_index, batch in enumerate(self.loader):
                if batch_index < skip_batches:
                    continue
                skip_batches = 0
                phase = self._apply_phase()
                with self._observed_section("h2d"):
                    batch = _to_device(batch, self.context.device)
                if batch["text_features"] is None or batch["text_pad_mask"] is None:
                    raise RuntimeError(
                        "This production trainer requires cached LLM2Vec embeddings. "
                        "Generate them with the documented cache workflow."
                    )
                if batch["text_features"].shape[-1] != self.config.model.llm_dim:
                    raise ValueError("Cached text embedding width does not match model.llm_dim")

                with self._observed_section("conditioning"):
                    cpu_generator, noise_generator = self._step_generators(self.micro_index)
                    text_features, text_mask, text_dropped = _drop_text_conditioning(
                        batch["text_features"],
                        batch["text_pad_mask"],
                        self.config.curriculum.text_dropout_probability,
                        cpu_generator,
                    )
                    conditioning = self.constraint_sampler.sample(
                        batch["clean_motion"], batch["lengths"], self.global_step, cpu_generator
                    )
                    dropped_values = text_dropped.detach().cpu().tolist()
                    for dropped_value, patterns in zip(
                        dropped_values, conditioning.pattern_names, strict=True
                    ):
                        dropped = bool(dropped_value)
                        constrained = bool(patterns)
                        curriculum_counts["samples"] += 1
                        curriculum_counts["text_dropped"] += float(dropped)
                        curriculum_counts["constrained"] += float(constrained)
                        curriculum_counts["two_patterns"] += float(len(patterns) == 2)
                        branch = (
                            "unconditional"
                            if dropped and not constrained
                            else "constraint_only"
                            if dropped
                            else "joint"
                            if constrained
                            else "text_only"
                        )
                        curriculum_counts[branch] += 1
                        for pattern in patterns:
                            curriculum_counts[f"pattern/{pattern}"] += 1
                    for source in batch["mixture_sources"]:
                        curriculum_counts[f"data_source/{source}"] += 1
                with self._observed_section("noise_and_diffusion"):
                    noise = torch.randn(
                        batch["clean_motion"].shape,
                        dtype=batch["clean_motion"].dtype,
                        device=self.context.device,
                        generator=noise_generator,
                    )
                    timesteps = torch.randint(
                        self.config.model.num_diffusion_steps,
                        (len(batch["clean_motion"]),),
                        device=self.context.device,
                        generator=noise_generator,
                    )
                    noisy_motion = self.diffusion.q_sample(batch["clean_motion"], timesteps, noise)

                sync_now = (self.micro_index + 1) % accumulation == 0
                sync_context = contextlib.nullcontext()
                if isinstance(self.model, DistributedDataParallel) and not sync_now:
                    sync_context = self.model.no_sync()
                with sync_context, _autocast_context(self.context.device, self.config.runtime.precision):
                    with self._observed_section("model_forward"):
                        prediction = self.model(
                            noisy_motion,
                            batch["valid_frames"],
                            text_features,
                            text_mask,
                            timesteps,
                            first_heading_angle=batch["first_heading_angle"],
                            motion_mask=conditioning.motion_mask,
                            observed_motion=conditioning.observed_motion,
                        )
                    with self._observed_section("seven_term_loss"):
                        losses = self.loss(prediction, batch["clean_motion"], batch["valid_frames"])
                with self._observed_section("finite_check"):
                    finite_flag = torch.isfinite(losses["total"]).to(dtype=torch.float32)
                    if self.context.world_size > 1:
                        dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                    if not bool(finite_flag.item()):
                        self._save(diagnostic_reason="nonfinite")
                        raise FloatingPointError(f"Non-finite loss at global_step={self.global_step}")
                # Backpropagate valid-frame numerators. At the optimizer
                # boundary gradients are normalized once by the global count,
                # making accumulation/DDP equivalent to one global batch.
                with self._observed_section("backward_and_ddp"):
                    self.scaler.scale(losses.frame_sums["total"]).backward()
                accumulated_valid_frames += int(losses.valid_frame_count.detach().item())
                for name, value in losses.frame_sums.items():
                    detached = value.detach().float()
                    accumulated_loss_sums[name] = accumulated_loss_sums.get(
                        name, torch.zeros_like(detached)
                    ) + detached
                self.micro_index += 1
                self.batch_in_epoch = batch_index + 1
                if not sync_now:
                    continue

                with self._observed_section("gradient_normalize_and_clip"):
                    self.scaler.unscale_(self.optimizer)
                    global_valid_frames = torch.tensor(
                        accumulated_valid_frames,
                        device=self.context.device,
                        dtype=torch.int64,
                    )
                    if self.context.world_size > 1:
                        dist.all_reduce(global_valid_frames, op=dist.ReduceOp.SUM)
                    gradient_scale = self.context.world_size / float(global_valid_frames.item())
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(gradient_scale)
                    if self.config.optimizer.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.optimizer.gradient_clip_norm
                        )
                with self._observed_section("optimizer"):
                    previous_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                step_skipped = self.scaler.is_enabled() and self.scaler.get_scale() < previous_scale
                if step_skipped:
                    accumulated_valid_frames = 0
                    accumulated_loss_sums.clear()
                    for name in curriculum_counts:
                        curriculum_counts[name] = 0.0
                    continue
                self.global_step += 1

                if self.ema is not None and self.global_step % self.config.ema.update_every == 0:
                    with self._observed_section("ema"):
                        self.ema.update(unwrap_model(self.model))

                if self.step_observer is not None:
                    self.step_observer.on_optimizer_step_end(self)

                if self.global_step % self.config.runtime.log_every == 0:
                    logged_sums = {
                        name: value.clone().to(self.context.device)
                        for name, value in accumulated_loss_sums.items()
                    }
                    if self.context.world_size > 1:
                        for value in logged_sums.values():
                            dist.all_reduce(value, op=dist.ReduceOp.SUM)
                    count_names = list(curriculum_counts)
                    count_values = torch.tensor(
                        [curriculum_counts[name] for name in count_names],
                        device=self.context.device,
                        dtype=torch.float32,
                    )
                    if self.context.world_size > 1:
                        dist.all_reduce(count_values, op=dist.ReduceOp.SUM)
                    global_counts = dict(zip(count_names, count_values.tolist()))
                    sample_count = max(1.0, global_counts["samples"])
                    record = {
                        "global_step": self.global_step,
                        "phase": phase,
                        "epoch": self.epoch,
                        "elapsed_seconds": time.time() - started,
                        "text_dropout_fraction": global_counts["text_dropped"] / sample_count,
                        "constraint_fraction": global_counts["constrained"] / sample_count,
                        "two_pattern_fraction": global_counts["two_patterns"] / sample_count,
                        "maximum_sparse_keyframes": conditioning.maximum_sparse_keyframes,
                    }
                    for branch in ("joint", "constraint_only", "text_only", "unconditional"):
                        record[f"conditioning/{branch}_fraction"] = global_counts[branch] / sample_count
                    for pattern in self.constraint_sampler.PATTERNS:
                        record[f"conditioning/{pattern}_per_sample"] = (
                            global_counts[f"pattern/{pattern}"] / sample_count
                        )
                    for source in self.dataset.mixture_sources:
                        record[f"data/{source}_fraction"] = (
                            global_counts[f"data_source/{source}"] / sample_count
                        )
                    denominator = float(global_valid_frames.item())
                    record.update(
                        {f"loss/{name}": float(value.item() / denominator) for name, value in logged_sums.items()}
                    )
                    self.logger.write(record)

                accumulated_valid_frames = 0
                accumulated_loss_sums.clear()
                for name in curriculum_counts:
                    curriculum_counts[name] = 0.0

                if self.global_step % self.config.runtime.checkpoint_every == 0 or (
                    self.config.runtime.milestone_every
                    and self.global_step % self.config.runtime.milestone_every == 0
                ) or self.global_step == self.config.curriculum.phase1_steps:
                    self._save()
                if self.global_step >= self.config.total_steps:
                    break

            else:
                self.epoch += 1
                self.batch_in_epoch = 0
                continue
            break

        if self.context.world_size > 1:
            dist.barrier()
        self._save()
        if self.context.is_main:
            export_inference_bundle(
                self.model,
                self.ema,
                self.output_dir,
                self.global_step,
                self.config,
            )
        if self.context.world_size > 1:
            dist.barrier()
