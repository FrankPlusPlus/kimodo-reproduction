# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DDP-capable training loop for the paper-aligned Kimodo reconstruction."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from kimodo.model.diffusion import Diffusion
from kimodo.monitoring import WandbMonitor

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
from .optim import build_optimizer, scheduled_learning_rate
from .provenance import collect_provenance, save_provenance
from .run_lock import ExclusiveRunLock


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
    """Rank-0 metrics logger.

    File / stdout / W&B publication runs on a background thread so the training
    step is not blocked on JuiceFS append latency. ``close()`` drains the queue.
    """

    def __init__(
        self,
        path: Path,
        enabled: bool,
        monitor: WandbMonitor | None = None,
        *,
        async_write: bool = True,
    ) -> None:
        self.path = path
        self.enabled = enabled
        self.monitor = monitor
        self._queue: queue.Queue[dict | None] | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            if async_write:
                self._queue = queue.Queue(maxsize=256)
                self._thread = threading.Thread(
                    target=self._worker,
                    name="kimodo-jsonl-logger",
                    daemon=True,
                )
                self._thread.start()

    def _write_sync(self, record: dict) -> None:
        payload = json.dumps(record, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        print(payload, flush=True)
        if self.monitor is not None:
            self.monitor.log(record, step=record.get("global_step"))

    def _worker(self) -> None:
        assert self._queue is not None
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._write_sync(item)
            except BaseException as error:  # noqa: BLE001 - surface on next write/close
                self._error = error
            finally:
                self._queue.task_done()

    def write(self, record: dict) -> None:
        if not self.enabled:
            return
        if self._error is not None:
            raise RuntimeError("async metrics logger failed") from self._error
        if self._queue is None:
            self._write_sync(record)
            return
        self._queue.put(record)

    def close(self) -> None:
        if self._queue is None or self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout=120)
        self._queue = None
        self._thread = None
        if self._error is not None:
            raise RuntimeError("async metrics logger failed") from self._error


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
        feature_cache_dir=config.data.feature_cache_dir,
        stats_path=config.model.stats_path,
        normalize=True,
        augment=True,
    )


def validate_paper_runtime_scale(config, context: DistributedContext) -> None:
    """Enforce explicit deployment and strict-paper runtime scale contracts."""
    effective_global_batch = (
        context.world_size
        * config.runtime.batch_size
        * config.runtime.gradient_accumulation_steps
    )
    deployment_mismatches = []
    if (
        config.runtime.expected_world_size is not None
        and context.world_size != config.runtime.expected_world_size
    ):
        deployment_mismatches.append(
            f"world_size={context.world_size} (config requires {config.runtime.expected_world_size})"
        )
    if (
        config.runtime.expected_global_batch is not None
        and effective_global_batch != config.runtime.expected_global_batch
    ):
        deployment_mismatches.append(
            "effective_global_batch="
            f"{effective_global_batch} (config requires {config.runtime.expected_global_batch})"
        )
    if deployment_mismatches:
        raise RuntimeError(
            "runtime deployment contract rejected this launch: "
            + "; ".join(deployment_mismatches)
        )
    if not config.paper_method_strict or not config.runtime.enforce_paper_scale:
        return
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


def _fork_parent_fingerprint(path: Path) -> str:
    """Identity for fork lineage without reading the 4.3G checkpoint."""
    stat = path.stat()
    payload = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        boot_t0 = time.perf_counter()
        self.context = initialize_distributed(config.runtime.device, config.runtime.distributed)
        self._boot_log(
            f"distributed ready rank={self.context.rank}/{self.context.world_size} "
            f"device={self.context.device}"
        )
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
        phase_t0 = time.perf_counter()
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
                "parent_checkpoint_sha256": _fork_parent_fingerprint(resume_path),
            }
        self.provenance = provenance
        self._boot_log(f"provenance ready in {time.perf_counter() - phase_t0:.1f}s")
        phase_t0 = time.perf_counter()
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
        self.global_step = int(config.runtime.initial_global_step)
        self.epoch = 0
        self.batch_in_epoch = 0
        self.micro_index = self.global_step * config.runtime.gradient_accumulation_steps
        self._last_dropout = None
        self._last_saved_position: tuple[int, int] | None = None
        self._boot_log(f"model/optimizer ready in {time.perf_counter() - phase_t0:.1f}s")

        # The denoiser and loss use the device-resident representation above,
        # while DataLoader workers construct motion features on CPU. Sharing
        # the same instance leaves skeleton index buffers on CUDA and makes
        # CPU forward kinematics fail before the first training step.
        phase_t0 = time.perf_counter()
        dataset_motion_rep = copy.deepcopy(self.motion_rep)
        for value in vars(dataset_motion_rep).values():
            if isinstance(value, torch.nn.Module):
                value.cpu()
        self._boot_log(
            f"loading manifest dataset ({config.data.manifest}) "
            f"workers={config.data.num_workers} persistent={config.data.persistent_workers} "
            f"prefetch={config.data.prefetch_factor}"
        )
        local_manifest = os.environ.get("KIMODO_LOCAL_MANIFEST_READ_PATH")
        skip_path_stat = os.environ.get("KIMODO_SKIP_MANIFEST_PATH_STAT", "0")
        index_mode = os.environ.get("KIMODO_FEATURE_CACHE_INDEX_MODE", "strict")
        self._boot_log(
            f"dataset load policy skip_path_stat={skip_path_stat} "
            f"index_mode={index_mode} "
            f"manifest={local_manifest or 'pvc'}"
        )
        dataset = build_training_dataset(config, dataset_motion_rep)
        self.dataset = dataset
        self._boot_log(
            f"dataset ready entries={len(dataset)} in {time.perf_counter() - phase_t0:.1f}s"
        )
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
            if config.data.multiprocessing_context != "auto":
                loader_kwargs["multiprocessing_context"] = (
                    config.data.multiprocessing_context
                )
        self.loader = DataLoader(**loader_kwargs)
        if len(self.loader) == 0:
            raise ValueError("Training DataLoader is empty; lower batch_size or add data")
        self._boot_log(f"dataloader ready len={len(self.loader)} batches/epoch")

        if config.runtime.resume:
            checkpoint_t0 = time.perf_counter()
            self._boot_log(
                f"loading {config.runtime.resume_mode} checkpoint {config.runtime.resume}"
            )
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
            if self.global_step > config.total_steps:
                raise ValueError(
                    f"resume checkpoint global_step={self.global_step} exceeds configured "
                    f"total_steps={config.total_steps}"
                )
            self._boot_log(
                f"resumed checkpoint global_step={self.global_step} epoch={self.epoch} "
                f"batch_in_epoch={self.batch_in_epoch} "
                f"elapsed_s={time.perf_counter() - checkpoint_t0:.1f}"
            )
            configured_lr = self._scheduled_learning_rate()
            for group in self.optimizer.param_groups:
                group["lr"] = configured_lr
                group["weight_decay"] = self._weight_decay_for_group(group)
            if (
                config.runtime.resume_mode == "fork"
                and config.runtime.reset_optimizer
            ):
                self._boot_log(
                    "fork reset optimizer moments; EMA restored from parent "
                    f"weight_decay={self.config.optimizer.weight_decay} "
                    f"last_layer_weight_decay={self.config.optimizer.last_layer_weight_decay}"
                )
            self._boot_log(f"optimizer learning_rate={configured_lr}")

        self.run_lock = ExclusiveRunLock(self.output_dir) if self.context.is_main else None
        lock_error = None
        if self.context.is_main:
            try:
                self.run_lock.acquire()
            except FileExistsError as error:
                lock_error = str(error)
        if self.context.world_size > 1:
            payload = [lock_error]
            dist.broadcast_object_list(payload, src=0)
            lock_error = payload[0]
        if lock_error is not None:
            raise FileExistsError(lock_error)
        self.wandb_monitor = WandbMonitor()
        if self.context.is_main:
            try:
                from .config import save_resolved_config

                save_resolved_config(config, self.output_dir / "config.resolved.yaml")
                save_provenance(self.provenance, self.output_dir / "provenance.json")
                self.wandb_monitor = WandbMonitor.from_env(
                    "train",
                    output_dir=self.output_dir / ".wandb",
                    identity_root=self.output_dir,
                    config=config.to_dict(),
                    metadata={
                        "kimodo/image_git_commit": self.provenance.get("image_git_commit"),
                        "kimodo/git_commit": self.provenance.get("git_commit"),
                        "kimodo/world_size": self.context.world_size,
                        "kimodo/output_dir": str(self.output_dir),
                    },
                )
            except Exception:
                self.run_lock.release()
                raise
        self.logger = JsonlLogger(
            self.output_dir / "train.jsonl", self.context.is_main, self.wandb_monitor
        )
        self._boot_log(f"trainer init complete in {time.perf_counter() - boot_t0:.1f}s")

    def _boot_log(self, message: str) -> None:
        if getattr(self, "context", None) is None or self.context.is_main:
            print(f"[kimodo-train] {message}", flush=True)

    def _scheduled_learning_rate(self) -> float:
        optimizer = self.config.optimizer
        return scheduled_learning_rate(
            self.global_step,
            peak_lr=float(optimizer.learning_rate),
            total_steps=int(self.config.total_steps),
            warmup_steps=int(optimizer.warmup_steps),
            warmup_start_lr=optimizer.warmup_start_lr,
            lr_end=optimizer.lr_end,
            schedule_start_step=int(optimizer.lr_schedule_start_step),
        )

    def _weight_decay_for_group(self, group: dict) -> float:
        last_decay = self.config.optimizer.last_layer_weight_decay
        if last_decay is not None and group.get("name") == "last_layer":
            return float(last_decay)
        return float(self.config.optimizer.weight_decay)

    def _optimizer_group(self, name: str) -> dict:
        for group in self.optimizer.param_groups:
            if group.get("name") == name:
                return group
        return self.optimizer.param_groups[0]

    def _apply_scheduled_optimizer_hyperparams(self) -> None:
        learning_rate = self._scheduled_learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
            group["weight_decay"] = self._weight_decay_for_group(group)

    def _body_layer_grad_norms(self) -> dict[str, float]:
        body = unwrap_model(self.model).body_model
        encoder = getattr(body, "seqTransEncoder", None)
        if encoder is None:
            return {}
        report: dict[str, float] = {}
        layer_squares = []
        for index, layer in enumerate(encoder.layers):
            squares = [
                parameter.grad.detach().float().pow(2).sum()
                for parameter in layer.parameters()
                if parameter.grad is not None
            ]
            value = float(torch.stack(squares).sum().sqrt()) if squares else 0.0
            report[f"body_layer_{index:02d}_grad_norm"] = value
            layer_squares.extend(squares)
        report["body_grad_norm"] = (
            float(torch.stack(layer_squares).sum().sqrt()) if layer_squares else 0.0
        )
        return report

    def _accumulate_curriculum_counts(
        self,
        curriculum_counts: dict[str, float],
        *,
        text_dropped: torch.Tensor,
        lengths: torch.Tensor,
        conditioning,
        mixture_sources: list[str],
    ) -> None:
        """Update logging counters without per-step training math.

        Phase-1 lanes never need sequence lengths (no benchmark duration bins),
        so we skip that host sync until Phase-2 actually emits benchmark lanes.
        """
        lanes = conditioning.sampling_lanes
        need_lengths = any(lane == "benchmark" for lane in lanes)
        # One host transfer for dropout flags (not one sync per sample).
        dropped_values = text_dropped.detach().to("cpu").tolist()
        length_values = lengths.detach().to("cpu").tolist() if need_lengths else None
        fps = float(self.config.data.fps)
        for index, (dropped_value, patterns, lane, component_count) in enumerate(
            zip(
                dropped_values,
                conditioning.pattern_names,
                lanes,
                conditioning.component_counts,
                strict=True,
            )
        ):
            dropped = bool(dropped_value)
            constrained = bool(patterns)
            curriculum_counts["samples"] += 1
            curriculum_counts["text_dropped"] += float(dropped)
            curriculum_counts["constrained"] += float(constrained)
            curriculum_counts["two_patterns"] += float(lane == "paper_two")
            curriculum_counts["none_lane"] += float(lane in {"none", "phase1_none"})
            curriculum_counts["paper_single_lane"] += float(lane == "paper_single")
            curriculum_counts["benchmark_lane"] += float(lane == "benchmark")
            curriculum_counts["benchmark_atomic"] += float(
                lane == "benchmark" and component_count == 1
            )
            curriculum_counts["benchmark_two_component"] += float(
                lane == "benchmark" and component_count == 2
            )
            curriculum_counts["benchmark_three_component"] += float(
                lane == "benchmark" and component_count == 3
            )
            curriculum_counts["benchmark_with_text"] += float(
                lane == "benchmark" and not dropped
            )
            curriculum_counts["benchmark_without_text"] += float(
                lane == "benchmark" and dropped
            )
            if need_lengths and lane == "benchmark":
                duration_seconds = float(length_values[index]) / fps
                curriculum_counts["benchmark_duration_lt_3s"] += float(duration_seconds < 3.0)
                curriculum_counts["benchmark_duration_3_to_10s"] += float(
                    3.0 <= duration_seconds <= 10.0
                )
                curriculum_counts["benchmark_duration_gt_10s"] += float(duration_seconds > 10.0)
            curriculum_counts["exact_two_component"] += float(component_count == 2)
            curriculum_counts["physical_multi_constraint"] += float(component_count >= 2)
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
        batch_size = float(len(lanes))
        curriculum_counts["sparse_keyframe_count_sum"] += (
            float(conditioning.sampled_sparse_keyframe_count_mean) * batch_size
        )
        curriculum_counts["sparse_constraint_load_sum"] += (
            float(conditioning.sparse_constraint_load_mean) * batch_size
        )
        curriculum_counts["mask_channel_load_sum"] += (
            float(conditioning.mask_channel_load_mean) * batch_size
        )
        curriculum_counts["mask_channel_load_max"] = max(
            float(curriculum_counts["mask_channel_load_max"]),
            float(conditioning.mask_channel_load_max),
        )
        for source in mixture_sources:
            curriculum_counts[f"data_source/{source}"] += 1

    def _observed_section(self, name: str):
        """Return an optional benchmark-only profiling range."""
        if self.step_observer is None or not hasattr(self.step_observer, "section"):
            return contextlib.nullcontext()
        return self.step_observer.section(name)

    def _export_current_inference_bundle_if_missing(self) -> None:
        destination = self.output_dir / "exports" / f"step-{self.global_step:09d}"
        published = False
        if self.context.is_main and not destination.is_dir():
            export_inference_bundle(
                self.model,
                self.ema,
                self.output_dir,
                self.global_step,
                self.config,
            )
            published = True
        if published:
            self.wandb_monitor.log(
                {
                    "lifecycle/ema_export_published": 1,
                    "lifecycle/ema_export_path": str(destination),
                },
                step=self.global_step,
            )
        if self.context.world_size > 1:
            dist.barrier()

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
            saved = self.checkpoints.save_diagnostic(state, diagnostic_reason)
            self.wandb_monitor.log(
                {
                    "lifecycle/diagnostic_checkpoint_saved": 1,
                    "lifecycle/diagnostic_reason": diagnostic_reason,
                    "lifecycle/checkpoint_path": str(saved),
                },
                step=self.global_step,
            )
            return saved
        saved = self.checkpoints.save(state)
        self._last_saved_position = position
        self.wandb_monitor.log(
            {
                "lifecycle/checkpoint_saved": 1,
                "lifecycle/checkpoint_path": str(saved),
            },
            step=self.global_step,
        )
        return saved

    def train(self) -> None:
        exit_code = 0
        try:
            self._train_impl()
            self.wandb_monitor.summary(
                {"kimodo/status": "completed", "kimodo/final_global_step": self.global_step}
            )
        except BaseException:
            exit_code = 1
            self.wandb_monitor.summary(
                {"kimodo/status": "failed", "kimodo/final_global_step": self.global_step}
            )
            raise
        finally:
            try:
                self.logger.close()
            except Exception:
                if exit_code == 0:
                    exit_code = 1
                print("[kimodo-train] metrics logger close failed", flush=True)
            self.wandb_monitor.finish(exit_code=exit_code)
            if self.run_lock is not None:
                self.run_lock.release()

    def _train_impl(self) -> None:
        accumulation = self.config.runtime.gradient_accumulation_steps
        self.optimizer.zero_grad(set_to_none=True)
        skip_batches = self.batch_in_epoch
        if skip_batches >= len(self.loader):
            completed_epochs, skip_batches = divmod(skip_batches, len(self.loader))
            self.epoch += completed_epochs
            self.batch_in_epoch = skip_batches
        started = time.time()
        interval_started = time.perf_counter()
        interval_optimizer_steps = 0
        interval_skipped_steps = 0
        interval_gradient_norm_sum = 0.0
        interval_gradient_clip_hits = 0
        interval_extreme_skips = 0
        interval_body_layer_sums: dict[str, float] = {}
        first_batch_logged = False
        first_committed_step = self.global_step + 1
        if self.context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.context.device)
        if (
            self.config.runtime.resume
            and self.config.runtime.milestone_every
            and self.global_step % self.config.runtime.milestone_every == 0
        ):
            # A job may fail after atomically publishing the trainer checkpoint
            # but before publishing its EMA bundle. Recreate that derived
            # artifact on resume without changing any training state.
            self._export_current_inference_bundle_if_missing()
        accumulated_valid_frames = 0
        accumulated_loss_sums: dict[str, torch.Tensor] = {}
        curriculum_counts = {
            "samples": 0.0,
            "text_dropped": 0.0,
            "constrained": 0.0,
            "two_patterns": 0.0,
            "none_lane": 0.0,
            "paper_single_lane": 0.0,
            "benchmark_lane": 0.0,
            "benchmark_atomic": 0.0,
            "benchmark_two_component": 0.0,
            "benchmark_three_component": 0.0,
            "benchmark_with_text": 0.0,
            "benchmark_without_text": 0.0,
            "benchmark_duration_lt_3s": 0.0,
            "benchmark_duration_3_to_10s": 0.0,
            "benchmark_duration_gt_10s": 0.0,
            "exact_two_component": 0.0,
            "physical_multi_constraint": 0.0,
            "joint": 0.0,
            "constraint_only": 0.0,
            "text_only": 0.0,
            "unconditional": 0.0,
            "sparse_keyframe_count_sum": 0.0,
            "sparse_constraint_load_sum": 0.0,
            "mask_channel_load_sum": 0.0,
            "mask_channel_load_max": 0.0,
            **{f"pattern/{name}": 0.0 for name in self.constraint_sampler.ALL_PATTERNS},
            **{f"data_source/{name}": 0.0 for name in self.dataset.mixture_sources},
        }

        self._boot_log(
            f"entering train loop from global_step={self.global_step} "
            f"target={self.config.total_steps}"
        )
        while self.global_step < self.config.total_steps:
            self.dataset.set_epoch(self.epoch)
            self.distributed_sampler.set_epoch(self.epoch)
            for batch_index, batch in enumerate(self.loader):
                if batch_index < skip_batches:
                    continue
                skip_batches = 0
                if not first_batch_logged:
                    self._boot_log(
                        f"first training batch ready epoch={self.epoch} batch_index={batch_index}"
                    )
                    first_batch_logged = True
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
                    self._accumulate_curriculum_counts(
                        curriculum_counts,
                        text_dropped=text_dropped,
                        lengths=batch["lengths"],
                        conditioning=conditioning,
                        mixture_sources=batch["mixture_sources"],
                    )
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
                        losses = self.loss(
                            prediction,
                            batch["clean_motion"],
                            batch["valid_frames"],
                        )
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
                    per_parameter_norms = [
                        parameter.grad.detach().float().norm(2)
                        for parameter in self.model.parameters()
                        if parameter.grad is not None
                    ]
                    gradient_norm = (
                        torch.stack(per_parameter_norms).norm(2)
                        if per_parameter_norms
                        else torch.zeros((), device=self.context.device)
                    )
                    gradient_finite = torch.isfinite(gradient_norm).to(dtype=torch.float32)
                    if self.context.world_size > 1:
                        dist.all_reduce(gradient_finite, op=dist.ReduceOp.MIN)
                    if not bool(gradient_finite.item()):
                        self._save(diagnostic_reason="nonfinite-gradient")
                        raise FloatingPointError(
                            f"Non-finite gradient norm at global_step={self.global_step}"
                        )
                    gradient_norm_value = float(gradient_norm.detach().float().item())
                    # Per-layer norms sync GPU→CPU for every body block. Do that
                    # only on log steps; collapse watches jsonl, not the 20-step mean.
                    log_every = max(int(self.config.runtime.log_every), 1)
                    step_body_layer_norms = (
                        self._body_layer_grad_norms()
                        if (self.global_step + 1) % log_every == 0
                        else {}
                    )
                    skip_extreme = torch.tensor(0.0, device=self.context.device)
                    skip_threshold = self.config.optimizer.skip_gradient_norm
                    if skip_threshold is not None and gradient_norm_value > skip_threshold:
                        skip_extreme.fill_(1.0)
                    if self.context.world_size > 1:
                        dist.all_reduce(skip_extreme, op=dist.ReduceOp.MAX)
                    if bool(skip_extreme.item()):
                        interval_skipped_steps += 1
                        interval_extreme_skips += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        abort_fraction = float(self.config.optimizer.skip_gradient_abort_fraction)
                        attempted = interval_extreme_skips + interval_optimizer_steps
                        if (
                            attempted >= max(self.config.runtime.log_every, 1)
                            and interval_extreme_skips / max(attempted, 1) > abort_fraction
                        ):
                            raise RuntimeError(
                                "extreme-gradient skip fraction "
                                f"{interval_extreme_skips}/{attempted} "
                                f"exceeded {abort_fraction}"
                            )
                        self.scaler.update()
                        accumulated_valid_frames = 0
                        accumulated_loss_sums.clear()
                        continue
                    if self.config.optimizer.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.optimizer.gradient_clip_norm
                        )
                self._apply_scheduled_optimizer_hyperparams()
                with self._observed_section("optimizer"):
                    previous_scale = self.scaler.get_scale()
                    if hasattr(self.optimizer, "track_update_stats"):
                        self.optimizer.track_update_stats = (
                            (self.global_step + 1) % log_every == 0
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                step_skipped = self.scaler.is_enabled() and self.scaler.get_scale() < previous_scale
                if step_skipped:
                    interval_skipped_steps += 1
                    accumulated_valid_frames = 0
                    accumulated_loss_sums.clear()
                    for name in curriculum_counts:
                        curriculum_counts[name] = 0.0
                    continue
                self.global_step += 1
                if self.global_step == first_committed_step:
                    self._boot_log(
                        f"first optimizer step committed global_step={self.global_step}"
                    )
                interval_optimizer_steps += 1
                interval_gradient_norm_sum += gradient_norm_value
                for name, value in step_body_layer_norms.items():
                    interval_body_layer_sums[name] = value
                interval_gradient_clip_hits += int(
                    self.config.optimizer.gradient_clip_norm is not None
                    and gradient_norm_value > self.config.optimizer.gradient_clip_norm
                )

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
                    count_names = [name for name in curriculum_counts if name != "mask_channel_load_max"]
                    count_values = torch.tensor(
                        [curriculum_counts[name] for name in count_names],
                        device=self.context.device,
                        dtype=torch.float32,
                    )
                    if self.context.world_size > 1:
                        dist.all_reduce(count_values, op=dist.ReduceOp.SUM)
                    global_counts = dict(zip(count_names, count_values.tolist()))
                    mask_load_max = torch.tensor(
                        float(curriculum_counts["mask_channel_load_max"]),
                        device=self.context.device,
                        dtype=torch.float32,
                    )
                    if self.context.world_size > 1:
                        dist.all_reduce(mask_load_max, op=dist.ReduceOp.MAX)
                    sample_count = max(1.0, global_counts["samples"])
                    benchmark_count = max(1.0, global_counts["benchmark_lane"])
                    interval_seconds_value = torch.tensor(
                        time.perf_counter() - interval_started,
                        device=self.context.device,
                        dtype=torch.float64,
                    )
                    if self.context.world_size > 1:
                        dist.all_reduce(interval_seconds_value, op=dist.ReduceOp.MAX)
                    interval_seconds = max(float(interval_seconds_value.item()), 1e-12)
                    optimization_values = torch.tensor(
                        [
                            interval_gradient_norm_sum,
                            float(interval_gradient_clip_hits),
                            float(interval_optimizer_steps),
                            float(interval_skipped_steps),
                            float(interval_extreme_skips),
                        ],
                        device=self.context.device,
                        dtype=torch.float64,
                    )
                    if self.context.world_size > 1:
                        dist.all_reduce(optimization_values, op=dist.ReduceOp.SUM)
                    gradient_observations = max(float(optimization_values[2].item()), 1.0)
                    attempted_observations = max(
                        float((optimization_values[2] + optimization_values[3]).item()), 1.0
                    )
                    if self.context.device.type == "cuda":
                        memory_values = torch.tensor(
                            [
                                torch.cuda.max_memory_allocated(self.context.device),
                                torch.cuda.max_memory_reserved(self.context.device),
                            ],
                            device=self.context.device,
                            dtype=torch.float64,
                        )
                        if self.context.world_size > 1:
                            dist.all_reduce(memory_values, op=dist.ReduceOp.MAX)
                        peak_allocated_bytes = int(memory_values[0].item())
                        peak_reserved_bytes = int(memory_values[1].item())
                    else:
                        peak_allocated_bytes = 0
                        peak_reserved_bytes = 0
                    record = {
                        "global_step": self.global_step,
                        "phase": phase,
                        "epoch": self.epoch,
                        "elapsed_seconds": time.time() - started,
                        "text_dropout_fraction": global_counts["text_dropped"] / sample_count,
                        "constraint_fraction": global_counts["constrained"] / sample_count,
                        "two_pattern_fraction": global_counts["two_patterns"] / sample_count,
                        "paper_two_pattern_fraction": global_counts["two_patterns"] / sample_count,
                        "none_lane_fraction": global_counts["none_lane"] / sample_count,
                        "paper_single_lane_fraction": (
                            global_counts["paper_single_lane"] / sample_count
                        ),
                        "benchmark_lane_fraction": global_counts["benchmark_lane"] / sample_count,
                        "benchmark_atomic_fraction": global_counts["benchmark_atomic"] / sample_count,
                        "benchmark_atomic_per_sample": (
                            global_counts["benchmark_atomic"] / sample_count
                        ),
                        "benchmark_atomic_within_benchmark": (
                            global_counts["benchmark_atomic"] / benchmark_count
                        ),
                        "benchmark_two_component_fraction": (
                            global_counts["benchmark_two_component"] / sample_count
                        ),
                        "benchmark_two_component_per_sample": (
                            global_counts["benchmark_two_component"] / sample_count
                        ),
                        "benchmark_two_component_within_benchmark": (
                            global_counts["benchmark_two_component"] / benchmark_count
                        ),
                        "benchmark_three_component_fraction": (
                            global_counts["benchmark_three_component"] / sample_count
                        ),
                        "benchmark_three_component_per_sample": (
                            global_counts["benchmark_three_component"] / sample_count
                        ),
                        "benchmark_three_component_within_benchmark": (
                            global_counts["benchmark_three_component"] / benchmark_count
                        ),
                        "benchmark_with_text_within_benchmark": (
                            global_counts["benchmark_with_text"] / benchmark_count
                        ),
                        "benchmark_without_text_within_benchmark": (
                            global_counts["benchmark_without_text"] / benchmark_count
                        ),
                        "benchmark_duration_lt_3s_within_benchmark": (
                            global_counts["benchmark_duration_lt_3s"] / benchmark_count
                        ),
                        "benchmark_duration_3_to_10s_within_benchmark": (
                            global_counts["benchmark_duration_3_to_10s"] / benchmark_count
                        ),
                        "benchmark_duration_gt_10s_within_benchmark": (
                            global_counts["benchmark_duration_gt_10s"] / benchmark_count
                        ),
                        "exact_two_component_fraction": (
                            global_counts["exact_two_component"] / sample_count
                        ),
                        "intended_multi_component_fraction": (
                            global_counts["physical_multi_constraint"] / sample_count
                        ),
                        "physical_multi_constraint_fraction": (
                            global_counts["physical_multi_constraint"] / sample_count
                        ),
                        "maximum_sparse_keyframes": conditioning.maximum_sparse_keyframes,
                        "scheduled_sparse_keyframes": conditioning.scheduled_sparse_keyframes,
                        "sampled_sparse_keyframe_cap_mean": (
                            conditioning.sampled_sparse_keyframe_cap_mean
                        ),
                        "sampled_sparse_keyframe_count_mean": (
                            global_counts["sparse_keyframe_count_sum"] / sample_count
                        ),
                        "sparse_constraint_load_mean": (
                            global_counts["sparse_constraint_load_sum"] / sample_count
                        ),
                        "conditioning/mask_channel_load_mean": (
                            global_counts["mask_channel_load_sum"] / sample_count
                        ),
                        "conditioning/mask_channel_load_max": float(mask_load_max.item()),
                        "optimizer/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                        "optimizer/weight_decay": float(
                            self._optimizer_group("rest").get(
                                "weight_decay", self.config.optimizer.weight_decay
                            )
                        ),
                        "optimizer/last_layer_weight_decay": float(
                            self._optimizer_group("last_layer").get(
                                "weight_decay",
                                self.config.optimizer.last_layer_weight_decay
                                if self.config.optimizer.last_layer_weight_decay is not None
                                else self.config.optimizer.weight_decay,
                            )
                        ),
                        "optimizer/update_norm": float(
                            getattr(self.optimizer, "last_update_norm", 0.0) or 0.0
                        ),
                        "optimizer/update_param_ratio": float(
                            getattr(self.optimizer, "last_update_param_ratio", 0.0)
                            or 0.0
                        ),
                        "optimizer/gradient_norm_before_clip": (
                            float(optimization_values[0].item()) / gradient_observations
                        ),
                        "optimizer/gradient_clip_fraction": (
                            float(optimization_values[1].item()) / gradient_observations
                        ),
                        "optimizer/skipped_step_fraction": (
                            float(optimization_values[3].item()) / attempted_observations
                        ),
                        "optimizer/extreme_gradient_skip_fraction": (
                            float(optimization_values[4].item()) / attempted_observations
                        ),
                        "ema/num_updates": self.ema.num_updates if self.ema is not None else 0,
                        "system/world_size": self.context.world_size,
                        "system/per_rank_batch": self.config.runtime.batch_size,
                        "system/gradient_accumulation_steps": (
                            self.config.runtime.gradient_accumulation_steps
                        ),
                        "system/effective_global_batch": (
                            self.config.runtime.batch_size
                            * self.config.runtime.gradient_accumulation_steps
                            * self.context.world_size
                        ),
                        "system/interval_seconds": interval_seconds,
                        "system/optimizer_steps_per_second": (
                            interval_optimizer_steps / interval_seconds
                        ),
                        "system/samples_per_second": (
                            self.config.runtime.batch_size
                            * self.config.runtime.gradient_accumulation_steps
                            * self.context.world_size
                            * interval_optimizer_steps
                            / interval_seconds
                        ),
                        "system/peak_cuda_allocated_bytes": peak_allocated_bytes,
                        "system/peak_cuda_reserved_bytes": peak_reserved_bytes,
                    }
                    for name, value in interval_body_layer_sums.items():
                        record[f"optimizer/{name}"] = value
                    for branch in ("joint", "constraint_only", "text_only", "unconditional"):
                        record[f"conditioning/{branch}_fraction"] = global_counts[branch] / sample_count
                    for pattern in self.constraint_sampler.ALL_PATTERNS:
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
                    for term_name, _ in self.loss.FEATURE_TERMS:
                        record[f"loss_weighted/{term_name}"] = (
                            record[f"loss/{term_name}"]
                            * float(getattr(self.config.loss, term_name))
                        )
                    record["loss_weighted/forward_kinematics"] = (
                        record["loss/forward_kinematics"]
                        * float(self.config.loss.forward_kinematics)
                    )
                    self.logger.write(record)
                    interval_started = time.perf_counter()
                    interval_optimizer_steps = 0
                    interval_skipped_steps = 0
                    interval_extreme_skips = 0
                    interval_gradient_norm_sum = 0.0
                    interval_gradient_clip_hits = 0
                    interval_body_layer_sums.clear()
                    if self.context.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(self.context.device)

                accumulated_valid_frames = 0
                accumulated_loss_sums.clear()
                for name in curriculum_counts:
                    curriculum_counts[name] = 0.0

                milestone_step = bool(
                    self.config.runtime.milestone_every
                    and self.global_step % self.config.runtime.milestone_every == 0
                )
                if (
                    self.global_step % self.config.runtime.checkpoint_every == 0
                    or milestone_step
                    or self.global_step == self.config.curriculum.phase1_steps
                ):
                    self._save()
                if milestone_step and self.global_step < self.config.total_steps:
                    self._export_current_inference_bundle_if_missing()
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
        self._export_current_inference_bundle_if_missing()
