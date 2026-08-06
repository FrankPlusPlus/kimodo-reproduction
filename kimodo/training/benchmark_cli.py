# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproducible end-to-end H200 benchmark for the production training step."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_fixture(output: str | Path, *, samples: int, frames: int, llm_dim: int) -> None:
    """Create fixed-shape inputs; values are synthetic but the production path is unchanged."""
    root = Path(output).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty benchmark fixture: {root}")
    root.mkdir(parents=True, exist_ok=True)

    stats = root / "stats"
    for name, width in (("global_root", 5), ("local_root", 4), ("body", 364)):
        folder = stats / name
        folder.mkdir(parents=True, exist_ok=False)
        np.save(folder / "mean.npy", np.zeros(width, dtype=np.float32), allow_pickle=False)
        np.save(folder / "std.npy", np.ones(width, dtype=np.float32), allow_pickle=False)

    joints = 30
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32), (frames, joints, 3, 3)
    ).copy()
    roots = np.zeros((frames, 3), dtype=np.float32)
    roots[:, 0] = np.linspace(0.0, 2.0, frames, dtype=np.float32)
    roots[:, 1] = 1.0
    motion = root / "motion-300f.npz"
    np.savez(motion, local_rot_mats=rotations, root_positions=roots)

    embedding = root / "llm2vec-4096d.npy"
    np.save(embedding, np.zeros((1, llm_dim), dtype=np.float32), allow_pickle=False)
    manifest = root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for index in range(samples):
            record = {
                "id": f"benchmark-{index:06d}",
                "motion": motion.name,
                "text": "A person walks forward.",
                "split": "train",
                "source_fps": 30,
                "text_embedding": embedding.name,
                "sample_kind": "full",
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "fixture": str(root),
                "samples": samples,
                "frames": frames,
                "llm_dim": llm_dim,
                "manifest_sha256": _sha256_file(manifest),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _git_identity(project_root: Path) -> dict:
    commit = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(project_root), "status", "--porcelain"], text=True
        ).strip()
    )
    return {"commit": commit, "dirty": dirty}


def _count_periodic_updates(start_step: int, end_step: int, update_every: int) -> int:
    """Count updates at divisible steps in the measured interval ``(start, end]``."""
    if start_step < 0 or end_step < start_step or update_every < 1:
        raise ValueError("Invalid periodic-update interval")
    return end_step // update_every - start_step // update_every


class EndToEndWindowObserver:
    """Measure a synchronized optimizer-step window without per-step synchronization."""

    def __init__(self, *, start_step: int, end_step: int, result_path: Path) -> None:
        if start_step < 1 or end_step <= start_step:
            raise ValueError("Benchmark window must contain at least one optimizer step")
        self.start_step = start_step
        self.end_step = end_step
        self.result_path = result_path
        self.started_at: float | None = None
        self.baseline_allocated = 0
        self.sections_enabled = False
        self.setup_timings: dict[str, float] = {}

    @staticmethod
    def _synchronize(context) -> None:
        import torch
        import torch.distributed as dist

        torch.cuda.synchronize(context.device)
        if context.world_size > 1:
            dist.barrier()
        torch.cuda.synchronize(context.device)

    def on_optimizer_step_end(self, trainer) -> None:
        import torch
        import torch.distributed as dist

        step = trainer.global_step
        context = trainer.context
        if context.device.type != "cuda":
            raise RuntimeError("The production benchmark requires CUDA")
        if step == self.start_step:
            self._synchronize(context)
            torch.cuda.reset_peak_memory_stats(context.device)
            self.baseline_allocated = torch.cuda.memory_allocated(context.device)
            self.started_at = time.perf_counter()
            self.sections_enabled = True
            return
        if step != self.end_step:
            return
        if self.started_at is None:
            raise RuntimeError("Benchmark end reached before its synchronized start")
        self._synchronize(context)
        elapsed = time.perf_counter() - self.started_at
        self.sections_enabled = False
        props = torch.cuda.get_device_properties(context.device)
        local = {
            "rank": context.rank,
            "visible_device": context.device.index,
            "gpu_name": props.name,
            "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
            "total_memory_bytes": int(props.total_memory),
            "baseline_allocated_bytes": int(self.baseline_allocated),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(context.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(context.device)),
            "elapsed_seconds": elapsed,
        }
        by_rank = [None] * context.world_size
        if context.world_size > 1:
            dist.all_gather_object(by_rank, local)
        else:
            by_rank[0] = local
        if context.is_main:
            measured_steps = self.end_step - self.start_step
            window_seconds = max(float(item["elapsed_seconds"]) for item in by_rank)
            effective_batch = (
                context.world_size
                * trainer.config.runtime.batch_size
                * trainer.config.runtime.gradient_accumulation_steps
            )
            config_json = json.dumps(
                trainer.config.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            result = {
                "schema_version": 1,
                "kind": "end_to_end_fixed_300_frames",
                "window": {
                    "start_after_optimizer_step": self.start_step,
                    "end_after_optimizer_step": self.end_step,
                    "measured_optimizer_steps": measured_steps,
                    "measured_micro_steps": (
                        measured_steps * trainer.config.runtime.gradient_accumulation_steps
                    ),
                    "wall_seconds": window_seconds,
                    "optimizer_step_mean_seconds": window_seconds / measured_steps,
                    "samples_per_second": measured_steps * effective_batch / window_seconds,
                    "nominal_frames_per_second": (
                        measured_steps * effective_batch * 300 / window_seconds
                    ),
                },
                "batch": {
                    "world_size": context.world_size,
                    "local_micro_batch": trainer.config.runtime.batch_size,
                    "gradient_accumulation_steps": (
                        trainer.config.runtime.gradient_accumulation_steps
                    ),
                    "effective_global_batch": effective_batch,
                },
                "data": {
                    "manifest": str(Path(trainer.config.data.manifest).resolve()),
                    "manifest_sha256": _sha256_file(trainer.config.data.manifest),
                    "fixed_frames": 300,
                    "workers_per_rank": trainer.config.data.num_workers,
                    "prefetch_factor": trainer.config.data.prefetch_factor,
                    "pin_memory": trainer.config.data.pin_memory,
                },
                "runtime": {
                    "precision": trainer.config.runtime.precision,
                    "phase": trainer._apply_phase(),
                    "seed": trainer.config.runtime.seed,
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
                },
                "setup": self.setup_timings,
                "config_sha256": hashlib.sha256(config_json).hexdigest(),
                "git": _git_identity(trainer.project_root),
                "rank_measurements": by_rank,
                "notes": {
                    "checkpoint_in_window": False,
                    "logging_in_window": False,
                    "ema": {
                        "enabled": trainer.config.ema.enabled,
                        "decay": trainer.config.ema.decay,
                        "update_every_optimizer_steps": trainer.config.ema.update_every,
                        "updates_in_window": (
                            _count_periodic_updates(
                                self.start_step,
                                self.end_step,
                                trainer.config.ema.update_every,
                            )
                            if trainer.config.ema.enabled
                            else 0
                        ),
                    },
                    "training_math_changed": False,
                },
            }
            self.result_path.parent.mkdir(parents=True, exist_ok=True)
            self.result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(
                json.dumps({"benchmark_result": str(self.result_path), **result["window"]}),
                flush=True,
            )

    def section(self, name: str):
        """Expose coarse production-step ranges to Nsight Systems."""
        if not self.sections_enabled:
            return contextlib.nullcontext()
        import torch

        return torch.cuda.nvtx.range(f"kimodo::{name}")


class _ObservedDataLoader:
    """Benchmark-only wrapper that exposes time blocked in DataLoader.__next__."""

    def __init__(self, loader, observer: EndToEndWindowObserver) -> None:
        self.loader = loader
        self.observer = observer

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        while True:
            try:
                with self.observer.section("data_loader_wait"):
                    batch = next(iterator)
            except StopIteration:
                return
            yield batch


def _annotate_two_stage_forward(trainer, observer: EndToEndWindowObserver) -> None:
    """Add benchmark-only root/body ranges without changing module outputs."""
    from .modeling import unwrap_model

    bare = unwrap_model(trainer.model)
    for name, module in (
        ("root_transformer_forward", bare.root_model),
        ("body_transformer_forward", bare.body_model),
    ):
        original_forward = module.forward

        def profiled_forward(*args, _name=name, _forward=original_forward, **kwargs):
            with observer.section(_name):
                return _forward(*args, **kwargs)

        module.forward = profiled_forward


def run_benchmark(args) -> None:
    from .config import load_training_config

    setup_started = time.perf_counter()
    fixture = Path(args.fixture).expanduser().resolve()
    initial_step = int(args.initial_step)
    final_step = initial_step + args.warmup_steps + args.measure_steps
    output_dir = Path(args.output_dir).expanduser().resolve()
    result_path = Path(args.result).expanduser().resolve()
    # Candidate-only execution switches (for example compile/foreach) are
    # applied first; benchmark-controlled data, scale, timing and output fields
    # below remain authoritative.
    overrides = list(args.set) + [
        f"data.manifest={fixture / 'manifest.jsonl'}",
        f"data.num_workers={args.num_workers}",
        f"data.prefetch_factor={args.prefetch_factor}",
        "data.pin_memory=true",
        "data.persistent_workers=false",
        "data.reference_verification=full",
        "data.reference_inventory=null",
        "data.require_paper_data_parity=false",
        f"model.stats_path={fixture / 'stats'}",
        f"runtime.output_dir={output_dir}",
        "runtime.device=auto",
        "runtime.precision=bf16",
        f"runtime.batch_size={args.batch_size}",
        f"runtime.gradient_accumulation_steps={args.accumulation}",
        "runtime.log_every=100000000",
        "runtime.checkpoint_every=100000000",
        "runtime.milestone_every=0",
        f"runtime.initial_global_step={initial_step}",
        f"runtime.max_steps_override={final_step}",
        "runtime.enforce_paper_scale=false",
        "paper_method_strict=false",
    ]
    config = load_training_config(args.config, overrides)
    config_loaded_at = time.perf_counter()
    observer = EndToEndWindowObserver(
        start_step=initial_step + args.warmup_steps,
        end_step=final_step,
        result_path=result_path,
    )

    # Imported only after CUDA visibility is fixed by the outer process.
    from . import engine as engine_module

    project_root = Path(__file__).resolve().parents[2]
    trainer = engine_module.KimodoTrainer(config, project_root, step_observer=observer)
    trainer_initialized_at = time.perf_counter()
    observer.setup_timings = {
        "config_and_override_seconds": config_loaded_at - setup_started,
        "trainer_initialization_seconds": trainer_initialized_at - config_loaded_at,
        "total_before_dataloader_iteration_seconds": trainer_initialized_at - setup_started,
    }
    trainer.loader = _ObservedDataLoader(trainer.loader, observer)
    _annotate_two_stage_forward(trainer, observer)
    required_micro_steps = (args.warmup_steps + args.measure_steps) * args.accumulation
    if len(trainer.loader) < required_micro_steps:
        raise ValueError(
            "Benchmark fixture is too small: the run would cross an epoch and restart "
            f"workers (loader batches/rank={len(trainer.loader)}, required={required_micro_steps}). "
            "Create a fixture with more rows."
        )
    # Benchmark outputs must not include checkpoint/export I/O. The optimizer,
    # EMA, curriculum, losses, DDP synchronization, loader and H2D remain real.
    trainer._save = lambda: None
    engine_module.export_inference_bundle = lambda *unused_args, **unused_kwargs: None
    try:
        trainer.train()
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser("fixture", help="Create fixed 300-frame benchmark assets")
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--samples", type=int, default=4096)
    fixture.add_argument("--frames", type=int, default=300)
    fixture.add_argument("--llm-dim", type=int, default=4096)

    run = subparsers.add_parser("run", help="Run one fresh-process benchmark candidate")
    run.add_argument("--config", required=True)
    run.add_argument("--fixture", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--result", required=True)
    run.add_argument("--batch-size", type=int, required=True)
    run.add_argument("--accumulation", type=int, required=True)
    run.add_argument("--warmup-steps", type=int, default=5)
    run.add_argument("--measure-steps", type=int, default=10)
    run.add_argument("--initial-step", type=int, default=0)
    run.add_argument("--num-workers", type=int, default=8)
    run.add_argument("--prefetch-factor", type=int, default=2)
    run.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict candidate config override; benchmark-controlled fields still win",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "fixture":
        if args.samples < 2 or args.frames != 300 or args.llm_dim != 4096:
            raise ValueError("Production benchmark requires samples>=2, frames=300 and llm_dim=4096")
        create_fixture(args.output, samples=args.samples, frames=args.frames, llm_dim=args.llm_dim)
    else:
        if min(args.batch_size, args.accumulation, args.warmup_steps, args.measure_steps) < 1:
            raise ValueError("Batch, accumulation, warmup and measurement counts must be positive")
        run_benchmark(args)


if __name__ == "__main__":
    main()
