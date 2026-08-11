# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic training checkpoint and deterministic resume support."""

from __future__ import annotations

import os
import random
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .modeling import unwrap_model

SCHEMA_VERSION = 3

# Data / asset lineage keys always compared on exact resume. code_snapshot is
# optional so throughput hotfixes on PVC can resume a checkpoint written under
# an earlier code tree (see KIMODO_RESUME_ALLOW_CODE_MISMATCH).
_PROVENANCE_KEYS = (
    "manifest",
    "stats",
    "official_bundle",
    "skeleton_assets",
    "code_snapshot",
)


def _resume_critical_config(config: dict) -> dict:
    value = dict(config)
    runtime = dict(value.get("runtime", {}))
    for key in (
        "output_dir",
        "resume",
        "resume_mode",
        "log_every",
        "checkpoint_every",
        "milestone_every",
        "keep_last_checkpoints",
        "dry_run",
        "max_steps_override",
        "initial_global_step",
    ):
        runtime.pop(key, None)
    value["runtime"] = runtime
    # Loader mechanics do not change per-sample RNG (seeded by epoch+index) or
    # optimizer math; allow throughput retunes across exact resumes.
    data = dict(value.get("data", {}))
    for key in (
        "num_workers",
        "persistent_workers",
        "prefetch_factor",
        "multiprocessing_context",
        "pin_memory",
        # Feature-cache path is a throughput switch; long-clip windowing moves
        # from pre-FK to feature space, so resume with KIMODO_RESUME_ALLOW_CODE_MISMATCH.
        "feature_cache_dir",
    ):
        data.pop(key, None)
    value["data"] = data
    return value


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _publish_mode(path: Path, mode: int = 0o644) -> None:
    """Make trainer artifacts readable by sidecar eval pods on shared PVC."""
    try:
        os.chmod(path, mode)
    except OSError:
        # Best-effort: some filesystems/ACLs refuse chmod from the writer.
        pass


def atomic_torch_save(value, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        _publish_mode(temporary)
        os.replace(temporary, destination)
        _publish_mode(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def exclusive_torch_save(value, path: str | Path) -> None:
    """Atomically publish a torch file while refusing concurrent overwrite."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        _publish_mode(temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to overwrite existing checkpoint: {destination}"
            ) from error
        _publish_mode(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text_write(value: str, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_mode(temporary)
        os.replace(temporary, destination)
        _publish_mode(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_training_state(
    *,
    model,
    optimizer,
    ema,
    scaler,
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    micro_index: int,
    config: dict,
    provenance: dict,
    rng_by_rank: list[dict] | None = None,
    resume_exact: bool = True,
    diagnostic_reason: str | None = None,
) -> dict:
    state = {
        "schema_version": SCHEMA_VERSION,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "micro_index": int(micro_index),
        "config": config,
        "provenance": provenance,
        "rng_by_rank": rng_by_rank if rng_by_rank is not None else [capture_rng_state()],
        "resume_exact": bool(resume_exact),
    }
    if diagnostic_reason is not None:
        state["diagnostic_reason"] = str(diagnostic_reason)
    return state


def load_training_state(
    path: str | Path,
    *,
    model,
    optimizer,
    ema,
    scaler,
    expected_provenance: dict | None = None,
    current_config: dict | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported trainer checkpoint schema: {state.get('schema_version')}")
    if state.get("resume_exact", True) is not True:
        raise ValueError("Diagnostic checkpoint is not an exact optimizer-boundary resume point")
    if expected_provenance is not None:
        saved = state.get("provenance", {})
        allow_code_mismatch = os.environ.get(
            "KIMODO_RESUME_ALLOW_CODE_MISMATCH", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        for key in _PROVENANCE_KEYS:
            if key == "code_snapshot" and allow_code_mismatch:
                continue
            if saved.get(key) != expected_provenance.get(key):
                raise ValueError(f"Resume provenance mismatch for {key}")
    if current_config is not None and _resume_critical_config(state["config"]) != _resume_critical_config(
        current_config
    ):
        raise ValueError(
            "Resume training-critical config differs from the checkpoint; only output/log/checkpoint "
            "controls and max_steps_override may change"
        )
    unwrap_model(model).load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    for parameter, parameter_state in optimizer.state.items():
        for key, value in parameter_state.items():
            if isinstance(value, torch.Tensor):
                parameter_state[key] = value.to(parameter.device)
    if ema is not None:
        if state["ema"] is None:
            raise ValueError("EMA is enabled but resume checkpoint has no EMA state")
        ema.load_state_dict(state["ema"])
        ema.to(next(unwrap_model(model).parameters()).device)
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    rng_by_rank = state.get("rng_by_rank")
    if not isinstance(rng_by_rank, list) or len(rng_by_rank) != world_size:
        raise ValueError(
            f"Checkpoint has RNG for {len(rng_by_rank) if isinstance(rng_by_rank, list) else 0} "
            f"ranks but resume world_size={world_size}"
        )
    restore_rng_state(rng_by_rank[rank])
    return state


class CheckpointManager:
    def __init__(
        self,
        output_dir: str | Path,
        keep_last: int,
        protected_steps: set[int] | None = None,
    ) -> None:
        self.directory = Path(output_dir) / "checkpoints"
        self.keep_last = int(keep_last)
        self.protected_steps = set(protected_steps or ())

    def save(self, state: dict) -> Path:
        path = self.directory / f"step-{state['global_step']:09d}.pt"
        exclusive_torch_save(state, path)
        pointer = self.directory / "latest.txt"
        atomic_text_write(path.name + "\n", pointer)
        if self.keep_last > 0:
            checkpoints = sorted(self.directory.glob("step-*.pt"))
            removable = [
                item
                for item in checkpoints
                if int(item.stem.removeprefix("step-")) not in self.protected_steps
            ]
            for old in removable[: -self.keep_last]:
                old.unlink()
        return path

    def save_diagnostic(self, state: dict, reason: str) -> Path:
        safe_reason = re.sub(r"[^a-zA-Z0-9_.-]+", "-", reason).strip("-") or "diagnostic"
        directory = self.directory.parent / "diagnostics"
        path = directory / (
            f"{safe_reason}-step-{state['global_step']:09d}-micro-{state['micro_index']:012d}.pt"
        )
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing diagnostic checkpoint: {path}")
        atomic_torch_save(state, path)
        return path

    def latest(self) -> Path | None:
        pointer = self.directory / "latest.txt"
        if not pointer.is_file():
            return None
        path = self.directory / pointer.read_text(encoding="utf-8").strip()
        return path if path.is_file() else None


def _replace_checkpoint_paths(value):
    if isinstance(value, dict):
        return {
            key: (
                "${checkpoint_dir}/model.pt"
                if key == "ckpt_path"
                else "${checkpoint_dir}/stats"
                if key == "stats_path"
                else _replace_checkpoint_paths(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_checkpoint_paths(item) for item in value]
    return value


def _find_unique_mapping_value(value, key: str):
    matches = []
    if isinstance(value, dict):
        if key in value:
            matches.append(value[key])
        for item in value.values():
            matches.extend(_find_unique_mapping_value(item, key))
    elif isinstance(value, list):
        for item in value:
            matches.extend(_find_unique_mapping_value(item, key))
    return matches


def _from_scratch_inference_config(config) -> dict:
    return {
        "_target_": "kimodo.model.Kimodo",
        "denoiser": {
            "_target_": "kimodo.model.TwostageDenoiser",
            "motion_rep": {
                "_target_": "kimodo.motion_rep.KimodoMotionRep",
                "skeleton": {
                    "_target_": "kimodo.skeleton.registry.build_skeleton",
                    "nbjoints": config.model.skeleton_joints,
                },
                "fps": config.model.fps,
                "stats_path": "${checkpoint_dir}/stats",
            },
            "motion_mask_mode": config.model.motion_mask_mode,
            "ckpt_path": "${checkpoint_dir}/model.pt",
            "detach_root_for_body": config.model.detach_root_for_body,
            "llm_shape": [config.model.llm_tokens, config.model.llm_dim],
            "use_text_mask": config.model.use_text_mask,
            "latent_dim": config.model.latent_dim,
            "ff_size": config.model.ff_size,
            "num_layers": config.model.num_layers,
            "num_heads": config.model.num_heads,
            "activation": config.model.activation,
            "dropout": 0.0,
            "pe_dropout": 0.0,
            "norm_first": config.model.norm_first,
            "num_text_tokens_override": config.model.num_text_tokens_override,
            "input_first_heading_angle": config.model.input_first_heading_angle,
        },
        "text_encoder": None,
        "num_base_steps": config.model.num_diffusion_steps,
        "cfg_type": "separated",
    }


def export_inference_bundle(model, ema, output_dir: str | Path, global_step: int, config) -> Path:
    """Export EMA weights, config and stats as a strict, self-contained bundle."""
    exports = Path(output_dir) / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    destination = exports / f"step-{global_step:09d}"
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".", dir=exports))
    state = ema.shadow if ema is not None else unwrap_model(model).state_dict()
    try:
        atomic_torch_save(dict(state), temporary / "model.pt")
        if config.model.checkpoint_dir:
            source_config = Path(config.model.checkpoint_dir) / "config.yaml"
            loaded = OmegaConf.load(source_config)
            raw = OmegaConf.to_container(loaded, resolve=False)
            inference_config = _replace_checkpoint_paths(raw)
            resolved = OmegaConf.to_container(
                OmegaConf.merge(
                    loaded,
                    OmegaConf.create({"checkpoint_dir": str(Path(config.model.checkpoint_dir).resolve())}),
                ),
                resolve=True,
            )
            stats_paths = _find_unique_mapping_value(resolved, "stats_path")
            if len(set(map(str, stats_paths))) != 1:
                raise ValueError(f"Expected one stats_path in official config, got {stats_paths}")
            stats_source = Path(stats_paths[0])
        else:
            inference_config = _from_scratch_inference_config(config)
            stats_source = Path(config.model.stats_path)
        if not stats_source.is_dir():
            raise FileNotFoundError(f"Cannot export inference stats; directory is missing: {stats_source}")
        OmegaConf.save(OmegaConf.create(inference_config), temporary / "config.yaml")
        shutil.copytree(stats_source, temporary / "stats")
        (temporary / "TRAINING_PROVENANCE.txt").write_text(
            "This bundle was produced by the reconstructed trainer. See the parent run's "
            "provenance.json and config.resolved.yaml.\n",
            encoding="utf-8",
        )
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite inference bundle: {destination}")
        os.replace(temporary, destination)
        # Sidecar eval pods often run as a non-root user on the same PVC.
        for path in [destination, *destination.rglob("*")]:
            _publish_mode(path, 0o755 if path.is_dir() else 0o644)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination
