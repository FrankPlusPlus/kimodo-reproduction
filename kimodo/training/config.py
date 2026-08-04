# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated configuration for the reconstructed Kimodo trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf


@dataclass
class DataConfig:
    manifest: str = ""
    split: str = "train"
    fps: int = 30
    max_seconds: float = 10.0
    min_frames: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = False
    require_cached_text: bool = True
    # ``full`` hashes every referenced motion/embedding at trainer startup and
    # is suitable only for small fixtures. ``inventory`` trusts a previously
    # built, content-addressed inventory and hashes only the manifest,
    # inventory, and inventory metadata at startup.
    reference_verification: str = "full"
    reference_inventory: str | None = None
    # Fail closed unless the manifest proves that the paper-described Qwen
    # paraphrases and diffusion-generated cross-motion transitions are present.
    # Keep this false for synthetic/engineering smoke runs; the paper-aligned
    # production profile enables it explicitly.
    require_paper_data_parity: bool = False


@dataclass
class ModelConfig:
    # For an official checkpoint, point this at a folder containing config.yaml,
    # checkpoint weights, and split normalization stats. Architecture fields are
    # then read from that official config rather than the defaults below.
    checkpoint_dir: str | None = None
    checkpoint_weights: str | None = None
    skeleton_joints: int = 30
    stats_path: str = ""
    fps: int = 30
    num_diffusion_steps: int = 1000
    motion_mask_mode: str = "concat"
    llm_dim: int = 4096
    llm_tokens: int = 1
    num_text_tokens_override: int = 50
    latent_dim: int = 1024
    ff_size: int = 2048
    num_layers: int = 16
    num_heads: int = 8
    activation: str = "gelu"
    norm_first: bool = False
    input_first_heading_angle: bool = True
    use_text_mask: bool = False
    # The released implementation jointly trains both stages but explicitly
    # detaches the root-to-body conversion in training mode.  Allowing body loss
    # through the bridge remains available as a gradient-coupled ablation.
    detach_root_for_body: bool = True


@dataclass
class CurriculumConfig:
    phase1_steps: int = 500_000
    phase2_steps: int = 500_000
    phase1_dropout: float = 0.1
    phase2_dropout: float = 0.0
    text_dropout_probability: float = 0.1
    no_constraint_probability: float = 0.1
    mix_two_probability: float = 0.25
    sparse_keyframes_min: int = 1
    sparse_keyframes_max: int = 20
    sparse_count_power: float = 1.0
    dense_path_min_fraction: float = 0.2
    dense_path_max_fraction: float = 0.8
    root_heading_probability: float = 0.5


@dataclass
class LossConfig:
    # The paper does not disclose the domain of the six direct feature losses.
    # This explicit switch keeps the default reproducible and makes the
    # physical-domain alternative available as an ablation.
    direct_feature_domain: str = "physical"
    smooth_l1_beta: float = 1.0
    root_position: float = 10.0
    root_heading: float = 2.0
    joint_position: float = 10.0
    joint_velocity: float = 3.0
    joint_rotation: float = 10.0
    foot_contact: float = 4.0
    forward_kinematics: float = 5.0


@dataclass
class OptimizerConfig:
    name: str = "adam_atan2"
    learning_rate: float = 2e-5
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    atan2_lambda: float = 8.0
    gradient_clip_norm: float | None = 1.0


@dataclass
class EMAConfig:
    enabled: bool = True
    decay: float = 0.995
    update_every: int = 10


@dataclass
class RuntimeConfig:
    output_dir: str = "outputs/kimodo-train"
    seed: int = 1234
    device: str = "auto"
    precision: str = "bf16"
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    log_every: int = 10
    checkpoint_every: int = 10_000
    milestone_every: int = 100_000
    keep_last_checkpoints: int = 3
    resume: str | None = None
    initial_global_step: int = 0
    distributed: str = "auto"
    # Keep method parity independent from the disclosed 16-GPU/global-batch
    # scale. Two-GPU reconstructions turn only this gate off.
    enforce_paper_scale: bool = True
    dry_run: bool = False
    max_steps_override: int | None = None


@dataclass
class TrainingConfig:
    schema_version: int = 1
    # Reject overrides that contradict values or semantics stated explicitly in
    # the paper. Unknown recipe choices remain configurable and documented.
    paper_method_strict: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def total_steps(self) -> int:
        if self.runtime.max_steps_override is not None:
            return self.runtime.max_steps_override
        return self.curriculum.phase1_steps + self.curriculum.phase2_steps

    def validate(self, *, require_paths: bool = True) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported training config schema_version={self.schema_version}")
        if self.data.fps <= 0 or self.model.fps <= 0:
            raise ValueError("data.fps and model.fps must be positive")
        if self.data.fps != self.model.fps:
            raise ValueError("data.fps must match model.fps; resampling belongs in preprocessing")
        if self.data.max_seconds <= 0 or self.data.min_frames < 2:
            raise ValueError("data.max_seconds must be positive and data.min_frames must be >= 2")
        if self.data.persistent_workers:
            raise ValueError(
                "persistent_workers is disabled: worker dataset copies would not receive set_epoch updates"
            )
        if self.data.reference_verification not in {"full", "inventory"}:
            raise ValueError("data.reference_verification must be 'full' or 'inventory'")
        if (
            require_paths
            and self.data.reference_verification == "inventory"
            and not self.data.reference_inventory
        ):
            raise ValueError(
                "data.reference_inventory is required when reference_verification='inventory'"
            )
        if self.curriculum.phase1_steps < 0 or self.curriculum.phase2_steps < 0:
            raise ValueError("phase step counts must be non-negative")
        if self.total_steps <= 0:
            raise ValueError("total training steps must be positive")
        for name in (
            "phase1_dropout",
            "phase2_dropout",
            "text_dropout_probability",
            "no_constraint_probability",
            "mix_two_probability",
            "root_heading_probability",
        ):
            value = getattr(self.curriculum, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"curriculum.{name} must be in [0, 1]")
        if self.curriculum.no_constraint_probability + self.curriculum.mix_two_probability > 1.0:
            raise ValueError("no_constraint_probability + mix_two_probability must not exceed 1")
        if self.curriculum.sparse_keyframes_min < 1:
            raise ValueError("sparse_keyframes_min must be at least 1")
        if self.curriculum.sparse_keyframes_max < self.curriculum.sparse_keyframes_min:
            raise ValueError("sparse_keyframes_max must be >= sparse_keyframes_min")
        if self.curriculum.sparse_count_power <= 0:
            raise ValueError("sparse_count_power must be positive")
        if not (
            0.0 < self.curriculum.dense_path_min_fraction
            <= self.curriculum.dense_path_max_fraction
            <= 1.0
        ):
            raise ValueError("dense path fractions must satisfy 0 < min <= max <= 1")
        if self.model.num_diffusion_steps != 1000:
            raise ValueError("The paper-aligned production configuration uses exactly 1000 diffusion steps")
        if self.model.motion_mask_mode != "concat":
            raise ValueError("Paper-aligned constraint conditioning requires motion_mask_mode='concat'")
        if self.loss.direct_feature_domain not in {"normalized", "physical"}:
            raise ValueError("loss.direct_feature_domain must be 'normalized' or 'physical'")
        if self.runtime.batch_size < 1 or self.runtime.gradient_accumulation_steps < 1:
            raise ValueError("batch and gradient accumulation sizes must be positive")
        if (
            self.runtime.log_every < 1
            or self.runtime.checkpoint_every < 1
            or self.runtime.milestone_every < 0
        ):
            raise ValueError("log/checkpoint intervals must be positive and milestone_every non-negative")
        if self.runtime.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("runtime.precision must be fp32, bf16, or fp16")
        if not 0 <= self.runtime.initial_global_step < self.total_steps:
            raise ValueError("initial_global_step must be in [0, total_steps)")
        if self.optimizer.name not in {"adam_atan2", "adamw"}:
            raise ValueError("optimizer.name must be 'adam_atan2' or 'adamw'")
        if self.optimizer.learning_rate <= 0:
            raise ValueError("optimizer.learning_rate must be positive")
        if self.optimizer.atan2_lambda <= 0:
            raise ValueError("optimizer.atan2_lambda must be positive")
        if self.loss.smooth_l1_beta <= 0:
            raise ValueError("loss.smooth_l1_beta must be positive")
        if self.ema.enabled and (not 0.0 < self.ema.decay < 1.0 or self.ema.update_every < 1):
            raise ValueError("EMA decay must be in (0,1) and update_every positive")
        if self.paper_method_strict:
            explicit_values = {
                "data.fps": (self.data.fps, 30),
                "data.max_seconds": (self.data.max_seconds, 10.0),
                "data.require_paper_data_parity": (self.data.require_paper_data_parity, True),
                "model.latent_dim": (self.model.latent_dim, 1024),
                "model.num_layers": (self.model.num_layers, 16),
                "model.num_heads": (self.model.num_heads, 8),
                "model.llm_dim": (self.model.llm_dim, 4096),
                "model.num_text_tokens_override": (self.model.num_text_tokens_override, 50),
                "model.input_first_heading_angle": (self.model.input_first_heading_angle, True),
                "curriculum.phase1_steps": (self.curriculum.phase1_steps, 500_000),
                "curriculum.phase2_steps": (self.curriculum.phase2_steps, 500_000),
                "curriculum.phase1_dropout": (self.curriculum.phase1_dropout, 0.1),
                "curriculum.phase2_dropout": (self.curriculum.phase2_dropout, 0.0),
                "curriculum.text_dropout_probability": (
                    self.curriculum.text_dropout_probability,
                    0.1,
                ),
                "curriculum.no_constraint_probability": (
                    self.curriculum.no_constraint_probability,
                    0.1,
                ),
                "curriculum.mix_two_probability": (self.curriculum.mix_two_probability, 0.25),
                "curriculum.sparse_keyframes_min": (self.curriculum.sparse_keyframes_min, 1),
                "curriculum.sparse_keyframes_max": (self.curriculum.sparse_keyframes_max, 20),
                "loss.root_position": (self.loss.root_position, 10.0),
                "loss.root_heading": (self.loss.root_heading, 2.0),
                "loss.joint_position": (self.loss.joint_position, 10.0),
                "loss.joint_velocity": (self.loss.joint_velocity, 3.0),
                "loss.joint_rotation": (self.loss.joint_rotation, 10.0),
                "loss.foot_contact": (self.loss.foot_contact, 4.0),
                "loss.forward_kinematics": (self.loss.forward_kinematics, 5.0),
                "optimizer.name": (self.optimizer.name, "adam_atan2"),
                "optimizer.learning_rate": (self.optimizer.learning_rate, 2e-5),
                "ema.enabled": (self.ema.enabled, True),
                "ema.decay": (self.ema.decay, 0.995),
                "ema.update_every": (self.ema.update_every, 10),
                "runtime.max_steps_override": (self.runtime.max_steps_override, None),
            }
            mismatches = [
                f"{name}={actual!r} (paper requires {expected!r})"
                for name, (actual, expected) in explicit_values.items()
                if actual != expected
            ]
            if mismatches:
                raise ValueError(
                    "paper_method_strict rejects explicit-paper deviations: "
                    + "; ".join(mismatches)
                )
        if require_paths:
            if not self.data.manifest or not Path(self.data.manifest).is_file():
                raise FileNotFoundError(f"Training manifest does not exist: {self.data.manifest!r}")
            if self.data.reference_verification == "inventory" and not Path(
                str(self.data.reference_inventory)
            ).is_file():
                raise FileNotFoundError(
                    "Reference inventory does not exist: "
                    f"{self.data.reference_inventory!r}. Build and fully verify it before training."
                )
            if self.data.reference_verification == "inventory":
                inventory_metadata = Path(
                    str(self.data.reference_inventory) + ".metadata.json"
                )
                if not inventory_metadata.is_file():
                    raise FileNotFoundError(
                        f"Reference inventory metadata does not exist: {inventory_metadata}"
                    )
            if self.model.checkpoint_dir is None and (
                not self.model.stats_path or not Path(self.model.stats_path).is_dir()
            ):
                raise FileNotFoundError(
                    "model.stats_path must point to split normalization stats for from-scratch training"
                )
            if self.model.checkpoint_dir is not None and not Path(self.model.checkpoint_dir).is_dir():
                raise FileNotFoundError(f"checkpoint_dir does not exist: {self.model.checkpoint_dir}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PATH_OVERLAY_FIELDS = frozenset(
    {
        "data.manifest",
        "data.reference_inventory",
        "model.checkpoint_dir",
        "model.checkpoint_weights",
        "model.stats_path",
        "runtime.output_dir",
        "runtime.resume",
    }
)


def _load_mapping(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} does not exist: {source}")
    loaded = OmegaConf.load(source)
    plain = OmegaConf.to_container(loaded, resolve=False)
    if not isinstance(plain, dict):
        raise TypeError(f"{label} must contain a top-level mapping")
    return plain


def _leaf_paths(value: dict[str, Any], prefix: str = "") -> Iterable[str]:
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            yield from _leaf_paths(item, name)
        else:
            yield name


def _load_paths_overlay(path: str | Path) -> dict[str, Any]:
    loaded = _load_mapping(path, label="paths YAML")
    schema_version = loaded.pop("schema_version", 1)
    if schema_version != 1:
        raise ValueError(f"Unsupported paths YAML schema_version={schema_version!r}")
    supplied = set(_leaf_paths(loaded))
    disallowed = sorted(supplied - _PATH_OVERLAY_FIELDS)
    if disallowed:
        raise ValueError(
            "paths YAML may contain only training resource/output paths; disallowed fields: "
            + ", ".join(disallowed)
        )
    return loaded


def load_training_config(
    path: str | Path,
    overrides: list[str] | None = None,
    *,
    paths: str | Path | None = None,
    overlays: Iterable[str | Path] | None = None,
) -> TrainingConfig:
    """Load base -> paths -> overlays -> dot-list using strict structured merging."""
    base = OmegaConf.structured(TrainingConfig)
    loaded = OmegaConf.load(path)
    layers: list[Any] = [base, loaded]
    if paths is not None:
        layers.append(OmegaConf.create(_load_paths_overlay(paths)))
    for overlay in overlays or ():
        layers.append(OmegaConf.create(_load_mapping(overlay, label="training overlay")))
    merged = OmegaConf.merge(*layers)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_struct(merged, True)
    obj = OmegaConf.to_object(merged)
    if not isinstance(obj, TrainingConfig):
        raise TypeError("Resolved config did not produce a TrainingConfig")
    obj.validate(require_paths=not obj.runtime.dry_run)
    return obj


def save_resolved_config(config: TrainingConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config.to_dict()), path)
