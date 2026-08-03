# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construct trainable denoisers without changing the released inference API."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from kimodo.model.loading import instantiate_from_dict, load_checkpoint_state_dict
from kimodo.model.twostage_denoiser import TwostageDenoiser
from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def set_model_dropout(model: torch.nn.Module, probability: float) -> None:
    for module in unwrap_model(model).modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = float(probability)
        elif isinstance(module, torch.nn.MultiheadAttention):
            # PyTorch stores attention-probability dropout as a float rather
            # than an nn.Dropout submodule.
            module.dropout = float(probability)


def canonical_denoiser_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Accept only known checkpoint namespaces; never silently drop keys."""
    keys = list(state)
    if not keys:
        raise ValueError("Checkpoint state dict is empty")
    if all(key.startswith("denoiser.backbone.") for key in keys):
        return {key.removeprefix("denoiser.backbone."): value for key, value in state.items()}
    if all(key.startswith("root_model.") or key.startswith("body_model.") for key in keys):
        return state
    raise ValueError(
        "Unsupported or mixed denoiser checkpoint namespace. Expected raw root_model/body_model "
        "keys or the legacy denoiser.backbone.* prefix."
    )


def _find_weights(checkpoint_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in checkpoint_dir.iterdir()
        if path.suffix in {".pt", ".pth", ".ckpt", ".safetensors"} and path.is_file()
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one checkpoint weights file in {checkpoint_dir}, found: "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]


def load_official_trainable_denoiser(checkpoint_dir: str | Path, device) -> TwostageDenoiser:
    """Instantiate an official inference bundle, then extract its bare denoiser."""
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    config_path = checkpoint_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Official bundle is missing config.yaml: {config_path}")
    model_conf = OmegaConf.load(config_path)
    runtime = OmegaConf.create({"checkpoint_dir": str(checkpoint_dir)})
    resolved = OmegaConf.to_container(OmegaConf.merge(model_conf, runtime), resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Official config must resolve to a mapping")
    resolved.pop("checkpoint_dir", None)
    # Never instantiate or train the external text encoder here.
    resolved["text_encoder"] = None
    wrapper = instantiate_from_dict(resolved, overrides={"device": str(device)})
    guided = wrapper.denoiser
    denoiser = guided.model if hasattr(guided, "model") else guided
    if not isinstance(denoiser, TwostageDenoiser):
        raise TypeError(f"Expected TwostageDenoiser in official bundle, got {type(denoiser)}")
    return denoiser


def build_trainable_denoiser(config, curriculum_config, device) -> TwostageDenoiser:
    if config.checkpoint_dir:
        return load_official_trainable_denoiser(config.checkpoint_dir, device)

    skeleton = build_skeleton(config.skeleton_joints)
    motion_rep = KimodoMotionRep(skeleton=skeleton, fps=config.fps, stats_path=config.stats_path)
    denoiser = TwostageDenoiser(
        motion_rep=motion_rep,
        motion_mask_mode=config.motion_mask_mode,
        llm_shape=[config.llm_tokens, config.llm_dim],
        use_text_mask=config.use_text_mask,
        latent_dim=config.latent_dim,
        ff_size=config.ff_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        activation=config.activation,
        dropout=curriculum_config.phase1_dropout,
        pe_dropout=curriculum_config.phase1_dropout,
        norm_first=config.norm_first,
        num_text_tokens_override=config.num_text_tokens_override,
        input_first_heading_angle=config.input_first_heading_angle,
        detach_root_for_body=config.detach_root_for_body,
    )
    if config.checkpoint_weights:
        state = canonical_denoiser_state_dict(load_checkpoint_state_dict(config.checkpoint_weights))
        denoiser.load_state_dict(state, strict=True)
    return denoiser.to(device)


def validate_model_contract(denoiser: TwostageDenoiser, config) -> None:
    motion_rep = denoiser.motion_rep
    if motion_rep.skeleton.nbjoints != config.skeleton_joints:
        raise ValueError("Constructed skeleton joint count differs from config")
    if motion_rep.fps != config.fps:
        raise ValueError(f"Checkpoint fps={motion_rep.fps} does not match training fps={config.fps}")
    if denoiser.motion_mask_mode != "concat":
        raise ValueError("Checkpoint is not compatible with paper constraint-mask concatenation")
    # Official bundles predate this reconstruction option; apply the requested
    # training policy after strict checkpoint instantiation without affecting
    # any parameter names or values.
    denoiser.detach_root_for_body = bool(config.detach_root_for_body)
    for stage in (denoiser.root_model, denoiser.body_model):
        if int(stage.llm_shape[-1]) != config.llm_dim:
            raise ValueError(
                f"Checkpoint text width={stage.llm_shape[-1]} does not match configured llm_dim={config.llm_dim}"
            )
