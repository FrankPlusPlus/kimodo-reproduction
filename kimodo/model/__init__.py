# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimodo model package with dependency-isolated lazy public imports."""

__all__ = [
    "Kimodo",
    "LLM2VecEncoder",
    "TMR",
    "TwostageDenoiser",
    "load_model",
    "load_checkpoint_bundle",
    "load_checkpoint_state_dict",
    "resolve_target",
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "DEFAULT_TEXT_ENCODER_URL",
    "MODEL_NAMES",
]


def __getattr__(name):
    if name == "Kimodo":
        from .kimodo_model import Kimodo

        return Kimodo
    if name == "LLM2VecEncoder":
        from .llm2vec import LLM2VecEncoder

        return LLM2VecEncoder
    if name == "TMR":
        from .tmr import TMR

        return TMR
    if name == "TwostageDenoiser":
        from .twostage_denoiser import TwostageDenoiser

        return TwostageDenoiser
    if name in {"load_model", "load_checkpoint_bundle"}:
        from .load_model import load_checkpoint_bundle, load_model

        return {"load_model": load_model, "load_checkpoint_bundle": load_checkpoint_bundle}[name]
    if name == "resolve_target":
        from .common import resolve_target

        return resolve_target
    if name in {
        "AVAILABLE_MODELS",
        "DEFAULT_MODEL",
        "DEFAULT_TEXT_ENCODER_URL",
        "MODEL_NAMES",
        "load_checkpoint_state_dict",
    }:
        from . import loading

        return getattr(loading, name)
    raise AttributeError(name)
