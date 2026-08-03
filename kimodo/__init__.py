# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimodo: text-driven and constrained motion generation model.

The public symbols stay unchanged, but are imported lazily so training with
precomputed text embeddings does not require PEFT/Transformers at import time.
"""

__all__ = [
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "load_model",
]


def __getattr__(name):
    if name in __all__:
        from .model.load_model import AVAILABLE_MODELS, DEFAULT_MODEL, load_model

        return {
            "AVAILABLE_MODELS": AVAILABLE_MODELS,
            "DEFAULT_MODEL": DEFAULT_MODEL,
            "load_model": load_model,
        }[name]
    raise AttributeError(name)
