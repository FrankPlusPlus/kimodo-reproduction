# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paper-aligned training utilities for Kimodo.

The upstream release contains the trainable denoiser and diffusion primitives but
does not ship the training loop.  This package reconstructs that missing surface
while keeping the released inference modules unchanged.
"""

from .config import TrainingConfig, load_training_config
from .constraints import ConstraintCurriculumSampler
from .ema import ExponentialMovingAverage
from .losses import KimodoLoss
from .optim import AdamAtan2

__all__ = [
    "AdamAtan2",
    "ConstraintCurriculumSampler",
    "ExponentialMovingAverage",
    "KimodoLoss",
    "TrainingConfig",
    "load_training_config",
]
