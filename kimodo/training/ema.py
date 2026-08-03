# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exponential moving average for denoiser parameters."""

from __future__ import annotations

from collections import OrderedDict

import torch


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.995) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = OrderedDict(
            (name, value.detach().clone()) for name, value in model.state_dict().items()
        )

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise RuntimeError("EMA/model state dictionaries no longer have the same keys")
        for name, value in current.items():
            shadow_value = self.shadow[name]
            if torch.is_floating_point(shadow_value):
                shadow_value.lerp_(value.detach().to(device=shadow_value.device), 1.0 - self.decay)
            else:
                shadow_value.copy_(value.detach().to(device=shadow_value.device))
        self.num_updates += 1

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        self.shadow = OrderedDict((name, value.clone()) for name, value in state["shadow"].items())

    def to(self, device: torch.device | str) -> "ExponentialMovingAverage":
        self.shadow = OrderedDict((name, value.to(device)) for name, value in self.shadow.items())
        return self

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)
