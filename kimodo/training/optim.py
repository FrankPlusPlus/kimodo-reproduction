# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optimizers used by the reconstructed training recipe."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.optim import Optimizer


class AdamAtan2(Optimizer):
    """Adam with the scale-invariant atan2 update from Everett et al. (2024).

    The cited optimizer replaces ``m / (sqrt(v) + eps)`` with
    ``4/pi * lambda * atan2(m, lambda * sqrt(v))``. Kimodo does not disclose
    lambda, betas, or weight decay. We therefore expose lambda and use the
    reference paper's experimental value ``lambda=8`` by default.

    Weight decay, when non-zero, is applied decoupled from the gradient.  This
    reconstruction defaults it to zero; Kimodo does not disclose that value.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        atan2_lambda: float = 8.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if atan2_lambda <= 0:
            raise ValueError("atan2_lambda must be positive")
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
                atan2_lambda=atan2_lambda,
            ),
        )
        self.last_update_norm = 0.0
        self.last_update_param_ratio = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        update_sq = torch.zeros((), dtype=torch.float32)
        param_sq = torch.zeros((), dtype=torch.float32)
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            atan2_lambda = group["atan2_lambda"]

            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("AdamAtan2 does not support sparse gradients")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.lerp_(gradient, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

                bias_correction1 = 1.0 - math.pow(beta1, state["step"])
                bias_correction2 = 1.0 - math.pow(beta2, state["step"])
                first_moment = exp_avg / bias_correction1
                second_moment_root = torch.sqrt(exp_avg_sq / bias_correction2)
                update = (
                    (4.0 / math.pi)
                    * atan2_lambda
                    * torch.atan2(first_moment, atan2_lambda * second_moment_root)
                )

                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(update, alpha=-lr)
                update_sq = update_sq + (update.detach().float().pow(2).sum() * (lr * lr))
                param_sq = param_sq + parameter.detach().float().pow(2).sum()
        self.last_update_norm = float(update_sq.sqrt())
        param_norm = float(param_sq.sqrt())
        self.last_update_param_ratio = (
            self.last_update_norm / param_norm if param_norm > 0.0 else float("nan")
        )
        return loss


def scheduled_learning_rate(
    step: int,
    *,
    peak_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    warmup_start_lr: float | None = None,
    lr_end: float | None = None,
    schedule_start_step: int = 0,
) -> float:
    """Linear warmup from a fork/start step, then linear decay to ``lr_end``.

    ``step`` is the current completed-step counter (the next optimizer write
    uses this value before it is incremented).
    """
    if peak_lr <= 0:
        raise ValueError("peak_lr must be positive")
    start = max(int(schedule_start_step), 0)
    current = max(int(step), start)
    warmup = max(int(warmup_steps), 0)
    warmup_start = peak_lr if warmup_start_lr is None else float(warmup_start_lr)
    elapsed = current - start
    if warmup > 0 and elapsed < warmup:
        return warmup_start + (elapsed / warmup) * (peak_lr - warmup_start)
    if lr_end is None:
        return peak_lr
    decay_start = start + warmup
    span = int(total_steps) - decay_start
    if span <= 0:
        return float(lr_end)
    fraction = min(max((current - decay_start) / span, 0.0), 1.0)
    return peak_lr + fraction * (float(lr_end) - peak_lr)


def build_optimizer(model: torch.nn.Module, config) -> Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if config.name == "adam_atan2":
        return AdamAtan2(
            parameters,
            lr=config.learning_rate,
            betas=tuple(config.betas),
            weight_decay=config.weight_decay,
            atan2_lambda=config.atan2_lambda,
        )
    if config.name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            betas=tuple(config.betas),
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.name}")
