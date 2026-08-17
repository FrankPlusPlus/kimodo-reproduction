"""Read-only LN-σ clamp probe: does 1/σ amplify last-layer grads?

Freeze weights. Compare an unclamped backward with one where LayerNorm
divides by max(σ, floor). Floor is measured on a healthy checkpoint
(parent 750k) on the same frozen batch.

Two clamp modes:

- ``backward``: forward still uses the true σ (predictions unchanged);
  backward pretends the divisor was floored. Isolates the 1/σ path.
- ``forward``: forward and backward both use the floor. Matches a min-σ
  training patch.

No optimizer step, no weight write.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

import torch
from torch import nn

from kimodo.training.body_jacobian_probe import _named_module_grad_norms, _zero_grads, flatten_grads
from kimodo.training.body_residual_cancel_probe import ResidualCancelHooks


def _safe_ratio(value: object, baseline: object) -> float:
    try:
        numerator = float(value)  # type: ignore[arg-type]
        denominator = float(baseline)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return numerator / denominator


def clamped_layer_norm(
    module: nn.LayerNorm,
    hidden: torch.Tensor,
    *,
    sigma_floor: float,
    mode: str,
) -> torch.Tensor:
    """LayerNorm with an optional σ floor.

    ``backward`` keeps the true forward scale and routes gradients through
    ``1/max(σ, floor)``. ``forward`` uses the floor in both directions.
    """
    if mode not in {"forward", "backward"}:
        raise ValueError(f"unsupported clamp mode: {mode}")
    mean = hidden.mean(dim=-1, keepdim=True)
    var = hidden.var(dim=-1, unbiased=False, keepdim=True)
    sigma = (var + float(module.eps)).sqrt()
    inv_true = 1.0 / sigma
    inv_clamped = 1.0 / sigma.clamp_min(float(sigma_floor))
    centered = hidden - mean
    scaled_true = centered * inv_true
    scaled_clamped = centered * inv_clamped
    if mode == "forward":
        # Floor in both directions: this is a min-σ training patch.
        normalized = scaled_clamped
    else:
        # Keep the true forward value, but backprop through 1/max(σ, floor).
        # The inverse must multiply ``centered`` inside the autograd graph;
        # detaching only ``inv`` leaves dL/d(x) using the true 1/σ.
        normalized = scaled_clamped + (scaled_true - scaled_clamped).detach()
    if module.elementwise_affine:
        weight = module.weight
        bias = module.bias
        if weight is not None:
            normalized = normalized * weight
        if bias is not None:
            normalized = normalized + bias
    return normalized


@contextmanager
def patch_layer_norm_sigma(
    layer: nn.Module,
    *,
    mode: str,
    floors: dict[str, float],
) -> Iterator[None]:
    """Temporarily replace ``norm1`` / ``norm2`` forward with a floored divisor."""
    restored: list[tuple[nn.Module, Any]] = []
    try:
        for slot, floor in floors.items():
            module = getattr(layer, slot, None)
            if module is None:
                raise AttributeError(f"layer is missing {slot}")
            original = module.forward

            def _forward(
                hidden: torch.Tensor,
                module: nn.LayerNorm = module,
                sigma_floor: float = float(floor),
                clamp_mode: str = mode,
            ) -> torch.Tensor:
                return clamped_layer_norm(module, hidden, sigma_floor=sigma_floor, mode=clamp_mode)

            module.forward = _forward  # type: ignore[method-assign]
            restored.append((module, original))
        yield
    finally:
        for module, original in restored:
            module.forward = original  # type: ignore[method-assign]


def _body_layer(model: nn.Module, index: int) -> nn.Module:
    from kimodo.training.modeling import unwrap_model

    encoder = getattr(unwrap_model(model).body_model, "seqTransEncoder", None)
    if encoder is None:
        raise TypeError("body_model is missing seqTransEncoder")
    return encoder.layers[int(index)]


def _layer_cancel(layers: Sequence[dict[str, Any]], index: int) -> dict[str, Any]:
    for layer in layers:
        if int(layer.get("index", -1)) == int(index):
            return layer
    return {}


def probe_sigma_clamp(
    model: nn.Module,
    *,
    noisy: torch.Tensor,
    valid_frames: torch.Tensor,
    text_features: torch.Tensor,
    text_pad_mask: torch.Tensor,
    timesteps: torch.Tensor,
    first_heading_angle: torch.Tensor,
    motion_mask: torch.Tensor,
    observed_motion: torch.Tensor,
    target: torch.Tensor,
    loss_function,
    layer_index: int = 15,
    clamp_mode: str = "none",
    sigma_floors: dict[str, float] | None = None,
    watch_layers: Sequence[int] = (14, 15),
) -> dict[str, Any]:
    """One training-mode forward + backward, optionally with L15 σ clamp."""
    from kimodo.training.modeling import set_model_dropout, unwrap_model

    bare = unwrap_model(model)
    set_model_dropout(bare, 0.0)
    bare.train()
    _zero_grads(bare)
    body = bare.body_model
    layer = _body_layer(bare, layer_index)
    floors = dict(sigma_floors or {})
    context: Any = (
        patch_layer_norm_sigma(layer, mode=clamp_mode, floors=floors)
        if clamp_mode in {"forward", "backward"} and floors
        else nullcontext()
    )
    with ResidualCancelHooks(body, watch_layers) as hooks, context:
        prediction = bare(
            noisy,
            valid_frames,
            text_features,
            text_pad_mask,
            timesteps,
            first_heading_angle=first_heading_angle,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
        )
        losses = loss_function(prediction, target, valid_frames)
        total = losses.frame_sums["total"]
        (prediction_grad,) = torch.autograd.grad(total, prediction, retain_graph=True, create_graph=False)
        total.backward()
    cancel = hooks.summarize()
    last = _layer_cancel(cancel, layer_index)
    ffn = last.get("ffn") if isinstance(last.get("ffn"), dict) else {}
    batch_body = flatten_grads(body.parameters())
    report = {
        "clamp_mode": clamp_mode,
        "sigma_floors": floors,
        "layer_index": int(layer_index),
        "loss_total_mean": float(losses.means["total"].detach()),
        "prediction_grad_norm": float(prediction_grad.detach().float().norm()),
        "body_batch_grad_norm": float(batch_body.norm()) if batch_body.numel() else float("nan"),
        "grad_norms": _named_module_grad_norms(body, "body"),
        "attn_cosine": last.get("mean_token_cosine"),
        "attn_ln_sigma": last.get("ln_sigma_mean"),
        "attn_negative_fraction": last.get("negative_cosine_fraction"),
        "ffn_cosine": ffn.get("mean_token_cosine"),
        "ffn_ln_sigma": ffn.get("ln_sigma_mean"),
        "ffn_negative_fraction": ffn.get("negative_cosine_fraction"),
        "layers": cancel,
    }
    _zero_grads(bare)
    return report


def summarize_sigma_clamp(
    rows: Sequence[dict[str, Any]],
    *,
    takeoff_step: int = 800000,
    healthy_step: int = 750000,
    drop_cut: float = 0.75,
    still_hot: float = 1.2,
) -> dict[str, Any]:
    """Compare takeoff baseline vs backward-clamp vs healthy baseline."""

    def _row(step: int, mode: str, norms: Sequence[str] | None = None) -> dict[str, Any] | None:
        wanted = tuple(sorted(norms)) if norms is not None else None
        for row in rows:
            if int(row.get("global_step") or -1) != int(step):
                continue
            if str(row.get("clamp_mode")) != mode:
                continue
            floors = row.get("sigma_floors") or {}
            got = tuple(sorted(floors))
            if wanted is not None and got != wanted:
                continue
            return row
        return None

    healthy = _row(healthy_step, "none")
    takeoff = _row(takeoff_step, "none")
    backward_ln1 = _row(takeoff_step, "backward", ("norm1",))
    backward_both = _row(takeoff_step, "backward", ("norm1", "norm2"))
    if backward_both is None:
        backward_both = _row(takeoff_step, "backward")
    forward_both = _row(takeoff_step, "forward")

    def _attn_grad(row: dict[str, Any] | None) -> float:
        if row is None:
            return float("nan")
        probe = row.get("probe") or row
        grads = probe.get("grad_norms") or {}
        value = grads.get("body.layer_15.self_attn")
        if value is None:
            value = probe.get("body_batch_grad_norm")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    takeoff_grad = _attn_grad(takeoff)
    healthy_grad = _attn_grad(healthy)
    backward_grad = _attn_grad(backward_both) if backward_both is not None else _attn_grad(backward_ln1)
    used = "backward_ln1_ln2" if backward_both is not None else "backward_ln1"
    drop = _safe_ratio(backward_grad, takeoff_grad)
    still = _safe_ratio(backward_grad, healthy_grad)
    takeoff_over_healthy = _safe_ratio(takeoff_grad, healthy_grad)
    if math.isfinite(drop) and 0.98 <= drop <= 1.02:
        verdict = "clamp_noop"
    elif math.isfinite(drop) and drop <= drop_cut:
        verdict = "sigma_amplifies_grads"
    elif math.isfinite(still) and still <= still_hot:
        verdict = "sigma_returns_to_healthy"
    elif math.isfinite(takeoff_over_healthy) and takeoff_over_healthy > still_hot and (
        not math.isfinite(drop) or drop > drop_cut
    ):
        verdict = "not_sigma"
    else:
        verdict = "incomplete"
    return {
        "verdict": verdict,
        "clamp_used": used,
        "takeoff_over_healthy_attn_grad": takeoff_over_healthy,
        "backward_over_takeoff_attn_grad": drop,
        "backward_over_healthy_attn_grad": still,
        "takeoff_ffn_cosine": ((takeoff or {}).get("probe") or takeoff or {}).get("ffn_cosine"),
        "healthy_ffn_cosine": ((healthy or {}).get("probe") or healthy or {}).get("ffn_cosine"),
        "forward_loss_ratio": _safe_ratio(
            ((forward_both or {}).get("probe") or forward_both or {}).get("loss_total_mean"),
            ((takeoff or {}).get("probe") or takeoff or {}).get("loss_total_mean"),
        ),
    }
