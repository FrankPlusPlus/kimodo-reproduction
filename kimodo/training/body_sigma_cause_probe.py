"""Causal interventions on what makes L15 pre-LN σ small and L15 grads large.

Two independent levers, applied one at a time on a frozen batch:

- Attn **scale** (r = attn_rms / residual_rms): multiply L15 in_proj or the
  attn output. Cosine is held (direction unchanged).
- Attn **direction** (cosine): rewrite attn tokens to a target cosine with
  the same length. r is held.

σ ≈ RMS(x + attn). Official SEED can sit at slightly negative cosine with
small r and healthy σ. Collapse needs small σ, which this probe attributes
to scale, direction, or both — not to whichever number moved in the logs.

No optimizer step. Weights are restored after each intervention.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn

from kimodo.training.body_jacobian_probe import _named_module_grad_norms, _zero_grads
from kimodo.training.body_residual_cancel_probe import ResidualCancelHooks, residual_pair_stats


def _f(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _safe_ratio(value: object, baseline: object) -> float:
    num, den = _f(value), _f(baseline)
    if den == 0.0 or not math.isfinite(num) or not math.isfinite(den):
        return float("nan")
    return num / den


def set_branch_cosine(residual: torch.Tensor, branch: torch.Tensor, target_cos: float) -> torch.Tensor:
    """Same ||branch||, cosine with residual set to ``target_cos``."""
    stream = residual.detach()
    vector = branch.detach()
    unit_x = stream / stream.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    parallel = (vector * unit_x).sum(dim=-1, keepdim=True) * unit_x
    perp = vector - parallel
    unit_perp = perp / perp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cosine = max(-1.0, min(1.0, float(target_cos)))
    if abs(cosine) >= 0.999:
        return (1.0 if cosine > 0 else -1.0) * unit_x * scale
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return (cosine * unit_x + sine * unit_perp) * scale


def _l15(model: nn.Module) -> nn.Module:
    from kimodo.training.modeling import unwrap_model

    encoder = unwrap_model(model).body_model.seqTransEncoder
    return encoder.layers[15]


def snapshot_l15_attn(model: nn.Module) -> dict[str, torch.Tensor]:
    layer = _l15(model)
    payload = {}
    for name, parameter in layer.self_attn.named_parameters():
        payload[name] = parameter.detach().clone()
    return payload


def restore_l15_attn(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    layer = _l15(model)
    with torch.no_grad():
        for name, parameter in layer.self_attn.named_parameters():
            parameter.copy_(snapshot[name])


def scale_l15_in_proj_(model: nn.Module, factor: float) -> None:
    layer = _l15(model)
    with torch.no_grad():
        layer.self_attn.in_proj_weight.mul_(float(factor))
        bias = getattr(layer.self_attn, "in_proj_bias", None)
        if isinstance(bias, torch.Tensor):
            bias.mul_(float(factor))


@contextmanager
def patch_attn_output(
    layer: nn.Module,
    *,
    scale: float = 1.0,
    target_cosine: float | None = None,
) -> Iterator[None]:
    residual_slot: dict[str, torch.Tensor] = {}

    def _layer_in(_module, inputs) -> None:
        residual_slot["x"] = inputs[0]

    def _attn_out(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        residual = residual_slot["x"]
        patched = tensor
        if target_cosine is not None:
            patched = set_branch_cosine(residual, patched, float(target_cosine)).to(
                dtype=tensor.dtype, device=tensor.device
            )
        if scale != 1.0:
            patched = patched * float(scale)
        if isinstance(output, tuple):
            return (patched,) + tuple(output[1:])
        return patched

    handles = [
        layer.register_forward_pre_hook(_layer_in),
        layer.self_attn.register_forward_hook(_attn_out),
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def measure_l15_sigma_and_grad(
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
    attn_scale: float = 1.0,
    target_cosine: float | None = None,
) -> dict[str, Any]:
    from kimodo.training.body_onset_path_probe import _call_denoiser
    from kimodo.training.modeling import set_model_dropout, unwrap_model

    bare = unwrap_model(model)
    set_model_dropout(bare, 0.0)
    bare.train()
    layer = _l15(bare)
    kwargs = dict(
        noisy=noisy,
        valid_frames=valid_frames,
        text_features=text_features,
        text_pad_mask=text_pad_mask,
        timesteps=timesteps,
        first_heading_angle=first_heading_angle,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
    )
    with patch_attn_output(layer, scale=attn_scale, target_cosine=target_cosine):
        with ResidualCancelHooks(bare.body_model, (15,)) as hooks:
            with torch.no_grad():
                _call_denoiser(bare, **kwargs)
        stats = (hooks.summarize() or [{}])[0]
        _zero_grads(bare)
        prediction = _call_denoiser(bare, **kwargs)
        losses = loss_function(prediction, target, valid_frames)
        losses.frame_sums["total"].backward()
        grads = _named_module_grad_norms(bare.body_model, "body")
        _zero_grads(bare)
    residual = _f(stats.get("residual_rms"))
    attn = _f(stats.get("attn_rms"))
    return {
        "ln_sigma_mean": stats.get("ln_sigma_mean"),
        "residual_rms": residual,
        "attn_rms": attn,
        "r_attn_over_res": _safe_ratio(attn, residual),
        "mean_token_cosine": stats.get("mean_token_cosine"),
        "sum_over_residual_rms": stats.get("sum_over_residual_rms"),
        "l15_attn_grad": grads.get("body.layer_15.self_attn"),
        "l15_grad": grads.get("body.layer_15"),
        "body_grad": grads.get("body.all"),
        "loss_total_mean": _f(losses.means["total"].detach()),
    }


def summarize_sigma_cause(
    rows: list[dict[str, Any]],
    *,
    sigma_cut: float = 0.12,
    grad_cut: float = 0.55,
) -> dict[str, Any]:
    """Which single lever moves 790k σ/grad back toward the healthy row."""
    by_name = {str(row.get("name")): row for row in rows}
    crashed = by_name.get("crashed_identity") or {}
    healthy = by_name.get("healthy_identity") or {}
    crashed_sigma = _f(crashed.get("ln_sigma_mean"))
    healthy_sigma = _f(healthy.get("ln_sigma_mean"))
    crashed_grad = _f(crashed.get("l15_attn_grad"))
    healthy_grad = _f(healthy.get("l15_attn_grad"))
    gap_sigma = crashed_sigma - healthy_sigma if math.isfinite(crashed_sigma) and math.isfinite(healthy_sigma) else float("nan")
    gap_grad = crashed_grad - healthy_grad if math.isfinite(crashed_grad) and math.isfinite(healthy_grad) else float("nan")

    def recovery(name: str) -> dict[str, float]:
        row = by_name.get(name) or {}
        sigma = _f(row.get("ln_sigma_mean"))
        grad = _f(row.get("l15_attn_grad"))
        sigma_rec = _safe_ratio(crashed_sigma - sigma, gap_sigma)
        grad_rec = _safe_ratio(crashed_grad - grad, gap_grad)
        return {"sigma_recovery": sigma_rec, "grad_recovery": grad_rec}

    scale = recovery("crashed_attn_scale_to_healthy")
    direction = recovery("crashed_cosine_to_healthy")
    scale_up = recovery("healthy_attn_scale_to_crashed")
    direction_down = recovery("healthy_cosine_to_crashed")

    def wins(left: dict[str, float], right: dict[str, float]) -> bool:
        return _f(left.get("sigma_recovery")) >= _f(right.get("sigma_recovery")) + 0.15

    if not math.isfinite(gap_sigma):
        verdict = "incomplete"
        config_hint = "incomplete"
    elif _f(scale.get("sigma_recovery")) >= 0.6 and wins(scale, direction):
        verdict = "attn_scale_causes_sigma_collapse"
        config_hint = "weight_decay_or_stop_before_attn_rms_growth"
    elif _f(direction.get("sigma_recovery")) >= 0.6 and wins(direction, scale):
        verdict = "attn_direction_causes_sigma_collapse"
        config_hint = "not_uniquely_a_filled_knob"
    elif _f(scale.get("sigma_recovery")) >= 0.4 and _f(direction.get("sigma_recovery")) >= 0.4:
        verdict = "scale_and_direction_both_required"
        config_hint = "weight_decay_caps_scale_direction_is_attractor"
    else:
        verdict = "neither_lever_recovers"
        config_hint = "look_upstream_of_l15_attn"
    return {
        "verdict": verdict,
        "config_hint": config_hint,
        "crashed_sigma": crashed_sigma,
        "healthy_sigma": healthy_sigma,
        "crashed_l15_attn_grad": crashed_grad,
        "healthy_l15_attn_grad": healthy_grad,
        "scale_recovery": scale,
        "direction_recovery": direction,
        "healthy_scale_up": scale_up,
        "healthy_direction_to_crash": direction_down,
        "sigma_cut": sigma_cut,
        "grad_cut": grad_cut,
    }
