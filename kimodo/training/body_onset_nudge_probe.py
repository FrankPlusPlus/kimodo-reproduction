"""Read-only probe: at onset, does the 7-term loss reward last-layer cancel?

Freeze weights. On a frozen batch, rotate last-layer ``attn(x)`` (or FFN)
so the token-wise cosine with residual ``x`` moves by a small delta, keeping
branch magnitude. Compare the seven-term training loss.

Use this on the *pre-flip* weights (kf-smooth 690k, cosine still +0.22), not
on already-dead 800k. Experiment 2 already answered lock-in after collapse.

- More-cancel lowers loss, more-add does not: the objective is teaching the
  flip. Later trains need an objective change or a cancel regularizer.
- More-cancel raises loss: the 7-term loss is not the teacher. post-norm
  1/σ plus Adam-atan2 is the remaining onset hypothesis.
- Both nudges barely move loss: indifferent at onset; do not pick a 16-GPU
  recipe from this probe.

Forward only. No optimizer step, no weight write.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn

from kimodo.training.body_residual_cancel_probe import residual_pair_stats


def _safe_ratio(value: object, baseline: object) -> float:
    try:
        numerator = float(value)  # type: ignore[arg-type]
        denominator = float(baseline)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return numerator / denominator


def _unit_orthogonal(residual: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    """Per-token unit vector in the plane of ``branch``, orthogonal to ``x``."""
    stream_norm = residual.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    unit_x = residual / stream_norm
    parallel = (branch * unit_x).sum(dim=-1, keepdim=True) * unit_x
    ortho = branch - parallel
    ortho_norm = ortho.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(residual)
    fallback[..., 0] = 1.0
    fallback = fallback - (fallback * unit_x).sum(dim=-1, keepdim=True) * unit_x
    fallback_norm = fallback.norm(dim=-1, keepdim=True)
    fallback2 = torch.zeros_like(residual)
    fallback2[..., 1] = 1.0
    fallback2 = fallback2 - (fallback2 * unit_x).sum(dim=-1, keepdim=True) * unit_x
    fallback2_norm = fallback2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    use_e1 = fallback_norm.squeeze(-1) < 1e-6
    fallback = torch.where(
        use_e1.unsqueeze(-1),
        fallback2 / fallback2_norm,
        fallback / fallback_norm.clamp_min(1e-8),
    )
    small = ortho_norm.squeeze(-1) < 1e-8
    return torch.where(small.unsqueeze(-1), fallback, ortho / ortho_norm.clamp_min(1e-8))


def shift_branch_cosine(
    residual: torch.Tensor,
    branch: torch.Tensor,
    cosine_delta: float,
) -> torch.Tensor:
    """Move token-wise cosine(x, branch) by ``cosine_delta``, keep ||branch||.

    Negative delta rotates toward anti-alignment (more cancellation).
    ``cosine_delta=0`` is identity.
    """
    if residual.shape != branch.shape:
        raise ValueError(
            f"shape mismatch: residual {tuple(residual.shape)} vs branch {tuple(branch.shape)}"
        )
    if cosine_delta == 0.0:
        return branch
    branch_norm = branch.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    stream_norm = residual.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    unit_x = residual / stream_norm
    cosine = (branch * unit_x).sum(dim=-1, keepdim=True) / branch_norm
    target = (cosine + float(cosine_delta)).clamp(-1.0, 1.0)
    sine = (1.0 - target.square()).clamp_min(0.0).sqrt()
    unit_ortho = _unit_orthogonal(residual, branch)
    return branch_norm * (target * unit_x + sine * unit_ortho)


def _body_layer(model: nn.Module, index: int) -> nn.Module:
    from kimodo.training.modeling import unwrap_model

    encoder = getattr(unwrap_model(model).body_model, "seqTransEncoder", None)
    if encoder is None:
        raise TypeError("body_model is missing seqTransEncoder")
    return encoder.layers[int(index)]


def _pair_snapshot(residual: torch.Tensor, tensor: torch.Tensor) -> dict[str, float]:
    stats = residual_pair_stats(residual.detach(), tensor.detach())
    return {
        "mean_token_cosine": stats["mean_token_cosine"],
        "ln_sigma_mean": stats["ln_sigma_mean"],
        "negative_cosine_fraction": stats["negative_cosine_fraction"],
        "attn_rms": stats["attn_rms"],
    }


@contextmanager
def patch_cosine_nudge(
    layer: nn.Module,
    *,
    cosine_delta: float,
    targets: Sequence[str],
) -> Iterator[dict[str, dict[str, float]]]:
    """Rewrite L15 attn and/or FFN so cosine(x, branch) moves by ``cosine_delta``."""
    wanted = {str(name) for name in targets}
    unknown = wanted - {"attn", "ffn"}
    if unknown:
        raise ValueError(f"unsupported nudge targets: {sorted(unknown)}")
    residual_slot: dict[str, torch.Tensor] = {}
    stats: dict[str, dict[str, float]] = {}
    handles: list[Any] = []

    def _layer_in(_module, inputs) -> None:
        residual_slot["attn"] = inputs[0]

    def _attn_out(_module, _inputs, output):
        residual = residual_slot["attn"]
        tensor = output[0] if isinstance(output, tuple) else output
        before = _pair_snapshot(residual, tensor)
        patched = (
            shift_branch_cosine(residual, tensor, cosine_delta) if "attn" in wanted else tensor
        )
        after = _pair_snapshot(residual, patched)
        stats["attn"] = {
            "before_cosine": before["mean_token_cosine"],
            "after_cosine": after["mean_token_cosine"],
            "before_ln_sigma": before["ln_sigma_mean"],
            "after_ln_sigma": after["ln_sigma_mean"],
            "before_negative_fraction": before["negative_cosine_fraction"],
            "after_negative_fraction": after["negative_cosine_fraction"],
            "before_rms": before["attn_rms"],
            "after_rms": after["attn_rms"],
        }
        if isinstance(output, tuple):
            return (patched,) + tuple(output[1:])
        return patched

    def _ffn_in(_module, inputs) -> None:
        residual_slot["ffn"] = inputs[0]

    def _ffn_out(_module, _inputs, output):
        residual = residual_slot["ffn"]
        before = _pair_snapshot(residual, output)
        patched = (
            shift_branch_cosine(residual, output, cosine_delta) if "ffn" in wanted else output
        )
        after = _pair_snapshot(residual, patched)
        stats["ffn"] = {
            "before_cosine": before["mean_token_cosine"],
            "after_cosine": after["mean_token_cosine"],
            "before_ln_sigma": before["ln_sigma_mean"],
            "after_ln_sigma": after["ln_sigma_mean"],
            "before_negative_fraction": before["negative_cosine_fraction"],
            "after_negative_fraction": after["negative_cosine_fraction"],
            "before_rms": before["attn_rms"],
            "after_rms": after["attn_rms"],
        }
        return patched

    try:
        handles.append(layer.register_forward_pre_hook(_layer_in))
        handles.append(layer.self_attn.register_forward_hook(_attn_out))
        handles.append(layer.linear1.register_forward_pre_hook(_ffn_in))
        handles.append(layer.linear2.register_forward_hook(_ffn_out))
        yield stats
    finally:
        for handle in handles:
            handle.remove()


def _loss_means(losses) -> dict[str, float]:
    return {name: float(value.detach()) for name, value in losses.means.items()}


def probe_onset_nudge(
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
    cosine_delta: float = 0.0,
    targets: Sequence[str] = ("attn",),
) -> dict[str, Any]:
    """One training-mode forward with optional last-layer cosine nudge."""
    from kimodo.training.modeling import set_model_dropout, unwrap_model

    bare = unwrap_model(model)
    set_model_dropout(bare, 0.0)
    bare.train()
    layer = _body_layer(bare, layer_index)
    with patch_cosine_nudge(
        layer, cosine_delta=float(cosine_delta), targets=targets
    ) as nudge_stats:
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
    return {
        "layer_index": int(layer_index),
        "cosine_delta": float(cosine_delta),
        "targets": list(targets),
        "loss_means": _loss_means(losses),
        "loss_total_mean": float(losses.means["total"].detach()),
        "nudge": nudge_stats,
    }


def _row(
    rows: Sequence[dict[str, Any]],
    step: int,
    cosine_delta: float,
    targets: Sequence[str],
) -> dict[str, Any] | None:
    wanted = tuple(sorted(targets))
    for row in rows:
        if int(row.get("global_step") or -1) != int(step):
            continue
        if abs(float(row.get("cosine_delta") or 0.0) - float(cosine_delta)) > 1e-9:
            continue
        got = tuple(sorted(row.get("targets") or []))
        if got != wanted:
            continue
        return row
    return None


def _total(row: dict[str, Any] | None) -> float:
    if row is None:
        return float("nan")
    probe = row.get("probe") or row
    value = probe.get("loss_total_mean")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _term_ratios(current: dict[str, Any] | None, baseline: dict[str, Any] | None) -> dict[str, float]:
    if current is None or baseline is None:
        return {}
    current_means = (current.get("probe") or current).get("loss_means") or {}
    baseline_means = (baseline.get("probe") or baseline).get("loss_means") or {}
    names = sorted(set(current_means) | set(baseline_means))
    return {name: _safe_ratio(current_means.get(name), baseline_means.get(name)) for name in names}


def _direction_verdict(cancel_ratio: float, add_ratio: float, cut: float) -> str:
    if not math.isfinite(cancel_ratio) or not math.isfinite(add_ratio):
        return "incomplete"
    cancel_move = cancel_ratio - 1.0
    add_move = add_ratio - 1.0
    if abs(cancel_move) < cut and abs(add_move) < cut:
        return "loss_indifferent_at_onset"
    if cancel_move <= -cut and (add_move - cancel_move) >= cut:
        return "loss_rewards_onset"
    if cancel_move >= cut and (cancel_move - add_move) >= cut:
        return "loss_punishes_onset"
    if cancel_move >= cut and add_move >= cut:
        return "any_nudge_hurts"
    if cancel_move <= -cut and add_move <= -cut:
        return "any_nudge_helps"
    return "loss_indifferent_at_onset"


def _step_report(
    rows: Sequence[dict[str, Any]],
    step: int,
    *,
    delta: float,
    cut: float,
    target: str,
) -> dict[str, Any]:
    identity = _row(rows, step, 0.0, (target,))
    if identity is None:
        identity = _row(rows, step, 0.0, ("attn",))
    if identity is None:
        identity = _row(rows, step, 0.0, ())
    more_cancel = _row(rows, step, -abs(delta), (target,))
    more_add = _row(rows, step, abs(delta), (target,))
    cancel_ratio = _safe_ratio(_total(more_cancel), _total(identity))
    add_ratio = _safe_ratio(_total(more_add), _total(identity))
    return {
        "step": int(step),
        "target": target,
        "verdict": _direction_verdict(cancel_ratio, add_ratio, cut),
        "more_cancel_loss_ratio": cancel_ratio,
        "more_add_loss_ratio": add_ratio,
        "more_cancel_term_ratios": _term_ratios(more_cancel, identity),
        "more_add_term_ratios": _term_ratios(more_add, identity),
        "identity_loss": _total(identity),
        "more_cancel_loss": _total(more_cancel),
        "more_add_loss": _total(more_add),
    }


def summarize_onset_nudge(
    rows: Sequence[dict[str, Any]],
    *,
    preflip_step: int = 690000,
    flipped_step: int = 695000,
    healthy_step: int = 650000,
    cosine_delta: float = 0.25,
    relative_cut: float = 0.005,
) -> dict[str, Any]:
    """Compare more-cancel vs more-add on pre-flip weights, with controls."""
    delta = abs(float(cosine_delta))
    cut = float(relative_cut)
    preflip_attn = _step_report(rows, preflip_step, delta=delta, cut=cut, target="attn")
    flipped_attn = _step_report(rows, flipped_step, delta=delta, cut=cut, target="attn")
    healthy_attn = _step_report(rows, healthy_step, delta=delta, cut=cut, target="attn")
    preflip_ffn = _step_report(rows, preflip_step, delta=delta, cut=cut, target="ffn")
    flipped_ffn = _step_report(rows, flipped_step, delta=delta, cut=cut, target="ffn")

    return {
        "verdict": preflip_attn["verdict"],
        "preflip_attn": preflip_attn,
        "flipped_attn": flipped_attn,
        "healthy_attn": healthy_attn,
        "preflip_ffn": preflip_ffn,
        "flipped_ffn": flipped_ffn,
        "cosine_delta": delta,
        "relative_cut": cut,
        "preflip_step": int(preflip_step),
        "flipped_step": int(flipped_step),
        "healthy_step": int(healthy_step),
    }
