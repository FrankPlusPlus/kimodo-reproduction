"""Read-only probe: does the training loss want last-layer cancellation?

Freeze weights. On a frozen batch, subtract the anti-parallel part of
``attn(x)`` (and optionally FFN) so ``x`` and the branch stop cancelling.
Compare the seven-term training loss.

- Loss goes up: the objective rewards wipe-and-replace. min-σ alone may
  pass 800k grads but quality can still follow 750k→790k.
- Loss goes down: cancellation is a numerical parasite. min-σ is enough.
- Loss barely moves: indifferent; do not treat cancellation as the loss's
  strategy.

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


def reduce_cancellation(
    residual: torch.Tensor,
    branch: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Remove ``strength`` times the anti-parallel component of ``branch`` vs ``x``.

    ``strength=0`` is identity. ``strength=1`` leaves cosine(x, branch') ≥ 0.
    """
    if residual.shape != branch.shape:
        raise ValueError(f"shape mismatch: residual {tuple(residual.shape)} vs branch {tuple(branch.shape)}")
    if strength == 0.0:
        return branch
    stream_norm = residual.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    unit = residual / stream_norm
    parallel_scale = (branch * unit).sum(dim=-1, keepdim=True)
    anti_scale = parallel_scale.clamp_max(0.0)
    return branch - float(strength) * anti_scale * unit


def _body_layer(model: nn.Module, index: int) -> nn.Module:
    from kimodo.training.modeling import unwrap_model

    encoder = getattr(unwrap_model(model).body_model, "seqTransEncoder", None)
    if encoder is None:
        raise TypeError("body_model is missing seqTransEncoder")
    return encoder.layers[int(index)]


@contextmanager
def patch_less_cancel(
    layer: nn.Module,
    *,
    strength: float,
    targets: Sequence[str],
) -> Iterator[dict[str, dict[str, float]]]:
    """Rewrite L15 attn and/or FFN outputs so they cancel ``x`` less."""
    wanted = {str(name) for name in targets}
    unknown = wanted - {"attn", "ffn"}
    if unknown:
        raise ValueError(f"unsupported cancel targets: {sorted(unknown)}")
    residual_slot: dict[str, torch.Tensor] = {}
    stats: dict[str, dict[str, float]] = {}
    handles: list[Any] = []

    def _layer_in(_module, inputs) -> None:
        residual_slot["attn"] = inputs[0]

    def _attn_out(_module, _inputs, output):
        residual = residual_slot["attn"]
        tensor = output[0] if isinstance(output, tuple) else output
        before = residual_pair_stats(residual.detach(), tensor.detach())
        patched = reduce_cancellation(residual, tensor, strength) if "attn" in wanted else tensor
        after = residual_pair_stats(residual.detach(), patched.detach())
        stats["attn"] = {
            "before_cosine": before["mean_token_cosine"],
            "after_cosine": after["mean_token_cosine"],
            "before_ln_sigma": before["ln_sigma_mean"],
            "after_ln_sigma": after["ln_sigma_mean"],
            "before_negative_fraction": before["negative_cosine_fraction"],
            "after_negative_fraction": after["negative_cosine_fraction"],
        }
        if isinstance(output, tuple):
            return (patched,) + tuple(output[1:])
        return patched

    def _ffn_in(_module, inputs) -> None:
        residual_slot["ffn"] = inputs[0]

    def _ffn_out(_module, _inputs, output):
        residual = residual_slot["ffn"]
        before = residual_pair_stats(residual.detach(), output.detach())
        patched = reduce_cancellation(residual, output, strength) if "ffn" in wanted else output
        after = residual_pair_stats(residual.detach(), patched.detach())
        stats["ffn"] = {
            "before_cosine": before["mean_token_cosine"],
            "after_cosine": after["mean_token_cosine"],
            "before_ln_sigma": before["ln_sigma_mean"],
            "after_ln_sigma": after["ln_sigma_mean"],
            "before_negative_fraction": before["negative_cosine_fraction"],
            "after_negative_fraction": after["negative_cosine_fraction"],
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


def probe_cancel_loss(
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
    strength: float = 0.0,
    targets: Sequence[str] = ("attn",),
) -> dict[str, Any]:
    """One training-mode forward with optional last-layer de-cancellation."""
    from kimodo.training.modeling import set_model_dropout, unwrap_model

    bare = unwrap_model(model)
    set_model_dropout(bare, 0.0)
    bare.train()
    layer = _body_layer(bare, layer_index)
    with patch_less_cancel(layer, strength=float(strength), targets=targets) as cancel_stats:
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
        "strength": float(strength),
        "targets": list(targets),
        "loss_means": _loss_means(losses),
        "loss_total_mean": float(losses.means["total"].detach()),
        "cancel": cancel_stats,
    }


def _row(rows: Sequence[dict[str, Any]], step: int, strength: float, targets: Sequence[str]) -> dict[str, Any] | None:
    wanted = tuple(sorted(targets))
    for row in rows:
        if int(row.get("global_step") or -1) != int(step):
            continue
        if abs(float(row.get("strength") or 0.0) - float(strength)) > 1e-9:
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


def _term_deltas(current: dict[str, Any] | None, baseline: dict[str, Any] | None) -> dict[str, float]:
    if current is None or baseline is None:
        return {}
    current_means = (current.get("probe") or current).get("loss_means") or {}
    baseline_means = (baseline.get("probe") or baseline).get("loss_means") or {}
    names = sorted(set(current_means) | set(baseline_means))
    return {name: _safe_ratio(current_means.get(name), baseline_means.get(name)) for name in names}


def _verdict_from_move(takeoff_move: float, healthy_move: float, cut: float) -> str:
    if not math.isfinite(takeoff_move):
        return "incomplete"
    bar = cut
    if math.isfinite(healthy_move):
        bar = max(cut, 2.0 * abs(healthy_move))
    if takeoff_move >= bar:
        return "loss_wants_cancellation"
    if takeoff_move <= -bar:
        return "cancellation_is_parasite"
    return "loss_indifferent"


def summarize_cancel_loss(
    rows: Sequence[dict[str, Any]],
    *,
    takeoff_step: int = 800000,
    healthy_step: int = 750000,
    strength: float = 1.0,
    relative_cut: float = 0.005,
) -> dict[str, Any]:
    """Compare identity vs full anti-parallel removal on takeoff vs healthy."""
    healthy_id = _row(rows, healthy_step, 0.0, ("attn",))
    takeoff_id = _row(rows, takeoff_step, 0.0, ("attn",))
    healthy_attn = _row(rows, healthy_step, strength, ("attn",))
    takeoff_attn = _row(rows, takeoff_step, strength, ("attn",))
    takeoff_ffn = _row(rows, takeoff_step, strength, ("ffn",))
    takeoff_both = _row(rows, takeoff_step, strength, ("attn", "ffn"))

    takeoff_attn_ratio = _safe_ratio(_total(takeoff_attn), _total(takeoff_id))
    healthy_attn_ratio = _safe_ratio(_total(healthy_attn), _total(healthy_id))
    takeoff_move = takeoff_attn_ratio - 1.0
    healthy_move = healthy_attn_ratio - 1.0
    attn_verdict = _verdict_from_move(takeoff_move, healthy_move, relative_cut)

    takeoff_ffn_ratio = _safe_ratio(_total(takeoff_ffn), _total(takeoff_id))
    takeoff_both_ratio = _safe_ratio(_total(takeoff_both), _total(takeoff_id))
    ffn_move = takeoff_ffn_ratio - 1.0
    ffn_verdict = _verdict_from_move(ffn_move, 0.0, relative_cut)

    if attn_verdict == "loss_wants_cancellation":
        verdict = "loss_wants_cancellation"
    elif attn_verdict == "cancellation_is_parasite":
        verdict = "cancellation_is_parasite"
    else:
        verdict = attn_verdict

    return {
        "verdict": verdict,
        "attn_verdict": attn_verdict,
        "ffn_verdict": ffn_verdict,
        "takeoff_attn_loss_ratio": takeoff_attn_ratio,
        "healthy_attn_loss_ratio": healthy_attn_ratio,
        "takeoff_ffn_loss_ratio": takeoff_ffn_ratio,
        "takeoff_both_loss_ratio": takeoff_both_ratio,
        "takeoff_attn_term_ratios": _term_deltas(takeoff_attn, takeoff_id),
        "strength": float(strength),
    }
