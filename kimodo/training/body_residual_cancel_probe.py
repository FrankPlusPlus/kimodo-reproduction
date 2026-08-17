"""Read-only residual-cancellation probe for post-norm last layers.

PyTorch TransformerEncoderLayer with norm_first=False does:

    x = norm1(x + attn(x))
    x = norm2(x + ffn(x))

If attn(x) points against x, the sum becomes quiet, LayerNorm divides by a
small σ, and backward gradients blow up. This probe photographs that sum.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


def _safe_ratio(value: object, baseline: object) -> float:
    try:
        numerator = float(value)  # type: ignore[arg-type]
        denominator = float(baseline)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return numerator / denominator


def residual_pair_stats(residual: torch.Tensor, attn_out: torch.Tensor) -> dict[str, float]:
    """Compare stream ``x`` with ``attn(x)`` and with ``x + attn(x)``.

    All tensors are ``[B, S, D]``. Cosine is the mean over tokens of the
    last-dim cosine between ``x`` and ``attn(x)``. Negative means they cancel.
    """
    if residual.shape != attn_out.shape:
        raise ValueError(f"shape mismatch: residual {tuple(residual.shape)} vs attn {tuple(attn_out.shape)}")
    stream = residual.detach().float()
    attn = attn_out.detach().float()
    summed = stream + attn
    stream_rms = float(stream.pow(2).mean().sqrt())
    attn_rms = float(attn.pow(2).mean().sqrt())
    sum_rms = float(summed.pow(2).mean().sqrt())
    stream_norm = stream.norm(dim=-1).clamp_min(1e-8)
    attn_norm = attn.norm(dim=-1).clamp_min(1e-8)
    cosine = (stream * attn).sum(dim=-1) / (stream_norm * attn_norm)
    sigma = summed.var(dim=-1, unbiased=False).clamp_min(0.0).sqrt()
    return {
        "residual_rms": stream_rms,
        "attn_rms": attn_rms,
        "sum_rms": sum_rms,
        "sum_over_residual_rms": (sum_rms / stream_rms) if stream_rms else float("nan"),
        "mean_token_cosine": float(cosine.mean()),
        "median_token_cosine": float(cosine.flatten().median()),
        "negative_cosine_fraction": float((cosine < 0).float().mean()),
        "ln_sigma_mean": float(sigma.mean()),
        "ln_sigma_p10": float(torch.quantile(sigma.flatten(), 0.10)),
        "ln_sigma_p50": float(sigma.median()),
    }


class ResidualCancelHooks:
    """Capture layer input ``x``, ``attn(x)``, and ``norm1`` input ``x+attn(x)``."""

    def __init__(self, body_model: nn.Module, layer_indices: Sequence[int]) -> None:
        self.body_model = body_model
        self.layer_indices = [int(index) for index in layer_indices]
        self.captured: dict[int, dict[str, torch.Tensor]] = {}
        self._handles: list[Any] = []

    def _slots(self) -> list[nn.Module]:
        encoder = getattr(self.body_model, "seqTransEncoder", None)
        if encoder is None:
            raise TypeError("body_model is missing seqTransEncoder")
        return list(encoder.layers)

    def __enter__(self) -> ResidualCancelHooks:
        self.captured = {index: {} for index in self.layer_indices}
        slots = self._slots()
        for index in self.layer_indices:
            layer = slots[index]
            slot = self.captured[index]

            def _layer_in(_module, inputs, slot=slot) -> None:
                slot["residual"] = inputs[0].detach()

            def _attn_out(_module, _inputs, output, slot=slot) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                slot["attn"] = tensor.detach()

            def _ln1_in(_module, inputs, slot=slot) -> None:
                slot["ln1_in"] = inputs[0].detach()

            def _ffn_residual(_module, inputs, slot=slot) -> None:
                slot["ffn_residual"] = inputs[0].detach()

            def _ln2_in(_module, inputs, slot=slot) -> None:
                slot["ln2_in"] = inputs[0].detach()

            self._handles.append(layer.register_forward_pre_hook(_layer_in))
            self._handles.append(layer.self_attn.register_forward_hook(_attn_out))
            self._handles.append(layer.norm1.register_forward_pre_hook(_ln1_in))
            self._handles.append(layer.linear1.register_forward_pre_hook(_ffn_residual))
            self._handles.append(layer.norm2.register_forward_pre_hook(_ln2_in))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summarize(self) -> list[dict[str, Any]]:
        rows = []
        for index in self.layer_indices:
            slot = self.captured.get(index) or {}
            residual = slot.get("residual")
            attn = slot.get("attn")
            ln1 = slot.get("ln1_in")
            if residual is None or attn is None:
                rows.append({"index": index, "error": "missing residual or attn"})
                continue
            stats = residual_pair_stats(residual, attn)
            if ln1 is not None:
                stats["ln1_in_rms"] = float(ln1.detach().float().pow(2).mean().sqrt())
                stats["ln1_matches_sum"] = float(
                    (ln1.detach().float() - (residual.float() + attn.float())).abs().mean()
                )
            ffn_residual = slot.get("ffn_residual")
            ln2 = slot.get("ln2_in")
            if ffn_residual is not None and ln2 is not None:
                ffn_out = ln2.detach().float() - ffn_residual.detach().float()
                stats["ffn"] = residual_pair_stats(ffn_residual, ffn_out)
            stats["index"] = index
            rows.append(stats)
        return rows


def probe_residual_cancel(
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
    layer_indices: Sequence[int] = (0, 7, 13, 14, 15),
) -> dict[str, Any]:
    from .modeling import set_model_dropout, unwrap_model

    bare = unwrap_model(model)
    set_model_dropout(bare, 0.0)
    bare.train()
    with ResidualCancelHooks(bare.body_model, layer_indices) as hooks:
        bare(
            noisy,
            valid_frames,
            text_features,
            text_pad_mask,
            timesteps,
            first_heading_angle=first_heading_angle,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
        )
    return {"layers": hooks.summarize()}


def compare_residual_rows(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, dict[str, float]]:
    def named(probe: dict[str, Any]) -> dict[int, dict[str, Any]]:
        return {int(layer["index"]): layer for layer in probe.get("layers") or [] if "index" in layer}

    report = {}
    base_layers = named(baseline)
    for index, base in base_layers.items():
        other = named(current).get(index)
        if other is None:
            continue
        report[f"layer_{index:02d}"] = {
            "residual_rms_ratio": _safe_ratio(other.get("residual_rms"), base.get("residual_rms")),
            "attn_rms_ratio": _safe_ratio(other.get("attn_rms"), base.get("attn_rms")),
            "sum_rms_ratio": _safe_ratio(other.get("sum_rms"), base.get("sum_rms")),
            "sum_over_residual_ratio": _safe_ratio(
                other.get("sum_over_residual_rms"), base.get("sum_over_residual_rms")
            ),
            "mean_token_cosine_delta": (
                float(other["mean_token_cosine"]) - float(base["mean_token_cosine"])
                if isinstance(other.get("mean_token_cosine"), (int, float))
                and isinstance(base.get("mean_token_cosine"), (int, float))
                else float("nan")
            ),
            "negative_cosine_fraction_delta": (
                float(other["negative_cosine_fraction"]) - float(base["negative_cosine_fraction"])
                if isinstance(other.get("negative_cosine_fraction"), (int, float))
                and isinstance(base.get("negative_cosine_fraction"), (int, float))
                else float("nan")
            ),
            "ln_sigma_ratio": _safe_ratio(other.get("ln_sigma_mean"), base.get("ln_sigma_mean")),
        }
    return report


def residual_timeline(rows: Sequence[dict[str, Any]], layer_indices: Sequence[int] = (0, 14, 15)) -> list[dict[str, Any]]:
    """One row per checkpoint with L15 (and contrast layers) cancellation scalars."""
    wanted = {int(index) for index in layer_indices}
    timeline = []
    for row in rows:
        named = {int(layer["index"]): layer for layer in (row.get("probe") or {}).get("layers") or []}
        entry: dict[str, Any] = {
            "global_step": row.get("global_step"),
            "constraint_step": row.get("constraint_step"),
        }
        for index in sorted(wanted):
            layer = named.get(index) or {}
            prefix = f"L{index:02d}"
            entry[f"{prefix}_mean_token_cosine"] = layer.get("mean_token_cosine")
            entry[f"{prefix}_negative_cosine_fraction"] = layer.get("negative_cosine_fraction")
            entry[f"{prefix}_ln_sigma_mean"] = layer.get("ln_sigma_mean")
            entry[f"{prefix}_sum_over_residual_rms"] = layer.get("sum_over_residual_rms")
            entry[f"{prefix}_attn_rms"] = layer.get("attn_rms")
            entry[f"{prefix}_residual_rms"] = layer.get("residual_rms")
        timeline.append(entry)
    return timeline


def _branch_stats(layer: dict[str, Any], branch: str) -> dict[str, float | None]:
    if branch == "attn":
        return {
            "cosine": layer.get("mean_token_cosine"),
            "sigma": layer.get("ln_sigma_mean"),
            "sum_over_residual_rms": layer.get("sum_over_residual_rms"),
        }
    nested = layer.get(branch)
    if not isinstance(nested, dict):
        return {"cosine": None, "sigma": None, "sum_over_residual_rms": None}
    return {
        "cosine": nested.get("mean_token_cosine"),
        "sigma": nested.get("ln_sigma_mean"),
        "sum_over_residual_rms": nested.get("sum_over_residual_rms"),
    }


def full_stack_timeline(
    rows: Sequence[dict[str, Any]],
    *,
    num_layers: int = 16,
    constraint_step: int | None = None,
) -> list[dict[str, Any]]:
    """Per-checkpoint attn/ffn cosine and σ for every body layer (experiment R)."""
    timeline: list[dict[str, Any]] = []
    for row in rows:
        if constraint_step is not None and int(row.get("constraint_step") or -1) != int(constraint_step):
            continue
        named = {
            int(layer["index"]): layer
            for layer in (row.get("probe") or {}).get("layers") or []
            if "index" in layer
        }
        entry: dict[str, Any] = {
            "global_step": row.get("global_step"),
            "constraint_step": row.get("constraint_step"),
        }
        for index in range(int(num_layers)):
            layer = named.get(index) or {}
            prefix = f"L{index:02d}"
            attn = _branch_stats(layer, "attn")
            ffn = _branch_stats(layer, "ffn")
            entry[f"{prefix}_attn_cosine"] = attn["cosine"]
            entry[f"{prefix}_attn_sigma"] = attn["sigma"]
            entry[f"{prefix}_ffn_cosine"] = ffn["cosine"]
            entry[f"{prefix}_ffn_sigma"] = ffn["sigma"]
        timeline.append(entry)
    return timeline


def summarize_attn_ffn_asymmetry(
    timeline: Sequence[dict[str, Any]],
    *,
    last_layer: int = 15,
    sigma_cut: float = 0.90,
    cosine_flip: float = 0.0,
) -> dict[str, Any]:
    """Compare when L15 attn vs FFN cross flip thresholds across checkpoints."""
    prefix = f"L{int(last_layer):02d}"
    attn_cross: int | None = None
    ffn_cross: int | None = None
    attn_sigma_drop: int | None = None
    ffn_sigma_drop: int | None = None
    for row in timeline:
        step = row.get("global_step")
        try:
            step_i = int(step)
        except (TypeError, ValueError):
            continue
        attn_cos = row.get(f"{prefix}_attn_cosine")
        ffn_cos = row.get(f"{prefix}_ffn_cosine")
        attn_sig = row.get(f"{prefix}_attn_sigma")
        ffn_sig = row.get(f"{prefix}_ffn_sigma")
        if attn_cross is None and isinstance(attn_cos, (int, float)) and float(attn_cos) <= cosine_flip:
            attn_cross = step_i
        if ffn_cross is None and isinstance(ffn_cos, (int, float)) and float(ffn_cos) <= cosine_flip:
            ffn_cross = step_i
        if attn_sigma_drop is None and isinstance(attn_sig, (int, float)) and float(attn_sig) < sigma_cut:
            attn_sigma_drop = step_i
        if ffn_sigma_drop is None and isinstance(ffn_sig, (int, float)) and float(ffn_sig) < sigma_cut:
            ffn_sigma_drop = step_i
    same_flip = attn_cross is not None and attn_cross == ffn_cross
    verdict = "incomplete"
    if attn_cross is not None and ffn_cross is not None:
        if attn_cross > ffn_cross:
            verdict = "attn_flips_after_ffn"
        elif attn_cross < ffn_cross:
            verdict = "attn_flips_before_ffn"
        else:
            verdict = "attn_ffn_flip_together"
    elif attn_cross is not None:
        verdict = "attn_flipped_ffn_never"
    elif ffn_cross is not None:
        verdict = "ffn_flipped_attn_never"
    return {
        "verdict": verdict,
        "l15_attn_flip_step": attn_cross,
        "l15_ffn_flip_step": ffn_cross,
        "l15_attn_sigma_drop_step": attn_sigma_drop,
        "l15_ffn_sigma_drop_step": ffn_sigma_drop,
        "sigma_cut": float(sigma_cut),
        "cosine_flip": float(cosine_flip),
    }


def summarize_residual_cancel(
    ratios: dict[str, dict[str, float]],
    *,
    last_layer: int = 15,
    inner_layer: int = 0,
    sigma_cut: float = 0.85,
    cosine_drop: float = -0.05,
) -> dict[str, Any]:
    last = ratios.get(f"layer_{last_layer:02d}") or {}
    inner = ratios.get(f"layer_{inner_layer:02d}") or {}
    sigma_ratio = last.get("ln_sigma_ratio")
    cosine_delta = last.get("mean_token_cosine_delta")
    sum_ratio = last.get("sum_over_residual_ratio")
    inner_sigma = inner.get("ln_sigma_ratio")
    try:
        sigma_ratio_f = float(sigma_ratio)
        cosine_delta_f = float(cosine_delta)
    except (TypeError, ValueError):
        return {"verdict": "incomplete", "last_layer": last, "inner_layer": inner}
    last_hit = math.isfinite(sigma_ratio_f) and sigma_ratio_f <= sigma_cut
    cosine_hit = math.isfinite(cosine_delta_f) and cosine_delta_f <= cosine_drop
    inner_quiet = True
    if isinstance(inner_sigma, (int, float)) and math.isfinite(float(inner_sigma)):
        inner_quiet = float(inner_sigma) >= 0.95
    if last_hit and cosine_hit and inner_quiet:
        verdict = "residual_cancellation"
    elif last_hit and inner_quiet:
        verdict = "sigma_collapse_without_cosine_drop"
    elif last_hit:
        verdict = "stack_sigma_collapse"
    else:
        verdict = "not_cancellation"
    return {
        "verdict": verdict,
        "l15_ln_sigma_ratio": sigma_ratio,
        "l15_mean_token_cosine_delta": cosine_delta,
        "l15_sum_over_residual_ratio": sum_ratio,
        "l00_ln_sigma_ratio": inner_sigma,
        "last_layer": last,
        "inner_layer": inner,
    }
