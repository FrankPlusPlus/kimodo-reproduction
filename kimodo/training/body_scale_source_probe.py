"""Decompose pre-LN σ change into residual scale vs additive geometry.

Post-norm LayerNorm sees ``x + attn(x)``. Its σ can fall for two independent
reasons that this probe keeps separate:

1. Residual scale: ``x`` itself shrinks (upstream γ / weight decay / input
   projection). Then σ falls even if ``x`` and ``attn(x)`` stay aligned.
2. Additive geometry: the *relative* length of ``x+attn(x)`` vs ``x``
   (``sum/x``) falls. Cosine flip is one way this can happen, not the only
   one, and cosine is not treated as the cause.

Log identity used throughout::

    Δlog σ ≈ Δlog residual_rms + Δlog(sum/x)

Weights are photographed separately so a γ cascade can be tested without
claiming that activation geometry caused the scale drop.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from kimodo.training.body_layer_weight_probe import grouped_rms, layer_matrix_rms
from kimodo.training.body_tail_gain_probe import collect_tail_matrices, tensor_rms


def _safe_ratio(value: object, baseline: object) -> float:
    try:
        numerator = float(value)  # type: ignore[arg-type]
        denominator = float(baseline)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return numerator / denominator


def _log_ratio(value: object, baseline: object) -> float:
    ratio = _safe_ratio(value, baseline)
    if not math.isfinite(ratio) or ratio <= 0.0:
        return float("nan")
    return math.log(ratio)


def collect_scale_matrices(
    state: dict[str, torch.Tensor],
    *,
    prefix: str = "body_model",
    layers: Sequence[int] = tuple(range(16)),
) -> dict[str, torch.Tensor]:
    """Layer matrices plus body input/text/output projections."""
    matrices = collect_tail_matrices(state, prefix=prefix, layers=layers)
    for name in ("input_linear.weight", "embed_text.weight", "output_linear.weight"):
        tensor = state.get(f"{prefix}.{name}")
        if isinstance(tensor, torch.Tensor):
            matrices[name.replace(".weight", "")] = tensor
    return matrices


def checkpoint_scale_weights(
    state: dict[str, torch.Tensor],
    *,
    layers: Sequence[int] = tuple(range(16)),
) -> dict[str, Any]:
    matrices = collect_scale_matrices(state, layers=layers)
    per_layer = layer_matrix_rms(matrices)
    report: dict[str, Any] = {
        "input_linear_rms": tensor_rms(matrices["input_linear"]) if "input_linear" in matrices else float("nan"),
        "embed_text_rms": tensor_rms(matrices["embed_text"]) if "embed_text" in matrices else float("nan"),
        "output_linear_rms": tensor_rms(matrices["output_linear"]) if "output_linear" in matrices else float("nan"),
        "layers": {},
    }
    for index in layers:
        name = f"layer_{int(index):02d}"
        report["layers"][name] = grouped_rms(per_layer.get(name) or {})
    gammas = [
        float(report["layers"][f"layer_{index:02d}"].get("ln1_gamma_rms") or float("nan"))
        for index in layers
    ]
    finite = [value for value in gammas if math.isfinite(value)]
    report["mean_ln1_gamma_rms"] = float(sum(finite) / len(finite)) if finite else float("nan")
    return report


def decompose_sigma_change(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Split Δlog σ into residual scale vs additive geometry (sum/x)."""
    dlog_sigma = _log_ratio(other.get("ln_sigma_mean"), base.get("ln_sigma_mean"))
    dlog_residual = _log_ratio(other.get("residual_rms"), base.get("residual_rms"))
    dlog_sum_over_x = _log_ratio(
        other.get("sum_over_residual_rms"),
        base.get("sum_over_residual_rms"),
    )
    reconstruction = dlog_residual + dlog_sum_over_x
    residual_share = _safe_ratio(dlog_residual, dlog_sigma)
    geometry_share = _safe_ratio(dlog_sum_over_x, dlog_sigma)
    same_sign_res = math.isfinite(dlog_sigma) and math.isfinite(dlog_residual) and dlog_sigma * dlog_residual > 0
    same_sign_geo = math.isfinite(dlog_sigma) and math.isfinite(dlog_sum_over_x) and dlog_sigma * dlog_sum_over_x > 0
    if not math.isfinite(dlog_sigma):
        verdict = "incomplete"
    elif abs(dlog_sigma) < 1e-4:
        verdict = "sigma_flat"
    elif same_sign_res and abs(dlog_residual) >= 2.0 * max(abs(dlog_sum_over_x), 1e-12):
        verdict = "residual_scale_dominates"
    elif same_sign_geo and abs(dlog_sum_over_x) >= 2.0 * max(abs(dlog_residual), 1e-12):
        verdict = "additive_geometry_dominates"
    else:
        verdict = "mixed"
    return {
        "verdict": verdict,
        "dlog_sigma": dlog_sigma,
        "dlog_residual": dlog_residual,
        "dlog_sum_over_x": dlog_sum_over_x,
        "reconstruction": reconstruction,
        "residual_share": residual_share,
        "geometry_share": geometry_share,
        "sigma_ratio": _safe_ratio(other.get("ln_sigma_mean"), base.get("ln_sigma_mean")),
        "residual_ratio": _safe_ratio(other.get("residual_rms"), base.get("residual_rms")),
        "sum_over_x_ratio": _safe_ratio(
            other.get("sum_over_residual_rms"),
            base.get("sum_over_residual_rms"),
        ),
        "cosine_end": other.get("mean_token_cosine"),
        "cosine_delta": (
            float(other["mean_token_cosine"]) - float(base["mean_token_cosine"])
            if isinstance(other.get("mean_token_cosine"), (int, float))
            and isinstance(base.get("mean_token_cosine"), (int, float))
            else float("nan")
        ),
    }


def layer_named(probe: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(layer["index"]): layer
        for layer in (probe.get("layers") or [])
        if isinstance(layer, dict) and "index" in layer
    }


def residual_stack_ratios(
    base_layers: dict[int, dict[str, Any]],
    other_layers: dict[int, dict[str, Any]],
) -> dict[str, float]:
    ratios = {}
    for index in sorted(set(base_layers) & set(other_layers)):
        ratios[f"L{index:02d}"] = _safe_ratio(
            other_layers[index].get("residual_rms"),
            base_layers[index].get("residual_rms"),
        )
    return ratios


def compare_weight_scale(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    base_layers = base.get("layers") or {}
    other_layers = other.get("layers") or {}
    gamma_ratios = []
    in_proj_ratios = []
    for name, base_layer in base_layers.items():
        other_layer = other_layers.get(name) or {}
        gamma_ratios.append(_safe_ratio(other_layer.get("ln1_gamma_rms"), base_layer.get("ln1_gamma_rms")))
        in_proj_ratios.append(_safe_ratio(other_layer.get("in_proj_rms"), base_layer.get("in_proj_rms")))
    finite_g = [value for value in gamma_ratios if math.isfinite(value)]
    finite_w = [value for value in in_proj_ratios if math.isfinite(value)]
    return {
        "input_linear_ratio": _safe_ratio(other.get("input_linear_rms"), base.get("input_linear_rms")),
        "embed_text_ratio": _safe_ratio(other.get("embed_text_rms"), base.get("embed_text_rms")),
        "mean_ln1_gamma_ratio": float(sum(finite_g) / len(finite_g)) if finite_g else float("nan"),
        "mean_in_proj_ratio": float(sum(finite_w) / len(finite_w)) if finite_w else float("nan"),
        "l00_ln1_gamma_ratio": _safe_ratio(
            (other_layers.get("layer_00") or {}).get("ln1_gamma_rms"),
            (base_layers.get("layer_00") or {}).get("ln1_gamma_rms"),
        ),
        "l15_ln1_gamma_ratio": _safe_ratio(
            (other_layers.get("layer_15") or {}).get("ln1_gamma_rms"),
            (base_layers.get("layer_15") or {}).get("ln1_gamma_rms"),
        ),
        "l15_in_proj_ratio": _safe_ratio(
            (other_layers.get("layer_15") or {}).get("in_proj_rms"),
            (base_layers.get("layer_15") or {}).get("in_proj_rms"),
        ),
    }


def summarize_scale_source(
    *,
    sigma: dict[str, Any],
    stack_ratios: dict[str, float],
    weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Join activation decomposition with optional weight ratios.

    Logic rules:
    - Do not treat cosine as a cause. Report it only as a correlate of sum/x.
    - If L00 residual shrinks with input_linear, the source is not L15-only.
    - If mean γ shrinks with residual while sum/x rises, the scale drop is
      a cascade / decay story, not cancellation.
    """
    l00 = stack_ratios.get("L00")
    l15 = stack_ratios.get("L15")
    stack_wide = (
        isinstance(l00, float)
        and math.isfinite(l00)
        and l00 < 0.95
        and isinstance(l15, float)
        and math.isfinite(l15)
        and l15 < 0.95
    )
    gamma_tracks_residual = False
    input_tracks_l00 = False
    if weights is not None:
        gamma = weights.get("mean_ln1_gamma_ratio")
        inp = weights.get("input_linear_ratio")
        if isinstance(gamma, float) and isinstance(l15, float) and math.isfinite(gamma) and math.isfinite(l15):
            gamma_tracks_residual = abs(math.log(max(gamma, 1e-8)) - math.log(max(l15, 1e-8))) < 0.15
        if isinstance(inp, float) and isinstance(l00, float) and math.isfinite(inp) and math.isfinite(l00):
            input_tracks_l00 = abs(math.log(max(inp, 1e-8)) - math.log(max(l00, 1e-8))) < 0.15
    source = sigma.get("verdict")
    if source == "residual_scale_dominates" and stack_wide:
        if gamma_tracks_residual or input_tracks_l00:
            source = "stack_scale_from_weights"
        else:
            source = "stack_residual_scale"
    return {
        "source": source,
        "sigma": sigma,
        "stack_wide_residual_shrink": stack_wide,
        "l00_residual_ratio": l00,
        "l15_residual_ratio": l15,
        "gamma_tracks_residual": gamma_tracks_residual,
        "input_tracks_l00": input_tracks_l00,
        "weights": weights,
        "note": (
            "cosine is a correlate of sum/x, not a cause. "
            "Use residual_share vs geometry_share."
        ),
    }
