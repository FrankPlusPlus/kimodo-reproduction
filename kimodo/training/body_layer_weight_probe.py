"""Read-only: did last-layer weights grow vs inner layers before the flip?

Weights only. No forward, no optimizer, no data. Used to choose last-layer
weight decay vs a stronger global decay before a 650k fork.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

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


def layer_matrix_rms(matrices: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """Group collected matrices by layer index."""
    rows: dict[str, dict[str, float]] = {}
    for name, tensor in matrices.items():
        if not name.startswith("layer_"):
            continue
        layer, _, rest = name.partition(".")
        slot = rows.setdefault(layer, {})
        slot[rest] = tensor_rms(tensor)
    return rows


def grouped_rms(layer: dict[str, float]) -> dict[str, float]:
    def _mean(keys: Sequence[str]) -> float:
        values = [layer[key] for key in keys if key in layer and math.isfinite(layer[key])]
        if not values:
            return float("nan")
        return float(sum(values) / len(values))

    attn_keys = ("self_attn.in_proj", "self_attn.out_proj")
    return {
        "attn_rms": _mean(attn_keys),
        "in_proj_rms": float(layer.get("self_attn.in_proj") or float("nan")),
        "out_proj_rms": float(layer.get("self_attn.out_proj") or float("nan")),
        "ffn_out_rms": float(layer.get("linear2") or float("nan")),
        "ln1_gamma_rms": float(layer.get("norm1.gamma") or float("nan")),
        "ln2_gamma_rms": float(layer.get("norm2.gamma") or float("nan")),
    }


def checkpoint_layer_rms(state: dict[str, torch.Tensor], *, layers: Sequence[int] = (0, 7, 14, 15)) -> dict[str, Any]:
    matrices = collect_tail_matrices(state, layers=layers)
    per_layer = layer_matrix_rms(matrices)
    report = {}
    for index in layers:
        name = f"layer_{int(index):02d}"
        report[name] = grouped_rms(per_layer.get(name) or {})
    return report


def _lookup(rows: Sequence[dict[str, Any]], step: int) -> dict[str, Any]:
    for row in rows:
        if int(row.get("global_step") or -1) == int(step):
            return (row.get("probe") or row) if isinstance(row.get("probe") or row, dict) else {}
    return {}


def _layer(probe: dict[str, Any], index: int) -> dict[str, float]:
    payload = probe.get(f"layer_{int(index):02d}") or {}
    return payload if isinstance(payload, dict) else {}


def summarize_layer_weight_timeline(
    rows: Sequence[dict[str, Any]],
    *,
    start_step: int = 650000,
    end_step: int = 695000,
    last_layer: int = 15,
    inner_layer: int = 0,
    grow_cut: float = 0.05,
) -> dict[str, Any]:
    """Compare last-layer vs inner-layer RMS growth from start_step to end_step."""
    start = _lookup(rows, start_step)
    end = _lookup(rows, end_step)
    last_start = _layer(start, last_layer)
    last_end = _layer(end, last_layer)
    inner_start = _layer(start, inner_layer)
    inner_end = _layer(end, inner_layer)
    last_ratio = _safe_ratio(last_end.get("in_proj_rms"), last_start.get("in_proj_rms"))
    inner_ratio = _safe_ratio(inner_end.get("in_proj_rms"), inner_start.get("in_proj_rms"))
    last_attn_ratio = _safe_ratio(last_end.get("attn_rms"), last_start.get("attn_rms"))
    inner_attn_ratio = _safe_ratio(inner_end.get("attn_rms"), inner_start.get("attn_rms"))
    last_ffn_ratio = _safe_ratio(last_end.get("ffn_out_rms"), last_start.get("ffn_out_rms"))
    relative = last_ratio / inner_ratio if inner_ratio and math.isfinite(inner_ratio) else float("nan")

    if not math.isfinite(last_ratio) or not math.isfinite(inner_ratio):
        verdict = "incomplete"
        config_hint = "incomplete"
    elif last_ratio >= 1.0 + grow_cut and relative >= 1.0 + grow_cut:
        verdict = "last_layer_grows"
        config_hint = "last_layer_wd_from_650k"
    elif last_ratio >= 1.0 + grow_cut or inner_ratio >= 1.0 + grow_cut:
        verdict = "uniform_growth"
        config_hint = "global_wd_from_650k"
    else:
        verdict = "no_weight_growth"
        config_hint = "global_wd_from_650k"

    return {
        "verdict": verdict,
        "config_hint": config_hint,
        "start_step": int(start_step),
        "end_step": int(end_step),
        "last_in_proj_ratio": last_ratio,
        "inner_in_proj_ratio": inner_ratio,
        "last_over_inner": relative,
        "last_attn_ratio": last_attn_ratio,
        "inner_attn_ratio": inner_attn_ratio,
        "last_ffn_ratio": last_ffn_ratio,
        "last_in_proj_start": last_start.get("in_proj_rms"),
        "last_in_proj_end": last_end.get("in_proj_rms"),
    }
