"""Read-only tail-gain probe: weight spectra vs activation rank vs incoming grads.

Attention maps at 800k did not peak. Jacobian still showed L15 self-attn
parameter grads 1.91x and L15 pre-LN channel variance 0.44x with L14 output
RMS unchanged. That is channel/feature collapse, not a token pointer.

This probe splits three remaining causes on the same 795k vs 800k weights:

- stored gain: singular values of L15 Q/K/V/O and L14 FFN
- feature rank: effective rank of the residual stream entering L15
- incoming grad: RMS of dL/d(attn_out) and dL/d(ffn_out)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


def tensor_rms(value: torch.Tensor) -> float:
    finite = value.detach().float().reshape(-1)
    if finite.numel() == 0:
        return float("nan")
    return float(finite.pow(2).mean().sqrt())


def _safe_ratio(value: object, baseline: object) -> float:
    try:
        numerator = float(value)  # type: ignore[arg-type]
        denominator = float(baseline)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return numerator / denominator


def matrix_spectrum(weight: torch.Tensor) -> dict[str, float]:
    """Top singular values of a 2-D weight. Biases are reported as RMS only."""
    matrix = weight.detach().float()
    if matrix.ndim == 1:
        rms = tensor_rms(matrix)
        return {
            "rms": rms,
            "l2": float(matrix.norm()),
            "spectral_norm": rms,
            "s2": float("nan"),
            "s5": float("nan"),
            "effective_rank": 1.0 if rms > 0 else 0.0,
            "smax_over_rms": 1.0 if rms > 0 else float("nan"),
        }
    if matrix.ndim != 2:
        raise ValueError(f"expected a matrix, got {tuple(matrix.shape)}")
    rms = tensor_rms(matrix)
    l2 = float(matrix.norm())
    singular = torch.linalg.svdvals(matrix.cpu())
    top = singular[:5]
    energy = singular.pow(2)
    denom = float(energy.sum())
    if denom > 0:
        effective_rank = float((singular.sum() ** 2) / denom)
    else:
        effective_rank = float("nan")
    smax = float(top[0]) if top.numel() else float("nan")
    return {
        "rms": rms,
        "l2": l2,
        "spectral_norm": smax,
        "s2": float(top[1]) if top.numel() > 1 else float("nan"),
        "s5": float(top[4]) if top.numel() > 4 else float("nan"),
        "effective_rank": effective_rank,
        "smax_over_rms": (smax / rms) if rms else float("nan"),
    }


def split_qkv_in_proj(in_proj_weight: torch.Tensor) -> dict[str, torch.Tensor]:
    if in_proj_weight.ndim != 2 or in_proj_weight.shape[0] % 3 != 0:
        raise ValueError(f"packed QKV weight must be [3d, d], got {tuple(in_proj_weight.shape)}")
    q_w, k_w, v_w = in_proj_weight.chunk(3, dim=0)
    return {"q": q_w, "k": k_w, "v": v_w}


def activation_effective_rank(tokens: torch.Tensor) -> dict[str, float]:
    """Effective rank of a token×channel matrix after flattening batch/time."""
    features = tokens.detach().float()
    if features.ndim == 3:
        features = features.reshape(-1, features.shape[-1])
    if features.ndim != 2:
        raise ValueError(f"expected [N, D] or [B, T, D], got {tuple(tokens.shape)}")
    centered = features - features.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered.cpu())
    energy = singular.pow(2)
    denom = float(energy.sum())
    if denom <= 0:
        return {"rms": tensor_rms(features), "channel_var": float("nan"), "effective_rank": float("nan"), "smax": float("nan")}
    channel_var = float(features.var(dim=-1, unbiased=False).mean())
    return {
        "rms": tensor_rms(features),
        "channel_var": channel_var,
        "effective_rank": float((singular.sum() ** 2) / denom),
        "smax": float(singular[0]),
        "s2": float(singular[1]) if singular.numel() > 1 else float("nan"),
        "participation": float((energy[:8].sum() / denom) if singular.numel() else float("nan")),
    }


def _layer_key(prefix: str, index: int, suffix: str) -> str:
    return f"{prefix}.seqTransEncoder.layers.{index}.{suffix}"


def collect_tail_matrices(state: dict[str, torch.Tensor], *, prefix: str = "body_model", layers: Sequence[int] = (0, 7, 13, 14, 15)) -> dict[str, torch.Tensor]:
    matrices: dict[str, torch.Tensor] = {}
    for index in layers:
        packed = state.get(_layer_key(prefix, index, "self_attn.in_proj_weight"))
        if isinstance(packed, torch.Tensor):
            matrices[f"layer_{index:02d}.self_attn.in_proj"] = packed
            for name, piece in split_qkv_in_proj(packed).items():
                matrices[f"layer_{index:02d}.self_attn.{name}"] = piece
        for suffix, label in (
            ("self_attn.out_proj.weight", "self_attn.out_proj"),
            ("linear1.weight", "linear1"),
            ("linear2.weight", "linear2"),
            ("norm1.weight", "norm1.gamma"),
            ("norm2.weight", "norm2.gamma"),
        ):
            tensor = state.get(_layer_key(prefix, index, suffix))
            if isinstance(tensor, torch.Tensor):
                matrices[f"layer_{index:02d}.{label}"] = tensor
    output = state.get(f"{prefix}.output_linear.weight")
    if isinstance(output, torch.Tensor):
        matrices["output_linear"] = output
    return matrices


def compare_matrix_spectra(
    current: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    report = {}
    for name, base in baseline.items():
        other = current.get(name)
        if other is None or tuple(other.shape) != tuple(base.shape):
            continue
        base_spec = matrix_spectrum(base)
        cur_spec = matrix_spectrum(other)
        delta = (other.detach().float() - base.detach().float()).reshape(-1)
        base_flat = base.detach().float().reshape(-1)
        base_norm = float(base_flat.norm())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                other.detach().float().reshape(1, -1),
                base_flat.reshape(1, -1),
                dim=1,
            ).item()
        )
        report[name] = {
            **{f"baseline_{key}": value for key, value in base_spec.items()},
            **{f"current_{key}": value for key, value in cur_spec.items()},
            "spectral_norm_ratio": _safe_ratio(cur_spec["spectral_norm"], base_spec["spectral_norm"]),
            "rms_ratio": _safe_ratio(cur_spec["rms"], base_spec["rms"]),
            "effective_rank_ratio": _safe_ratio(cur_spec["effective_rank"], base_spec["effective_rank"]),
            "relative_update": float(delta.norm() / base_norm) if base_norm else float("nan"),
            "cosine": cosine,
        }
    return report


class TailIOHooks:
    """Capture residual-stream rank and incoming grads at selected body layers."""

    def __init__(self, body_model: nn.Module, layer_indices: Sequence[int]) -> None:
        self.body_model = body_model
        self.layer_indices = [int(index) for index in layer_indices]
        self.forward: dict[int, dict[str, torch.Tensor]] = {}
        self.backward: dict[int, dict[str, float]] = {}
        self._handles: list[Any] = []

    def _slots(self) -> list[nn.Module]:
        encoder = getattr(self.body_model, "seqTransEncoder", None)
        if encoder is None:
            raise TypeError("body_model is missing seqTransEncoder")
        return list(encoder.layers)

    def __enter__(self) -> TailIOHooks:
        self.forward = {}
        self.backward = {}
        slots = self._slots()
        for index in self.layer_indices:
            layer = slots[index]
            slot: dict[str, torch.Tensor] = {}
            self.forward[index] = slot
            self.backward[index] = {}

            def _ln1(_module, inputs, slot=slot) -> None:
                slot["ln1_in"] = inputs[0].detach()

            def _ln2(_module, inputs, slot=slot) -> None:
                slot["ln2_in"] = inputs[0].detach()

            def _attn_out(_module, _inputs, output, slot=slot, index=index) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                slot["attn_out"] = tensor.detach()

                def _grad(grad, index=index):
                    self.backward[index]["attn_out_grad_rms"] = tensor_rms(grad)
                    return None

                if tensor.requires_grad:
                    tensor.register_hook(_grad)

            def _ff_out(_module, _inputs, output, slot=slot, index=index) -> None:
                slot["ff_out"] = output.detach()

                def _grad(grad, index=index):
                    self.backward[index]["ff_out_grad_rms"] = tensor_rms(grad)
                    return None

                if output.requires_grad:
                    output.register_hook(_grad)

            def _layer_out(_module, _inputs, output, slot=slot, index=index) -> None:
                slot["layer_out"] = output.detach()

                def _grad(grad, index=index):
                    self.backward[index]["layer_out_grad_rms"] = tensor_rms(grad)
                    return None

                if output.requires_grad:
                    output.register_hook(_grad)

            self._handles.append(layer.norm1.register_forward_pre_hook(_ln1))
            self._handles.append(layer.norm2.register_forward_pre_hook(_ln2))
            self._handles.append(layer.self_attn.register_forward_hook(_attn_out))
            self._handles.append(layer.linear2.register_forward_hook(_ff_out))
            self._handles.append(layer.register_forward_hook(_layer_out))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summarize(self) -> list[dict[str, Any]]:
        rows = []
        for index in self.layer_indices:
            slot = self.forward.get(index) or {}
            row: dict[str, Any] = {"index": index, **(self.backward.get(index) or {})}
            for name, tensor in slot.items():
                row[f"{name}_rms"] = tensor_rms(tensor)
                if name in {"ln1_in", "ln2_in", "layer_out"}:
                    rank = activation_effective_rank(tensor)
                    row[f"{name}_channel_var"] = rank["channel_var"]
                    row[f"{name}_effective_rank"] = rank["effective_rank"]
                    row[f"{name}_smax"] = rank["smax"]
                    row[f"{name}_participation8"] = rank["participation"]
            rows.append(row)
        return rows


def probe_tail_io(
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
    layer_indices: Sequence[int] = (0, 7, 13, 14, 15),
) -> dict[str, Any]:
    from .modeling import set_model_dropout, unwrap_model

    bare = unwrap_model(model)
    set_model_dropout(bare, 0.0)
    bare.train()
    for parameter in bare.parameters():
        parameter.grad = None
    body = bare.body_model
    with TailIOHooks(body, layer_indices) as hooks:
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
        total = loss_function(prediction, target, valid_frames).frame_sums["total"]
        total.backward()
    for parameter in bare.parameters():
        parameter.grad = None
    return {
        "prediction_rms": tensor_rms(prediction),
        "layers": hooks.summarize(),
    }


def _named_layers(probe: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(layer["index"]): layer for layer in probe.get("layers") or []}


def compare_tail_io(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, dict[str, float]]:
    report = {}
    base_layers = _named_layers(baseline)
    for index, base in base_layers.items():
        other = _named_layers(current).get(index)
        if other is None:
            continue
        keys = sorted(set(base) | set(other))
        ratios = {}
        for key in keys:
            if key == "index":
                continue
            if isinstance(base.get(key), (int, float)) and isinstance(other.get(key), (int, float)):
                ratios[f"{key}_ratio"] = _safe_ratio(other.get(key), base.get(key))
        report[f"layer_{index:02d}"] = ratios
    return report


def summarize_tail_gain(
    spectra: dict[str, dict[str, float]],
    io_ratios: dict[str, dict[str, float]] | None = None,
    *,
    last_layer: int = 15,
    inner_layer: int = 0,
    ratio_cut: float = 1.2,
    quiet: float = 1.1,
) -> dict[str, Any]:
    last = f"layer_{last_layer:02d}"
    inner = f"layer_{inner_layer:02d}"
    qkv_ratio = max(
        (
            spectra.get(f"{last}.self_attn.{name}", {}).get("spectral_norm_ratio") or 0.0
            for name in ("q", "k", "v", "out_proj", "in_proj")
        ),
        default=0.0,
    )
    ff_ratio = max(
        (
            spectra.get(f"layer_14.{name}", {}).get("spectral_norm_ratio") or 0.0
            for name in ("linear1", "linear2")
        ),
        default=0.0,
    )
    inner_ratio = spectra.get(f"{inner}.self_attn.in_proj", {}).get("spectral_norm_ratio")
    io = io_ratios or {}
    last_io = io.get(last) or {}
    rank_ratio = last_io.get("ln1_in_effective_rank_ratio")
    var_ratio = last_io.get("ln1_in_channel_var_ratio")
    attn_g_ratio = last_io.get("attn_out_grad_rms_ratio")
    votes = []
    if qkv_ratio >= ratio_cut:
        votes.append("weight_gain")
    if isinstance(rank_ratio, float) and math.isfinite(rank_ratio) and rank_ratio <= (1.0 / ratio_cut):
        votes.append("feature_rank_collapse")
    elif isinstance(var_ratio, float) and math.isfinite(var_ratio) and var_ratio <= (1.0 / ratio_cut):
        votes.append("feature_rank_collapse")
    if isinstance(attn_g_ratio, float) and math.isfinite(attn_g_ratio) and attn_g_ratio >= ratio_cut:
        votes.append("incoming_grad")
    if not votes:
        if qkv_ratio < quiet and (inner_ratio is None or inner_ratio < quiet):
            votes.append("quiet_weights")
        else:
            votes.append("mixed")
    # Prefer the most specific stored-cause labels when several fire.
    if "weight_gain" in votes and "feature_rank_collapse" in votes:
        verdict = "weight_gain_and_rank_collapse"
    elif len(votes) == 1:
        verdict = votes[0]
    else:
        verdict = "mixed:" + "+".join(votes)
    return {
        "verdict": verdict,
        "votes": votes,
        "l15_qkv_spectral_ratio": qkv_ratio,
        "l14_ff_spectral_ratio": ff_ratio,
        "l00_in_proj_spectral_ratio": inner_ratio,
        "l15_ln1_effective_rank_ratio": rank_ratio,
        "l15_ln1_channel_var_ratio": var_ratio,
        "l15_attn_out_grad_rms_ratio": attn_g_ratio,
    }
