"""Read-only body attention probe: last-layer pointer vs uniform mixing.

This is not a training step. It loads existing checkpoints, runs one forward
with dropout 0, and records per-layer self-attention:

- Shannon entropy of motion-query attention
- mass on prefix tokens (text / time / heading)
- mass on constrained (keyframe) motion tokens
- peak probability

The pointer hypothesis: 800k last-layer attention is more peaked on keyframe
tokens than 795k, while inner layers stay close to uniform. That is a change
in what the last layer computes, not a gradient-clip problem.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


def _mean_masked(value: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean of ``value`` over True positions of ``mask``. Broadcasts mask to value."""
    if mask.ndim == value.ndim - 1:
        mask = mask.unsqueeze(1)
    keep = mask.to(dtype=value.dtype)
    keep = keep.expand_as(value)
    denom = float(keep.sum())
    if denom <= 0.0:
        return float("nan")
    return float((value * keep).sum() / denom)


def _safe_ratio(value: object, baseline: object) -> float:
    try:
        numerator = float(value)  # type: ignore[arg-type]
        denominator = float(baseline)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return numerator / denominator


def keyframe_frame_mask(motion_mask: torch.Tensor) -> torch.Tensor:
    """True on frames that have any constrained channel. ``[B, T]``."""
    if motion_mask.ndim != 3:
        raise ValueError(f"motion_mask must be [B, T, D], got {tuple(motion_mask.shape)}")
    return motion_mask.detach().bool().any(dim=-1)


def attention_pointer_stats(
    weights: torch.Tensor,
    *,
    valid_frames: torch.Tensor,
    keyframe_frames: torch.Tensor,
) -> dict[str, float]:
    """Summarize one layer's attention maps.

    ``weights`` is ``[B, S, S]`` (heads already averaged) or ``[B, H, S, S]``.
    Sequence layout is prefix tokens then T motion tokens.
    """
    maps = weights.detach().float()
    if maps.ndim == 3:
        maps = maps.unsqueeze(1)
    if maps.ndim != 4 or maps.shape[-1] != maps.shape[-2]:
        raise ValueError(f"attention must be [B, H, S, S] or [B, S, S], got {tuple(weights.shape)}")
    batch, _heads, seq, _seq = maps.shape
    if valid_frames.shape != keyframe_frames.shape:
        raise ValueError("valid_frames and keyframe_frames must match")
    if valid_frames.shape[0] != batch:
        raise ValueError(f"batch mismatch: attn {batch} vs frames {valid_frames.shape[0]}")
    motion_len = int(valid_frames.shape[1])
    prefix_len = seq - motion_len
    if prefix_len < 1:
        raise ValueError(f"sequence {seq} is shorter than motion length {motion_len}")

    motion_queries = maps[:, :, prefix_len:, :]
    valid = valid_frames.bool()
    keyframe = keyframe_frames.bool() & valid
    unconstrained = valid & ~keyframe

    prefix_mass = motion_queries[:, :, :, :prefix_len].sum(dim=-1)
    motion_mass = motion_queries[:, :, :, prefix_len:].sum(dim=-1)
    key_selector = torch.zeros(batch, seq, dtype=motion_queries.dtype, device=maps.device)
    key_selector[:, prefix_len:] = keyframe.to(dtype=motion_queries.dtype)
    keyframe_mass = (motion_queries * key_selector[:, None, None, :]).sum(dim=-1)

    motion_keys = motion_queries[:, :, :, prefix_len:]
    eye = torch.eye(motion_len, dtype=motion_keys.dtype, device=maps.device)
    self_mass = (motion_keys * eye[None, None, :, :]).sum(dim=-1)

    probs = motion_queries.clamp_min(0.0)
    entropy = -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1)
    valid_key_count = valid.to(dtype=probs.dtype).sum(dim=-1) + float(prefix_len)
    log_keys = valid_key_count.clamp_min(2.0).log().view(batch, 1, 1)
    normalized_entropy = entropy / log_keys
    max_prob = motion_queries.max(dim=-1).values

    keyframe_fraction = (keyframe.to(dtype=probs.dtype).sum(dim=-1) / valid.to(dtype=probs.dtype).sum(dim=-1).clamp_min(1.0))
    uniform_keyframe = _mean_masked(keyframe_fraction.view(batch, 1, 1).expand_as(keyframe_mass), valid)

    def pack(query_mask: torch.Tensor) -> dict[str, float]:
        mass = _mean_masked(keyframe_mass, query_mask)
        uniform = uniform_keyframe
        return {
            "entropy": _mean_masked(entropy, query_mask),
            "normalized_entropy": _mean_masked(normalized_entropy, query_mask),
            "max_prob": _mean_masked(max_prob, query_mask),
            "prefix_mass": _mean_masked(prefix_mass, query_mask),
            "motion_mass": _mean_masked(motion_mass, query_mask),
            "keyframe_mass": mass,
            "self_mass": _mean_masked(self_mass, query_mask),
            "uniform_keyframe_mass": uniform,
            "keyframe_lift": (mass / uniform) if math.isfinite(uniform) and uniform > 0.0 else float("nan"),
        }

    n_valid = int(valid.sum().item())
    n_keyframe = int(keyframe.sum().item())
    n_unconstrained = int(unconstrained.sum().item())
    return {
        "prefix_len": float(prefix_len),
        "seq_len": float(seq),
        "head_count": float(maps.shape[1]),
        "valid_query_count": float(n_valid),
        "keyframe_query_count": float(n_keyframe),
        "unconstrained_query_count": float(n_unconstrained),
        "all_valid": pack(valid),
        "unconstrained": pack(unconstrained),
        "keyframe_queries": pack(keyframe),
    }


class BodyAttentionHooks:
    """Force ``need_weights=True`` on body self-attention and keep the maps."""

    def __init__(self, body_model: nn.Module, layer_indices: Sequence[int] | None = None) -> None:
        self.body_model = body_model
        self.layer_indices = None if layer_indices is None else [int(index) for index in layer_indices]
        self.weights: dict[int, torch.Tensor] = {}
        self._patched: list[nn.Module] = []

    def _layer_slots(self) -> list[nn.Module]:
        encoder = getattr(self.body_model, "seqTransEncoder", None)
        if encoder is None:
            raise TypeError("body_model is missing seqTransEncoder")
        return list(encoder.layers)

    def __enter__(self) -> BodyAttentionHooks:
        self.weights = {}
        slots = self._layer_slots()
        chosen = range(len(slots)) if self.layer_indices is None else self.layer_indices
        for index in chosen:
            layer = slots[index]
            attention = layer.self_attn
            original = attention.forward

            def wrapped(*args, _original=original, _index=index, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                output, weights = _original(*args, **kwargs)
                if weights is None:
                    raise RuntimeError(
                        f"body layer {_index} self_attn returned no weights; "
                        "need_weights=True was ignored"
                    )
                self.weights[_index] = weights.detach()
                return output, weights

            attention.forward = wrapped
            self._patched.append(attention)
        return self

    def __exit__(self, *exc: object) -> None:
        for module in self._patched:
            if "forward" in module.__dict__:
                del module.forward
        self._patched.clear()


def probe_body_attention(
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
    layer_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """One training-mode forward. No backward, no optimizer step."""
    from .modeling import set_model_dropout, unwrap_model

    bare = unwrap_model(model)
    set_model_dropout(bare, 0.0)
    bare.train()
    body = bare.body_model
    with BodyAttentionHooks(body, layer_indices=layer_indices) as hooks:
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
        frames = keyframe_frame_mask(motion_mask)
        layers = []
        for index in sorted(hooks.weights):
            stats = attention_pointer_stats(
                hooks.weights[index],
                valid_frames=valid_frames,
                keyframe_frames=frames,
            )
            stats["index"] = index
            layers.append(stats)
    return {
        "prediction_rms": float(prediction.detach().float().pow(2).mean().sqrt()),
        "keyframe_frame_fraction": float(frames.float().mean()),
        "valid_frame_fraction": float(valid_frames.float().mean()),
        "layers": layers,
    }


def _layer_named(probe: dict[str, Any]) -> dict[int, dict[str, Any]]:
    named = {}
    for layer in probe.get("layers") or []:
        named[int(layer["index"])] = layer
    return named


def compare_attention_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ratios / deltas of each row against the first checkpoint."""
    if not rows:
        return []
    baseline_probe = rows[0].get("probe") or {}
    baseline_layers = _layer_named(baseline_probe)
    comparisons = []
    for row in rows:
        probe = row.get("probe") or {}
        current_layers = _layer_named(probe)
        layer_cmp = {}
        for index, baseline in baseline_layers.items():
            current = current_layers.get(index)
            if current is None:
                continue
            unc_b = baseline.get("unconstrained") or {}
            unc_c = current.get("unconstrained") or {}
            layer_cmp[f"layer_{index:02d}"] = {
                "unconstrained_entropy_ratio": _safe_ratio(unc_c.get("entropy"), unc_b.get("entropy")),
                "unconstrained_normalized_entropy_ratio": _safe_ratio(
                    unc_c.get("normalized_entropy"), unc_b.get("normalized_entropy")
                ),
                "unconstrained_max_prob_ratio": _safe_ratio(unc_c.get("max_prob"), unc_b.get("max_prob")),
                "unconstrained_keyframe_mass_delta": (
                    float(unc_c["keyframe_mass"]) - float(unc_b["keyframe_mass"])
                    if isinstance(unc_c.get("keyframe_mass"), (int, float))
                    and isinstance(unc_b.get("keyframe_mass"), (int, float))
                    else float("nan")
                ),
                "unconstrained_keyframe_lift_ratio": _safe_ratio(unc_c.get("keyframe_lift"), unc_b.get("keyframe_lift")),
                "unconstrained_prefix_mass_delta": (
                    float(unc_c["prefix_mass"]) - float(unc_b["prefix_mass"])
                    if isinstance(unc_c.get("prefix_mass"), (int, float))
                    and isinstance(unc_b.get("prefix_mass"), (int, float))
                    else float("nan")
                ),
                "all_valid_entropy_ratio": _safe_ratio(
                    (current.get("all_valid") or {}).get("entropy"),
                    (baseline.get("all_valid") or {}).get("entropy"),
                ),
            }
        comparisons.append(
            {
                "global_step": row.get("global_step"),
                "layers": layer_cmp,
            }
        )
    return comparisons


def _classify_pointer(
    last: dict[str, Any] | None,
    inner: dict[str, Any] | None,
    *,
    entropy_cut: float = 0.85,
    entropy_quiet: float = 0.95,
    mass_cut: float = 0.04,
    mass_quiet: float = 0.03,
) -> str:
    if not last:
        return "incomplete"
    entropy_ratio = last.get("unconstrained_entropy_ratio")
    mass_delta = last.get("unconstrained_keyframe_mass_delta")
    try:
        entropy_ratio = float(entropy_ratio)
        mass_delta = float(mass_delta)
    except (TypeError, ValueError):
        return "incomplete"
    if not math.isfinite(entropy_ratio) or not math.isfinite(mass_delta):
        return "incomplete"
    peaked = entropy_ratio <= entropy_cut and mass_delta >= mass_cut
    quiet = entropy_ratio >= entropy_quiet and abs(mass_delta) <= mass_quiet
    inner_ratio = None
    if inner is not None:
        try:
            inner_ratio = float(inner.get("unconstrained_entropy_ratio"))
        except (TypeError, ValueError):
            inner_ratio = None
    if peaked:
        if inner_ratio is not None and math.isfinite(inner_ratio) and inner_ratio >= entropy_quiet:
            return "pointer"
        return "stack_collapse"
    if quiet:
        return "not_attention"
    return "mixed"


def summarize_pointer_grid(
    cells: Sequence[dict[str, Any]],
    *,
    healthy_step: int = 795000,
    takeoff_step: int = 800000,
    last_layer: int | None = None,
    inner_layer: int = 0,
) -> dict[str, Any]:
    """Classify 2x2 (weight × constraint clock) attention change."""
    keyed = {(int(cell["weight_step"]), int(cell["constraint_step"])): cell for cell in cells}
    required = (
        (healthy_step, healthy_step),
        (takeoff_step, healthy_step),
        (healthy_step, takeoff_step),
        (takeoff_step, takeoff_step),
    )
    missing = [pair for pair in required if pair not in keyed]
    if missing:
        return {"error": f"missing cells {missing}", "verdict": "incomplete"}

    def _row(cell: dict[str, Any]) -> dict[str, Any]:
        return {"global_step": cell["weight_step"], "probe": cell["probe"]}

    def _pick_last(comparison: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
        layers = comparison.get("layers") or {}
        if last_layer is not None:
            name = f"layer_{last_layer:02d}"
            return name, layers.get(name)
        names = sorted(layers)
        if not names:
            return None, None
        return names[-1], layers[names[-1]]

    weight_at_healthy = compare_attention_rows(
        [_row(keyed[(healthy_step, healthy_step)]), _row(keyed[(takeoff_step, healthy_step)])]
    )[1]
    clock_at_healthy = compare_attention_rows(
        [_row(keyed[(healthy_step, healthy_step)]), _row(keyed[(healthy_step, takeoff_step)])]
    )[1]
    last_name, last_weight = _pick_last(weight_at_healthy)
    _clock_name, last_clock = _pick_last(clock_at_healthy)
    inner_name = f"layer_{inner_layer:02d}"
    inner_weight = (weight_at_healthy.get("layers") or {}).get(inner_name)
    verdict = _classify_pointer(last_weight, inner_weight)
    clock_verdict = _classify_pointer(last_clock, (clock_at_healthy.get("layers") or {}).get(inner_name))
    if verdict == "pointer" and clock_verdict == "not_attention":
        grid_verdict = "weights"
    elif verdict == "not_attention" and clock_verdict in {"pointer", "stack_collapse", "mixed"}:
        grid_verdict = "clock"
    elif verdict == "incomplete":
        grid_verdict = "incomplete"
    else:
        grid_verdict = verdict
    return {
        "healthy_step": healthy_step,
        "takeoff_step": takeoff_step,
        "last_layer": last_name,
        "inner_layer": inner_name,
        "verdict": grid_verdict,
        "weight_verdict": verdict,
        "clock_verdict": clock_verdict,
        "weight_at_healthy_clock": last_weight,
        "clock_at_healthy_weight": last_clock,
        "inner_weight_at_healthy_clock": inner_weight,
        "layers_weight_at_healthy_clock": weight_at_healthy.get("layers"),
    }


def median_pointer_grids(grids: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [grid for grid in grids if grid.get("verdict") not in {None, "incomplete"} and "error" not in grid]
    if not valid:
        return {"verdict": "incomplete", "world_size": len(grids)}
    labels = [grid.get("verdict") for grid in valid]
    verdict = max(set(labels), key=labels.count)
    return {
        "world_size": len(grids),
        "valid_ranks": len(valid),
        "verdict": verdict,
        "rank_verdicts": {
            label: labels.count(label)
            for label in ("weights", "pointer", "stack_collapse", "not_attention", "clock", "mixed", "incomplete")
        },
        "last_layer": valid[0].get("last_layer"),
        "weight_at_healthy_clock": valid[0].get("weight_at_healthy_clock"),
        "clock_at_healthy_weight": valid[0].get("clock_at_healthy_weight"),
    }
