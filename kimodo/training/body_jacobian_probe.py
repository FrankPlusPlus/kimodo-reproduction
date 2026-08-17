"""Read-only body Jacobian probe: layer gain vs batch-gradient alignment.

This is not a training step. It loads existing checkpoints, runs one forward
and a small number of backwards on a frozen batch, and records:

- per-body-layer parameter gradient norms
- activation RMS
- LayerNorm input variance (post-norm ``norm1`` / ``norm2``)
- dL/d(prediction) (should stay flat if the training gnorm climb is dŷ/dθ)
- pairwise cosines of a few per-sample body parameter gradients
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn


def tensor_rms(value: torch.Tensor) -> float:
    finite = value.detach().float()
    if finite.numel() == 0:
        return float("nan")
    return float(finite.pow(2).mean().sqrt())


def last_dim_variance(value: torch.Tensor) -> float:
    finite = value.detach().float()
    if finite.numel() == 0:
        return float("nan")
    return float(finite.var(dim=-1, unbiased=False).mean())


def parameter_grad_l2(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float32)
    found = False
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        found = True
        total = total + grad.detach().float().pow(2).sum()
    if not found:
        return float("nan")
    return float(total.sqrt())


def flatten_grads(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    pieces = [
        parameter.grad.detach().float().reshape(-1)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not pieces:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(pieces)


def pairwise_cosines(vectors: Sequence[torch.Tensor]) -> dict[str, float]:
    """Mean pairwise cosine of flattened gradients. 1 = identical direction."""
    if len(vectors) < 2:
        return {
            "pair_count": 0,
            "mean_cosine": float("nan"),
            "min_cosine": float("nan"),
            "mean_angle_deg": float("nan"),
        }
    unit = []
    for vector in vectors:
        norm = float(vector.norm())
        if not math.isfinite(norm) or norm == 0.0:
            unit.append(None)
        else:
            unit.append(vector / norm)
    cosines: list[float] = []
    for left in range(len(unit)):
        if unit[left] is None:
            continue
        for right in range(left + 1, len(unit)):
            if unit[right] is None:
                continue
            cosine = float(torch.clamp(torch.dot(unit[left], unit[right]), -1.0, 1.0))
            cosines.append(cosine)
    if not cosines:
        return {
            "pair_count": 0,
            "mean_cosine": float("nan"),
            "min_cosine": float("nan"),
            "mean_angle_deg": float("nan"),
        }
    mean = sum(cosines) / len(cosines)
    return {
        "pair_count": len(cosines),
        "mean_cosine": mean,
        "min_cosine": min(cosines),
        "mean_angle_deg": math.degrees(math.acos(max(-1.0, min(1.0, mean)))),
    }


def alignment_ratio(vectors: Sequence[torch.Tensor]) -> dict[str, float]:
    """||mean g_i|| / mean(||g_i||). ~1 if aligned, ~1/sqrt(N) if random."""
    norms = [float(vector.norm()) for vector in vectors]
    valid = [(vector, norm) for vector, norm in zip(vectors, norms) if math.isfinite(norm) and norm > 0]
    if not valid:
        return {"sample_count": 0, "mean_sample_norm": float("nan"), "mean_vector_norm": float("nan"), "ratio": float("nan")}
    stacked = torch.stack([vector for vector, _ in valid], dim=0)
    mean_vector = stacked.mean(dim=0)
    mean_sample_norm = sum(norm for _, norm in valid) / len(valid)
    mean_vector_norm = float(mean_vector.norm())
    return {
        "sample_count": len(valid),
        "mean_sample_norm": mean_sample_norm,
        "mean_vector_norm": mean_vector_norm,
        "ratio": mean_vector_norm / mean_sample_norm if mean_sample_norm else float("nan"),
    }


class BodyLayerHooks:
    """Capture per-layer activation RMS and LayerNorm input variance."""

    def __init__(self, body_model: nn.Module) -> None:
        self.body_model = body_model
        self._handles: list[Any] = []
        self.layers: list[dict[str, float]] = []

    def _layer_slots(self) -> list[nn.Module]:
        encoder = getattr(self.body_model, "seqTransEncoder", None)
        if encoder is None:
            raise TypeError("body_model is missing seqTransEncoder")
        return list(encoder.layers)

    def __enter__(self) -> BodyLayerHooks:
        self.layers = [
            {
                "index": index,
                "ln1_in_var": float("nan"),
                "ln2_in_var": float("nan"),
                "output_rms": float("nan"),
            }
            for index, _layer in enumerate(self._layer_slots())
        ]
        for index, layer in enumerate(self._layer_slots()):
            slot = self.layers[index]

            def _ln1(_module, inputs, slot=slot) -> None:
                slot["ln1_in_var"] = last_dim_variance(inputs[0])

            def _ln2(_module, inputs, slot=slot) -> None:
                slot["ln2_in_var"] = last_dim_variance(inputs[0])

            def _out(_module, _inputs, output, slot=slot) -> None:
                slot["output_rms"] = tensor_rms(output)

            self._handles.append(layer.norm1.register_forward_pre_hook(_ln1))
            self._handles.append(layer.norm2.register_forward_pre_hook(_ln2))
            self._handles.append(layer.register_forward_hook(_out))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def affine_stats(value: torch.Tensor) -> dict[str, float]:
    finite = value.detach().float()
    if finite.numel() == 0:
        return {"rms": float("nan"), "max_abs": float("nan"), "mean": float("nan")}
    return {
        "rms": float(finite.pow(2).mean().sqrt()),
        "max_abs": float(finite.abs().max()),
        "mean": float(finite.mean()),
    }


def layer_norm_scales(root: nn.Module, prefix: str) -> dict[str, dict[str, float]]:
    """Per-layer LayerNorm γ (weight) scale. This is the clamp candidate."""
    report: dict[str, dict[str, float]] = {}
    encoder = getattr(root, "seqTransEncoder", None)
    if encoder is None:
        return report
    for index, layer in enumerate(encoder.layers):
        for slot in ("norm1", "norm2"):
            module = getattr(layer, slot, None)
            weight = getattr(module, "weight", None) if module is not None else None
            if not isinstance(weight, torch.Tensor):
                continue
            report[f"{prefix}.layer_{index:02d}.{slot}.gamma"] = affine_stats(weight)
    return report


def _named_module_grad_norms(root: nn.Module, prefix: str) -> dict[str, float]:
    report = {f"{prefix}.all": parameter_grad_l2(root.parameters())}
    encoder = getattr(root, "seqTransEncoder", None)
    if encoder is not None:
        for index, layer in enumerate(encoder.layers):
            report[f"{prefix}.layer_{index:02d}"] = parameter_grad_l2(layer.parameters())
            for slot in ("self_attn", "linear1", "linear2", "norm1", "norm2"):
                module = getattr(layer, slot, None)
                if module is None:
                    continue
                report[f"{prefix}.layer_{index:02d}.{slot}"] = parameter_grad_l2(module.parameters())
        report[f"{prefix}.input_linear"] = parameter_grad_l2(root.input_linear.parameters())
        report[f"{prefix}.output_linear"] = parameter_grad_l2(root.output_linear.parameters())
    return report


def _zero_grads(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def probe_forward_backward(
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
    pair_samples: int,
) -> dict[str, Any]:
    """One training-mode forward + batch/per-sample backwards. No optimizer step."""
    from .modeling import set_model_dropout, unwrap_model

    bare = unwrap_model(model)
    set_model_dropout(bare, 0.0)
    bare.train()
    _zero_grads(bare)

    body = bare.body_model
    with BodyLayerHooks(body) as hooks:
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
        (prediction_grad,) = torch.autograd.grad(
            total,
            prediction,
            retain_graph=True,
            create_graph=False,
        )
        total.backward(retain_graph=True)

    batch_body_flat = flatten_grads(body.parameters()).cpu()
    report: dict[str, Any] = {
        "loss_total_mean": float(losses.means["total"].detach()),
        "valid_frame_count": int(losses.valid_frame_count.detach()),
        "prediction_grad_norm": float(prediction_grad.detach().float().norm()),
        "prediction_rms": tensor_rms(prediction),
        "body_activation_layers": list(hooks.layers),
        "grad_norms": {
            **_named_module_grad_norms(bare.root_model, "root"),
            **_named_module_grad_norms(body, "body"),
        },
        "ln_scales": {
            **layer_norm_scales(bare.root_model, "root"),
            **layer_norm_scales(body, "body"),
        },
        "body_batch_grad_norm": float(batch_body_flat.norm()) if batch_body_flat.numel() else float("nan"),
    }

    sample_count = int(valid_frames.shape[0])
    used = min(int(pair_samples), sample_count)
    sample_vectors: list[torch.Tensor] = []
    sample_norms: list[float] = []
    for index in range(used):
        _zero_grads(bare)
        sample_mask = torch.zeros_like(valid_frames)
        sample_mask[index] = valid_frames[index]
        sample_loss = loss_function(prediction, target, sample_mask).frame_sums["total"]
        sample_loss.backward(retain_graph=True)
        vector = flatten_grads(body.parameters()).cpu()
        sample_vectors.append(vector)
        sample_norms.append(float(vector.norm()) if vector.numel() else float("nan"))

    _zero_grads(bare)
    report["per_sample"] = {
        "count": used,
        "body_grad_norms": sample_norms,
        "alignment": alignment_ratio(sample_vectors),
        "pairwise": pairwise_cosines(sample_vectors),
    }
    return report


def _median(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    ordered = sorted(finite)
    return ordered[len(ordered) // 2]


def _classify_ratio_pair(
    weight_ratio: object,
    clock_ratio: object,
    *,
    ratio_cut: float = 1.2,
    clock_quiet: float = 1.1,
) -> str | None:
    try:
        weight = float(weight_ratio)  # type: ignore[arg-type]
        clock = float(clock_ratio)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(weight) or not math.isfinite(clock):
        return None
    weight_hit = weight >= ratio_cut
    clock_hit = clock >= ratio_cut
    if weight_hit and clock < clock_quiet:
        return "weights"
    if clock_hit and weight < clock_quiet:
        return "clock"
    if weight_hit and clock_hit:
        return "interaction"
    return None


def _flatten_named_ratios(comparison: dict[str, Any]) -> dict[str, float]:
    named: dict[str, float] = {}
    for key in (
        "prediction_grad_norm_ratio",
        "body_batch_grad_norm_ratio",
        "mean_sample_body_grad_ratio",
    ):
        value = comparison.get(key)
        if isinstance(value, (int, float)):
            named[key] = float(value)
    for name, value in (comparison.get("body_layer_grad_ratios") or {}).items():
        if isinstance(value, (int, float)):
            named[str(name)] = float(value)
    for name, value in (comparison.get("body_ln_gamma_rms_ratios") or {}).items():
        if isinstance(value, (int, float)):
            named[f"ln_gamma:{name}"] = float(value)
    for index, value in enumerate(comparison.get("body_ln1_var_ratios") or []):
        if isinstance(value, (int, float)):
            named[f"body.layer_{index:02d}.ln1_in_var"] = float(value)
    return named


def summarize_takeoff_grid(
    cells: Sequence[dict[str, Any]],
    *,
    healthy_step: int = 795000,
    takeoff_step: int = 800000,
    ratio_cut: float = 1.2,
    clock_quiet: float = 1.1,
) -> dict[str, Any]:
    """Classify 2x2 (weight × constraint clock) as weights / clock / interaction.

    cells need ``weight_step``, ``constraint_step``, and ``probe``.
    """
    keyed = {(int(cell["weight_step"]), int(cell["constraint_step"])): cell for cell in cells}
    required = (
        (healthy_step, healthy_step),
        (takeoff_step, healthy_step),
        (healthy_step, takeoff_step),
        (takeoff_step, takeoff_step),
    )
    missing = [pair for pair in required if pair not in keyed]
    if missing:
        return {"error": f"missing cells {missing}", "hotspots": [], "verdict": "incomplete"}

    def _row(cell: dict[str, Any]) -> dict[str, Any]:
        return {"global_step": cell["weight_step"], "probe": cell["probe"]}

    healthy_healthy = keyed[(healthy_step, healthy_step)]
    takeoff_healthy = keyed[(takeoff_step, healthy_step)]
    healthy_takeoff = keyed[(healthy_step, takeoff_step)]
    takeoff_takeoff = keyed[(takeoff_step, takeoff_step)]
    weight_at_healthy_clock = compare_checkpoint_rows([_row(healthy_healthy), _row(takeoff_healthy)])[1]
    clock_at_healthy_weight = compare_checkpoint_rows([_row(healthy_healthy), _row(healthy_takeoff)])[1]
    weight_at_takeoff_clock = compare_checkpoint_rows([_row(healthy_takeoff), _row(takeoff_takeoff)])[1]
    clock_at_takeoff_weight = compare_checkpoint_rows([_row(takeoff_healthy), _row(takeoff_takeoff)])[1]
    weight_names = _flatten_named_ratios(weight_at_healthy_clock)
    clock_names = _flatten_named_ratios(clock_at_healthy_weight)
    hotspots = []
    votes = {"weights": 0, "clock": 0, "interaction": 0}
    for name in sorted(set(weight_names) | set(clock_names)):
        weight_ratio = weight_names.get(name)
        clock_ratio = clock_names.get(name)
        label = _classify_ratio_pair(weight_ratio, clock_ratio, ratio_cut=ratio_cut, clock_quiet=clock_quiet)
        if label is None:
            continue
        votes[label] += 1
        hotspots.append(
            {
                "name": name,
                "verdict": label,
                "weight_at_healthy_clock": weight_ratio,
                "clock_at_healthy_weight": clock_ratio,
                "weight_at_takeoff_clock": _flatten_named_ratios(weight_at_takeoff_clock).get(name),
                "clock_at_takeoff_weight": _flatten_named_ratios(clock_at_takeoff_weight).get(name),
            }
        )
    hotspots.sort(
        key=lambda item: max(
            abs((item["weight_at_healthy_clock"] or 1.0) - 1.0),
            abs((item["clock_at_healthy_weight"] or 1.0) - 1.0),
        ),
        reverse=True,
    )
    if votes["weights"] + votes["clock"] + votes["interaction"] == 0:
        verdict = "quiet"
    else:
        verdict = max(votes, key=votes.get)
    return {
        "healthy_step": healthy_step,
        "takeoff_step": takeoff_step,
        "prediction_grad_norm": {
            "weight_at_healthy_clock": weight_at_healthy_clock.get("prediction_grad_norm_ratio"),
            "clock_at_healthy_weight": clock_at_healthy_weight.get("prediction_grad_norm_ratio"),
        },
        "body_batch_grad_norm": {
            "weight_at_healthy_clock": weight_at_healthy_clock.get("body_batch_grad_norm_ratio"),
            "clock_at_healthy_weight": clock_at_healthy_weight.get("body_batch_grad_norm_ratio"),
        },
        "votes": votes,
        "verdict": verdict,
        "hotspots": hotspots[:24],
    }


def median_takeoff_grids(grids: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Median the 2x2 ratios across independent ranks."""
    valid = [grid for grid in grids if grid.get("verdict") not in {None, "incomplete"} and "error" not in grid]
    if not valid:
        return {"verdict": "incomplete", "hotspots": [], "world_size": len(grids)}
    names = []
    seen = set()
    for grid in valid:
        for hotspot in grid.get("hotspots") or []:
            name = hotspot.get("name")
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
    merged_hotspots = []
    votes = {"weights": 0, "clock": 0, "interaction": 0}
    for name in names:
        weight_vals = []
        clock_vals = []
        for grid in valid:
            match = next((item for item in grid.get("hotspots") or [] if item.get("name") == name), None)
            if match is None:
                continue
            if isinstance(match.get("weight_at_healthy_clock"), (int, float)):
                weight_vals.append(float(match["weight_at_healthy_clock"]))
            if isinstance(match.get("clock_at_healthy_weight"), (int, float)):
                clock_vals.append(float(match["clock_at_healthy_weight"]))
        weight_ratio = _median(weight_vals)
        clock_ratio = _median(clock_vals)
        label = _classify_ratio_pair(weight_ratio, clock_ratio)
        if label is None:
            continue
        votes[label] += 1
        merged_hotspots.append(
            {
                "name": name,
                "verdict": label,
                "weight_at_healthy_clock": weight_ratio,
                "clock_at_healthy_weight": clock_ratio,
                "rank_count": max(len(weight_vals), len(clock_vals)),
            }
        )
    merged_hotspots.sort(
        key=lambda item: max(
            abs((item["weight_at_healthy_clock"] or 1.0) - 1.0),
            abs((item["clock_at_healthy_weight"] or 1.0) - 1.0),
        ),
        reverse=True,
    )
    verdict_votes = [grid.get("verdict") for grid in valid if grid.get("verdict") in votes]
    if verdict_votes:
        verdict = max(set(verdict_votes), key=verdict_votes.count)
    elif votes["weights"] + votes["clock"] + votes["interaction"] == 0:
        verdict = "quiet"
    else:
        verdict = max(votes, key=votes.get)
    return {
        "world_size": len(grids),
        "valid_ranks": len(valid),
        "verdict": verdict,
        "rank_verdicts": {label: verdict_votes.count(label) for label in ("weights", "clock", "interaction", "quiet")},
        "votes": votes,
        "prediction_grad_norm": {
            "weight_at_healthy_clock": _median(
                [
                    float(grid["prediction_grad_norm"]["weight_at_healthy_clock"])
                    for grid in valid
                    if isinstance((grid.get("prediction_grad_norm") or {}).get("weight_at_healthy_clock"), (int, float))
                ]
            ),
            "clock_at_healthy_weight": _median(
                [
                    float(grid["prediction_grad_norm"]["clock_at_healthy_weight"])
                    for grid in valid
                    if isinstance((grid.get("prediction_grad_norm") or {}).get("clock_at_healthy_weight"), (int, float))
                ]
            ),
        },
        "body_batch_grad_norm": {
            "weight_at_healthy_clock": _median(
                [
                    float(grid["body_batch_grad_norm"]["weight_at_healthy_clock"])
                    for grid in valid
                    if isinstance((grid.get("body_batch_grad_norm") or {}).get("weight_at_healthy_clock"), (int, float))
                ]
            ),
            "clock_at_healthy_weight": _median(
                [
                    float(grid["body_batch_grad_norm"]["clock_at_healthy_weight"])
                    for grid in valid
                    if isinstance((grid.get("body_batch_grad_norm") or {}).get("clock_at_healthy_weight"), (int, float))
                ]
            ),
        },
        "hotspots": merged_hotspots[:24],
    }


def compare_checkpoint_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ratios of each row against the first checkpoint (usually 650k)."""
    if not rows:
        return []
    baseline = rows[0]
    comparisons = []
    for row in rows:
        comparisons.append(
            {
                "global_step": row.get("global_step"),
                "prediction_grad_norm_ratio": _ratio(
                    row.get("probe", {}).get("prediction_grad_norm"),
                    baseline.get("probe", {}).get("prediction_grad_norm"),
                ),
                "body_batch_grad_norm_ratio": _ratio(
                    row.get("probe", {}).get("body_batch_grad_norm"),
                    baseline.get("probe", {}).get("body_batch_grad_norm"),
                ),
                "mean_sample_body_grad_ratio": _ratio(
                    row.get("probe", {}).get("per_sample", {}).get("alignment", {}).get("mean_sample_norm"),
                    baseline.get("probe", {}).get("per_sample", {}).get("alignment", {}).get("mean_sample_norm"),
                ),
                "alignment_ratio": row.get("probe", {}).get("per_sample", {}).get("alignment", {}).get("ratio"),
                "mean_pairwise_cosine": row.get("probe", {}).get("per_sample", {}).get("pairwise", {}).get(
                    "mean_cosine"
                ),
                "body_layer_grad_ratios": _layer_ratios(
                    row.get("probe", {}).get("grad_norms", {}),
                    baseline.get("probe", {}).get("grad_norms", {}),
                    prefix="body.layer_",
                ),
                "body_ln1_var_ratios": _hook_ratios(
                    row.get("probe", {}).get("body_activation_layers", []),
                    baseline.get("probe", {}).get("body_activation_layers", []),
                    key="ln1_in_var",
                ),
                "body_ln_gamma_rms_ratios": _ln_gamma_ratios(
                    row.get("probe", {}).get("ln_scales", {}),
                    baseline.get("probe", {}).get("ln_scales", {}),
                    prefix="body.layer_",
                ),
            }
        )
    return comparisons


def _ratio(value: object, baseline: object) -> float:
    try:
        numerator = float(value)  # type: ignore[arg-type]
        denominator = float(baseline)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return numerator / denominator


def _layer_ratios(current: dict, baseline: dict, prefix: str) -> dict[str, float]:
    return {
        name: _ratio(current.get(name), baseline.get(name))
        for name in baseline
        if name.startswith(prefix)
    }


def _hook_ratios(current: Sequence[dict], baseline: Sequence[dict], key: str) -> list[float]:
    out = []
    for left, right in zip(baseline, current):
        out.append(_ratio(right.get(key), left.get(key)))
    return out


def _ln_gamma_ratios(current: dict, baseline: dict, prefix: str) -> dict[str, float]:
    return {
        name: _ratio(
            (current.get(name) or {}).get("rms"),
            (baseline.get(name) or {}).get("rms"),
        )
        for name in baseline
        if name.startswith(prefix)
    }
