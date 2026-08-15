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


def _named_module_grad_norms(root: nn.Module, prefix: str) -> dict[str, float]:
    report = {f"{prefix}.all": parameter_grad_l2(root.parameters())}
    encoder = getattr(root, "seqTransEncoder", None)
    if encoder is not None:
        for index, layer in enumerate(encoder.layers):
            report[f"{prefix}.layer_{index:02d}"] = parameter_grad_l2(layer.parameters())
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
