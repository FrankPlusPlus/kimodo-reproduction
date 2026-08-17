"""Read-only probe: what pushes last-layer attention cosine through zero?

Two measurements on a frozen batch. No checkpoint write.

P. Infinitesimal slope: mix last-layer attn/FFN toward anti-alignment
   with a scalar α (α=0 is identity) and read dL/dα from the 7-term loss.
   Negative: the objective locally rewards the flip (Exp O's ±0.25 was too
   coarse). Positive: Exp O holds at differential scale.

Q. Virtual optimizer steps on an in-memory clone, then restore. Compare
   Adam-atan2, LayerNorm σ-detached backward, Adam, and last-layer wd=1.
   Maps onto which undisclosed knob to change next.

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

import torch
from torch import nn

from kimodo.training.body_jacobian_probe import _zero_grads
from kimodo.training.body_residual_cancel_probe import ResidualCancelHooks, residual_pair_stats
from kimodo.training.optim import AdamAtan2, scheduled_learning_rate


def _safe_float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def mix_toward_cancel(
    residual: torch.Tensor,
    branch: torch.Tensor,
    alpha: torch.Tensor | float,
) -> torch.Tensor:
    """``α=0`` identity; ``α=1`` fully anti-parallel with the same ||branch||."""
    if residual.shape != branch.shape:
        raise ValueError(
            f"shape mismatch: residual {tuple(residual.shape)} vs branch {tuple(branch.shape)}"
        )
    stream = residual.detach()
    vector = branch.detach()
    unit_x = stream / stream.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    target = -unit_x * scale
    return branch + alpha * (target - vector)


def layer_norm_detach_sigma(module: nn.LayerNorm, hidden: torch.Tensor) -> torch.Tensor:
    """True forward σ, but backward treats σ as a constant."""
    mean = hidden.mean(dim=-1, keepdim=True)
    var = hidden.var(dim=-1, unbiased=False, keepdim=True)
    sigma = (var + float(module.eps)).sqrt()
    normalized = (hidden - mean) / sigma.detach()
    if module.elementwise_affine:
        if module.weight is not None:
            normalized = normalized * module.weight
        if module.bias is not None:
            normalized = normalized + module.bias
    return normalized


def _body_layer(model: nn.Module, index: int) -> nn.Module:
    from kimodo.training.modeling import unwrap_model

    encoder = getattr(unwrap_model(model).body_model, "seqTransEncoder", None)
    if encoder is None:
        raise TypeError("body_model is missing seqTransEncoder")
    return encoder.layers[int(index)]


def _last_layer_params(model: nn.Module, index: int) -> list[nn.Parameter]:
    return [parameter for parameter in _body_layer(model, index).parameters() if parameter.requires_grad]


def _other_params(model: nn.Module, index: int) -> list[nn.Parameter]:
    last_ids = {id(parameter) for parameter in _last_layer_params(model, index)}
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in last_ids
    ]


@contextmanager
def patch_alpha_cancel(
    layer: nn.Module,
    *,
    alpha: torch.Tensor,
    targets: Sequence[str],
) -> Iterator[dict[str, dict[str, float]]]:
    wanted = {str(name) for name in targets}
    unknown = wanted - {"attn", "ffn"}
    if unknown:
        raise ValueError(f"unsupported mix targets: {sorted(unknown)}")
    residual_slot: dict[str, torch.Tensor] = {}
    stats: dict[str, dict[str, float]] = {}
    handles: list[Any] = []

    def _layer_in(_module, inputs) -> None:
        residual_slot["attn"] = inputs[0]

    def _attn_out(_module, _inputs, output):
        residual = residual_slot["attn"]
        tensor = output[0] if isinstance(output, tuple) else output
        patched = mix_toward_cancel(residual, tensor, alpha) if "attn" in wanted else tensor
        before = residual_pair_stats(residual.detach(), tensor.detach())
        after = residual_pair_stats(residual.detach(), patched.detach())
        stats["attn"] = {
            "before_cosine": before["mean_token_cosine"],
            "after_cosine": after["mean_token_cosine"],
        }
        if isinstance(output, tuple):
            return (patched,) + tuple(output[1:])
        return patched

    def _ffn_in(_module, inputs) -> None:
        residual_slot["ffn"] = inputs[0]

    def _ffn_out(_module, _inputs, output):
        residual = residual_slot["ffn"]
        patched = mix_toward_cancel(residual, output, alpha) if "ffn" in wanted else output
        before = residual_pair_stats(residual.detach(), output.detach())
        after = residual_pair_stats(residual.detach(), patched.detach())
        stats["ffn"] = {
            "before_cosine": before["mean_token_cosine"],
            "after_cosine": after["mean_token_cosine"],
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


@contextmanager
def patch_detach_layer_sigma(layer: nn.Module) -> Iterator[None]:
    restored: list[tuple[nn.Module, Any]] = []
    try:
        for slot in ("norm1", "norm2"):
            module = getattr(layer, slot, None)
            if module is None:
                raise AttributeError(f"layer is missing {slot}")
            original = module.forward

            def _forward(hidden: torch.Tensor, module: nn.LayerNorm = module) -> torch.Tensor:
                return layer_norm_detach_sigma(module, hidden)

            module.forward = _forward  # type: ignore[method-assign]
            restored.append((module, original))
        yield
    finally:
        for module, original in restored:
            module.forward = original  # type: ignore[method-assign]


def _call_denoiser(
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
) -> torch.Tensor:
    return model(
        noisy,
        valid_frames,
        text_features,
        text_pad_mask,
        timesteps,
        first_heading_angle=first_heading_angle,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
    )


def measure_last_layer(
    model: nn.Module,
    *,
    layer_index: int = 15,
    noisy: torch.Tensor,
    valid_frames: torch.Tensor,
    text_features: torch.Tensor,
    text_pad_mask: torch.Tensor,
    timesteps: torch.Tensor,
    first_heading_angle: torch.Tensor,
    motion_mask: torch.Tensor,
    observed_motion: torch.Tensor,
) -> dict[str, float]:
    from kimodo.training.modeling import set_model_dropout, unwrap_model

    bare = unwrap_model(model)
    set_model_dropout(bare, 0.0)
    bare.train()
    with ResidualCancelHooks(bare.body_model, (int(layer_index),)) as hooks:
        with torch.no_grad():
            _call_denoiser(
                bare,
                noisy=noisy,
                valid_frames=valid_frames,
                text_features=text_features,
                text_pad_mask=text_pad_mask,
                timesteps=timesteps,
                first_heading_angle=first_heading_angle,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
            )
    last = (hooks.summarize() or [{}])[0]
    ffn = last.get("ffn") if isinstance(last.get("ffn"), dict) else {}
    return {
        "attn_cosine": _safe_float(last.get("mean_token_cosine")),
        "attn_sigma": _safe_float(last.get("ln_sigma_mean")),
        "ffn_cosine": _safe_float(ffn.get("mean_token_cosine")),
        "ffn_sigma": _safe_float(ffn.get("ln_sigma_mean")),
    }


def probe_onset_slope(
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
    targets: Sequence[str] = ("attn",),
) -> dict[str, Any]:
    """One training-mode forward; return dL/dα at α=0 toward anti-alignment."""
    from kimodo.training.modeling import set_model_dropout, unwrap_model

    bare = unwrap_model(model)
    set_model_dropout(bare, 0.0)
    bare.train()
    _zero_grads(bare)
    layer = _body_layer(bare, layer_index)
    alpha = torch.zeros((), device=noisy.device, dtype=torch.float32, requires_grad=True)
    with patch_alpha_cancel(layer, alpha=alpha, targets=targets) as mix_stats:
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
        (d_total,) = torch.autograd.grad(total, alpha, retain_graph=True, create_graph=False)
        term_grads: dict[str, float] = {}
        names = [name for name in losses.frame_sums if name != "total"]
        for index, name in enumerate(names):
            (grad,) = torch.autograd.grad(
                losses.frame_sums[name],
                alpha,
                retain_graph=index < len(names) - 1,
                create_graph=False,
            )
            term_grads[name] = _safe_float(grad.detach())
    valid = float(losses.valid_frame_count.detach())
    d_sum = _safe_float(d_total.detach())
    _zero_grads(bare)
    return {
        "layer_index": int(layer_index),
        "targets": list(targets),
        "loss_total_mean": _safe_float(losses.means["total"].detach()),
        "valid_frame_count": int(valid),
        "d_loss_sum_d_alpha": d_sum,
        "d_loss_mean_d_alpha": d_sum / valid if valid else float("nan"),
        "term_d_sum_d_alpha": term_grads,
        "mix": mix_stats,
    }


def _build_virtual_optimizer(
    model: nn.Module,
    *,
    variant: str,
    layer_index: int,
    lr: float,
    betas: tuple[float, float],
    atan2_lambda: float,
) -> torch.optim.Optimizer:
    last = _last_layer_params(model, layer_index)
    rest = _other_params(model, layer_index)
    if variant == "adam":
        return torch.optim.Adam(
            [{"params": rest, "weight_decay": 0.0}, {"params": last, "weight_decay": 0.0}],
            lr=lr,
            betas=betas,
        )
    last_decay = 1.0 if variant == "atan2_last_wd1" else 0.0
    return AdamAtan2(
        [{"params": rest, "weight_decay": 0.0}, {"params": last, "weight_decay": last_decay}],
        lr=lr,
        betas=betas,
        weight_decay=0.0,
        atan2_lambda=atan2_lambda,
    )


def probe_virtual_steps(
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
    optimizer_config,
    global_step: int,
    total_steps: int,
    variant: str,
    layer_index: int = 15,
    n_steps: int = 20,
    log_every: int = 5,
) -> dict[str, Any]:
    """Take ``n_steps`` in memory, photograph cosine, restore weights."""
    from kimodo.training.modeling import set_model_dropout, unwrap_model

    allowed = {"atan2", "detach_sigma", "adam", "atan2_last_wd1"}
    if variant not in allowed:
        raise ValueError(f"unsupported virtual-step variant: {variant}")
    bare = unwrap_model(model)
    set_model_dropout(bare, 0.0)
    bare.train()
    snapshot = {key: value.detach().clone() for key, value in bare.state_dict().items()}
    lr = scheduled_learning_rate(
        int(global_step),
        peak_lr=float(optimizer_config.learning_rate),
        total_steps=int(total_steps),
        warmup_steps=int(optimizer_config.warmup_steps),
        warmup_start_lr=optimizer_config.warmup_start_lr,
        lr_end=optimizer_config.lr_end,
        schedule_start_step=int(optimizer_config.lr_schedule_start_step),
    )
    betas = tuple(float(value) for value in optimizer_config.betas)
    optimizer = _build_virtual_optimizer(
        bare,
        variant=variant,
        layer_index=layer_index,
        lr=lr,
        betas=(betas[0], betas[1]),
        atan2_lambda=float(optimizer_config.atan2_lambda),
    )
    forward = dict(
        noisy=noisy,
        valid_frames=valid_frames,
        text_features=text_features,
        text_pad_mask=text_pad_mask,
        timesteps=timesteps,
        first_heading_angle=first_heading_angle,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
    )
    layer = _body_layer(bare, layer_index)
    trace: list[dict[str, float]] = []

    def _record(step: int) -> None:
        pair = measure_last_layer(bare, layer_index=layer_index, **forward)
        pair["step"] = float(step)
        trace.append(pair)

    try:
        _record(0)
        detach = variant == "detach_sigma"
        for step in range(1, int(n_steps) + 1):
            _zero_grads(bare)
            context = patch_detach_layer_sigma(layer) if detach else nullcontext()
            with context:
                prediction = _call_denoiser(bare, **forward)
                losses = loss_function(prediction, target, valid_frames)
                losses.frame_sums["total"].backward()
            optimizer.step()
            if step % int(log_every) == 0 or step == int(n_steps):
                row = measure_last_layer(bare, layer_index=layer_index, **forward)
                row["step"] = float(step)
                row["loss_total_mean"] = _safe_float(losses.means["total"].detach())
                trace.append(row)
        start = trace[0] if trace else {}
        end = trace[-1] if trace else {}
        return {
            "variant": variant,
            "n_steps": int(n_steps),
            "lr": float(lr),
            "layer_index": int(layer_index),
            "trace": trace,
            "attn_cosine_start": _safe_float(start.get("attn_cosine")),
            "attn_cosine_end": _safe_float(end.get("attn_cosine")),
            "attn_cosine_delta": _safe_float(end.get("attn_cosine")) - _safe_float(start.get("attn_cosine")),
            "ffn_cosine_start": _safe_float(start.get("ffn_cosine")),
            "ffn_cosine_end": _safe_float(end.get("ffn_cosine")),
            "attn_sigma_end": _safe_float(end.get("attn_sigma")),
        }
    finally:
        bare.load_state_dict(snapshot, strict=True)
        _zero_grads(bare)


def _probe(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    payload = row.get("probe") or row
    return payload if isinstance(payload, dict) else {}


def _find_slope(rows: Sequence[dict[str, Any]], step: int, target: str) -> dict[str, Any]:
    for row in rows:
        if int(row.get("global_step") or -1) != int(step):
            continue
        if row.get("kind") != "slope":
            continue
        if list(row.get("targets") or []) == [target]:
            return _probe(row)
    return {}


def _slope_verdict(d_mean: float, cut: float) -> str:
    if not math.isfinite(d_mean):
        return "incomplete"
    if d_mean <= -cut:
        return "loss_rewards_local_cancel"
    if d_mean >= cut:
        return "loss_punishes_local_cancel"
    return "loss_flat_local"


def _drive_verdict(delta: float, cut: float) -> bool:
    return math.isfinite(delta) and delta <= -cut


def summarize_onset_path(
    rows: Sequence[dict[str, Any]],
    *,
    preflip_step: int = 690000,
    flipped_step: int = 695000,
    healthy_step: int = 650000,
    slope_cut: float = 0.005,
    cosine_cut: float = 0.01,
) -> dict[str, Any]:
    """Map P+Q at the pre-flip checkpoint onto a config hint."""
    preflip_attn = _find_slope(rows, preflip_step, "attn")
    preflip_ffn = _find_slope(rows, preflip_step, "ffn")
    d_mean = _safe_float(preflip_attn.get("d_loss_mean_d_alpha"))
    slope = _slope_verdict(d_mean, float(slope_cut))

    def _delta(step: int, variant: str) -> float:
        for row in rows:
            if int(row.get("global_step") or -1) != int(step):
                continue
            if row.get("kind") != "virtual":
                continue
            if str(row.get("variant") or "") != variant:
                continue
            return _safe_float((_probe(row)).get("attn_cosine_delta"))
        return float("nan")

    atan2_delta = _delta(preflip_step, "atan2")
    detach_delta = _delta(preflip_step, "detach_sigma")
    adam_delta = _delta(preflip_step, "adam")
    wd1_delta = _delta(preflip_step, "atan2_last_wd1")
    cut = float(cosine_cut)
    atan2_drives = _drive_verdict(atan2_delta, cut)
    detach_drives = _drive_verdict(detach_delta, cut)
    adam_drives = _drive_verdict(adam_delta, cut)
    wd1_drives = _drive_verdict(wd1_delta, cut)

    if slope == "loss_rewards_local_cancel":
        path = "loss_rewards_local_cancel"
        config_hint = "last_layer_wd_and_or_objective"
    elif atan2_drives and not detach_drives:
        path = "sigma_path_drives_cancel"
        config_hint = "last_layer_wd_from_preflip"
    elif atan2_drives and not adam_drives:
        path = "atan2_rule_drives_cancel"
        config_hint = "atan2_lambda_or_last_layer_lr"
    elif atan2_drives and not wd1_drives:
        path = "last_layer_wd_holds"
        config_hint = "last_layer_wd_from_preflip"
    elif atan2_drives:
        path = "virtual_steps_drive_cancel"
        config_hint = "last_layer_wd_from_preflip"
    else:
        path = "slow_multibatch_drift"
        config_hint = "stronger_wd_from_500_or_650"

    return {
        "verdict": path,
        "config_hint": config_hint,
        "preflip_slope": slope,
        "preflip_ffn_slope": _slope_verdict(
            _safe_float(preflip_ffn.get("d_loss_mean_d_alpha")), float(slope_cut)
        ),
        "preflip_d_loss_mean_d_alpha": d_mean,
        "preflip_ffn_d_loss_mean_d_alpha": _safe_float(preflip_ffn.get("d_loss_mean_d_alpha")),
        "preflip_atan2_cosine_delta": atan2_delta,
        "preflip_detach_sigma_cosine_delta": detach_delta,
        "preflip_adam_cosine_delta": adam_delta,
        "preflip_last_wd1_cosine_delta": wd1_delta,
        "healthy_atan2_cosine_delta": _delta(healthy_step, "atan2"),
        "flipped_atan2_cosine_delta": _delta(flipped_step, "atan2"),
        "preflip_step": int(preflip_step),
        "flipped_step": int(flipped_step),
        "healthy_step": int(healthy_step),
        "slope_cut": float(slope_cut),
        "cosine_cut": cut,
    }
