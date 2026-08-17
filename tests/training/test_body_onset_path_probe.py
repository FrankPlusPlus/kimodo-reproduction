from __future__ import annotations

import math

import pytest
import torch

from kimodo.training.body_onset_path_probe import mix_toward_cancel, summarize_onset_path


def test_mix_identity_at_zero_alpha():
    residual = torch.randn(2, 3, 8)
    branch = torch.randn(2, 3, 8)
    out = mix_toward_cancel(residual, branch, 0.0)
    assert torch.allclose(out, branch)


def test_mix_alpha_one_is_antiparallel():
    residual = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    branch = torch.tensor([[[0.22, math.sqrt(1.0 - 0.22**2), 0.0, 0.0]]])
    out = mix_toward_cancel(residual, branch, 1.0)
    cosine = torch.nn.functional.cosine_similarity(residual, out, dim=-1)
    assert float(cosine) == pytest.approx(-1.0, abs=1e-5)
    assert torch.allclose(out.norm(dim=-1), branch.norm(dim=-1), atol=1e-5)


def test_mix_alpha_tensor_gets_gradient():
    residual = torch.ones(1, 2, 4)
    branch = torch.ones(1, 2, 4)
    alpha = torch.zeros((), requires_grad=True)
    out = mix_toward_cancel(residual, branch, alpha)
    out.sum().backward()
    assert alpha.grad is not None
    assert float(alpha.grad) != 0.0


def _slope(step: int, target: str, d_mean: float) -> dict:
    return {
        "global_step": step,
        "kind": "slope",
        "targets": [target],
        "probe": {"d_loss_mean_d_alpha": d_mean},
    }


def _virtual(step: int, variant: str, delta: float) -> dict:
    return {
        "global_step": step,
        "kind": "virtual",
        "variant": variant,
        "probe": {"attn_cosine_delta": delta},
    }


def _q(step: int, atan2: float, detach: float, adam: float, wd1: float) -> list[dict]:
    return [
        _virtual(step, "atan2", atan2),
        _virtual(step, "detach_sigma", detach),
        _virtual(step, "adam", adam),
        _virtual(step, "atan2_last_wd1", wd1),
    ]


def test_summarize_sigma_path_when_detach_stops_drop():
    rows = (
        [_slope(690000, "attn", 0.02), _slope(690000, "ffn", 0.04)]
        + _q(690000, -0.05, 0.0, -0.04, -0.05)
        + _q(650000, 0.0, 0.0, 0.0, 0.0)
        + _q(695000, -0.02, 0.0, 0.0, 0.0)
    )
    report = summarize_onset_path(rows)
    assert report["verdict"] == "sigma_path_drives_cancel"
    assert report["config_hint"] == "last_layer_wd_from_preflip"


def test_summarize_atan2_rule_when_adam_does_not_drop():
    rows = (
        [_slope(690000, "attn", 0.02), _slope(690000, "ffn", 0.02)]
        + _q(690000, -0.05, -0.04, 0.001, -0.05)
    )
    report = summarize_onset_path(rows)
    assert report["verdict"] == "atan2_rule_drives_cancel"
    assert report["config_hint"] == "atan2_lambda_or_last_layer_lr"


def test_summarize_last_layer_wd_holds():
    rows = (
        [_slope(690000, "attn", 0.02), _slope(690000, "ffn", 0.02)]
        + _q(690000, -0.05, -0.04, -0.04, 0.002)
    )
    report = summarize_onset_path(rows)
    assert report["verdict"] == "last_layer_wd_holds"
    assert report["config_hint"] == "last_layer_wd_from_preflip"


def test_summarize_slow_drift_when_virtual_steps_do_not_drop():
    rows = (
        [_slope(690000, "attn", 0.02), _slope(690000, "ffn", 0.02)]
        + _q(690000, -0.001, 0.0, 0.0, 0.0)
    )
    report = summarize_onset_path(rows)
    assert report["verdict"] == "slow_multibatch_drift"
    assert report["config_hint"] == "stronger_wd_from_500_or_650"


def test_summarize_loss_rewards_beats_virtual():
    rows = (
        [_slope(690000, "attn", -0.02), _slope(690000, "ffn", 0.02)]
        + _q(690000, -0.05, 0.0, 0.0, 0.0)
    )
    report = summarize_onset_path(rows)
    assert report["verdict"] == "loss_rewards_local_cancel"
    assert report["config_hint"] == "last_layer_wd_and_or_objective"
