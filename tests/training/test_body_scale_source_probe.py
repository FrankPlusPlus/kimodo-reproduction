from __future__ import annotations

import math

from kimodo.training.body_scale_source_probe import (
    decompose_sigma_change,
    residual_stack_ratios,
    summarize_scale_source,
)


def _layer(residual: float, sigma: float, sum_over_x: float, cosine: float = 0.2) -> dict:
    return {
        "residual_rms": residual,
        "ln_sigma_mean": sigma,
        "sum_over_residual_rms": sum_over_x,
        "mean_token_cosine": cosine,
    }


def test_residual_scale_dominates_when_sumx_rises():
    report = decompose_sigma_change(
        _layer(1.27, 1.31, 1.03, 0.21),
        _layer(0.86, 0.95, 1.11, 0.45),
    )
    assert report["verdict"] == "residual_scale_dominates"
    assert report["residual_share"] > 1.0
    assert report["geometry_share"] < 0.0


def test_additive_geometry_dominates_when_residual_flat():
    report = decompose_sigma_change(
        _layer(0.80, 0.81, 1.02, -0.29),
        _layer(0.80, 0.48, 0.64, -0.78),
    )
    assert report["verdict"] == "additive_geometry_dominates"
    assert report["geometry_share"] > 0.8
    assert abs(report["residual_share"]) < 0.1


def test_stack_scale_from_weights_when_gamma_tracks():
    stack = residual_stack_ratios(
        {0: {"residual_rms": 1.24}, 15: {"residual_rms": 1.27}},
        {0: {"residual_rms": 1.03}, 15: {"residual_rms": 0.86}},
    )
    sigma = decompose_sigma_change(
        _layer(1.27, 1.31, 1.03),
        _layer(0.86, 0.95, 1.11),
    )
    summary = summarize_scale_source(
        sigma=sigma,
        stack_ratios=stack,
        weights={"mean_ln1_gamma_ratio": 0.68, "input_linear_ratio": 0.84},
    )
    assert stack["L00"] < 0.95
    assert summary["source"] == "stack_scale_from_weights"
    assert summary["stack_wide_residual_shrink"] is True


def test_log_identity_reconstructs():
    report = decompose_sigma_change(
        _layer(1.0, 1.0, 1.0),
        _layer(0.5, 0.4, 0.8),
    )
    assert math.isclose(report["dlog_sigma"], report["reconstruction"], rel_tol=1e-6)
