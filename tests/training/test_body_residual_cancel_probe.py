from __future__ import annotations

import pytest
import torch

from kimodo.training.body_residual_cancel_probe import (
    compare_residual_rows,
    residual_pair_stats,
    residual_timeline,
    summarize_residual_cancel,
)


def test_opposite_vectors_cancel():
    residual = torch.ones(2, 3, 4)
    attn = -torch.ones(2, 3, 4)
    stats = residual_pair_stats(residual, attn)
    assert stats["mean_token_cosine"] == pytest.approx(-1.0)
    assert stats["negative_cosine_fraction"] == pytest.approx(1.0)
    assert stats["sum_rms"] == pytest.approx(0.0, abs=1e-6)
    assert stats["ln_sigma_mean"] == pytest.approx(0.0, abs=1e-6)


def test_aligned_vectors_add():
    residual = torch.ones(1, 2, 8)
    stats = residual_pair_stats(residual, residual)
    assert stats["mean_token_cosine"] == pytest.approx(1.0)
    assert stats["sum_over_residual_rms"] == pytest.approx(2.0)
    assert stats["negative_cosine_fraction"] == pytest.approx(0.0)


def test_summarize_flags_last_layer_only():
    ratios = {
        "layer_00": {
            "ln_sigma_ratio": 1.01,
            "mean_token_cosine_delta": 0.0,
            "sum_over_residual_ratio": 1.0,
        },
        "layer_15": {
            "ln_sigma_ratio": 0.66,
            "mean_token_cosine_delta": -0.12,
            "sum_over_residual_ratio": 0.70,
        },
    }
    report = summarize_residual_cancel(ratios)
    assert report["verdict"] == "residual_cancellation"


def test_compare_residual_rows_delta():
    def probe(cosine: float, sigma: float) -> dict:
        return {
            "layers": [
                {
                    "index": 15,
                    "residual_rms": 1.0,
                    "attn_rms": 1.0,
                    "sum_rms": sigma,
                    "sum_over_residual_rms": sigma,
                    "mean_token_cosine": cosine,
                    "negative_cosine_fraction": 0.2 if cosine > 0 else 0.8,
                    "ln_sigma_mean": sigma,
                }
            ]
        }

    ratios = compare_residual_rows(probe(-0.2, 0.4), probe(0.1, 0.8))
    last = ratios["layer_15"]
    assert last["ln_sigma_ratio"] == pytest.approx(0.5)
    assert last["mean_token_cosine_delta"] == pytest.approx(-0.3)


def test_residual_timeline_keeps_constraint_clock():
    rows = [
        {
            "global_step": 400000,
            "constraint_step": 400000,
            "probe": {
                "layers": [
                    {
                        "index": 15,
                        "mean_token_cosine": 0.2,
                        "negative_cosine_fraction": 0.1,
                        "ln_sigma_mean": 0.9,
                        "sum_over_residual_rms": 1.1,
                        "attn_rms": 1.0,
                        "residual_rms": 1.0,
                    }
                ]
            },
        }
    ]
    timeline = residual_timeline(rows)
    assert timeline[0]["global_step"] == 400000
    assert timeline[0]["constraint_step"] == 400000
    assert timeline[0]["L15_mean_token_cosine"] == pytest.approx(0.2)
