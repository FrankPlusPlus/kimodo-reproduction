from __future__ import annotations

import pytest

from kimodo.training.body_onset_path_probe import summarize_term_slope_timeline
from kimodo.training.body_residual_cancel_probe import (
    full_stack_timeline,
    summarize_attn_ffn_asymmetry,
)
from kimodo.training.wd03_gnorm_flip_timeline import (
    bucket_train_jsonl,
    summarize_gnorm_flip_relation,
)


def _layer_row(index: int, attn_cos: float, ffn_cos: float, attn_sig: float, ffn_sig: float) -> dict:
    return {
        "index": index,
        "mean_token_cosine": attn_cos,
        "ln_sigma_mean": attn_sig,
        "ffn": {
            "mean_token_cosine": ffn_cos,
            "ln_sigma_mean": ffn_sig,
        },
    }


def test_full_stack_timeline_expands_attn_and_ffn():
    rows = [
        {
            "global_step": 750000,
            "constraint_step": 750000,
            "probe": {"layers": [_layer_row(15, -0.02, -0.52, 0.96, 0.98)]},
        }
    ]
    timeline = full_stack_timeline(rows, num_layers=16, constraint_step=750000)
    assert len(timeline) == 1
    assert timeline[0]["L15_attn_cosine"] == pytest.approx(-0.02)
    assert timeline[0]["L15_ffn_cosine"] == pytest.approx(-0.52)


def test_asymmetry_detects_ffn_flips_before_attn():
    timeline = [
        {
            "global_step": 650000,
            "L15_attn_cosine": 0.2,
            "L15_ffn_cosine": -0.5,
            "L15_attn_sigma": 1.0,
            "L15_ffn_sigma": 0.98,
        },
        {
            "global_step": 750000,
            "L15_attn_cosine": -0.02,
            "L15_ffn_cosine": -0.6,
            "L15_attn_sigma": 0.96,
            "L15_ffn_sigma": 0.97,
        },
    ]
    report = summarize_attn_ffn_asymmetry(timeline)
    assert report["verdict"] == "attn_flips_after_ffn"
    assert report["l15_ffn_flip_step"] == 650000
    assert report["l15_attn_flip_step"] == 750000


def test_term_slope_timeline_collects_per_term():
    rows = [
        {
            "global_step": 690000,
            "kind": "slope",
            "targets": ["attn"],
            "probe": {
                "d_loss_mean_d_alpha": 0.007,
                "loss_total_mean": 0.24,
                "term_d_sum_d_alpha": {"joint_position": 0.004, "foot_contact": 0.001},
            },
        }
    ]
    timeline = summarize_term_slope_timeline(rows)
    assert timeline[0]["global_step"] == 690000
    assert timeline[0]["term_joint_position_d_alpha"] == pytest.approx(0.004)


def test_gnorm_flip_relation_orders_events():
    merged = [
        {"step": 690000, "gnorm_p10": 0.45, "attn_cosine": 0.22},
        {"step": 696000, "gnorm_p10": 0.62, "attn_cosine": 0.18},
        {"step": 750000, "gnorm_p10": 0.42, "attn_cosine": -0.02},
    ]
    summary = summarize_gnorm_flip_relation(merged, gnorm_p10_rise=0.55)
    assert summary["verdict"] == "gnorm_rise_before_flip"
    assert summary["gnorm_p10_rise_step"] == 696000
    assert summary["attn_flip_step"] == 750000


def test_bucket_train_jsonl_aggregates_windows():
    rows = [
        {"global_step": 650020, "optimizer/gradient_norm_before_clip": 0.5, "optimizer/gradient_clip_fraction": 0.0},
        {"global_step": 650040, "optimizer/gradient_norm_before_clip": 0.7, "optimizer/gradient_clip_fraction": 0.0},
    ]
    timeline = bucket_train_jsonl(rows, step_start=650000, step_every=10000)
    assert timeline[0]["step"] == 650000
    assert timeline[0]["gnorm_median"] == pytest.approx(0.6)
