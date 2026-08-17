from __future__ import annotations

import math

import pytest
import torch

from kimodo.training.body_tail_gain_probe import (
    activation_effective_rank,
    collect_tail_matrices,
    compare_matrix_spectra,
    matrix_spectrum,
    split_qkv_in_proj,
    summarize_tail_gain,
)


def test_matrix_spectrum_on_scaled_identity():
    identity = torch.eye(4)
    base = matrix_spectrum(identity)
    assert base["spectral_norm"] == pytest.approx(1.0)
    scaled = matrix_spectrum(3 * identity)
    assert scaled["spectral_norm"] == pytest.approx(3.0)
    assert scaled["rms"] == pytest.approx(3.0 * base["rms"])


def test_split_qkv_chunks_packed_projection():
    packed = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    pieces = split_qkv_in_proj(packed)
    assert pieces["q"].shape == (2, 3)
    assert torch.equal(pieces["q"], packed[:2])
    assert torch.equal(pieces["v"], packed[4:])


def test_activation_rank_drops_on_repeated_channels():
    full = torch.randn(32, 8)
    collapsed = torch.zeros(32, 8)
    collapsed[:, 0] = torch.linspace(-1, 1, 32)
    full_rank = activation_effective_rank(full)
    low_rank = activation_effective_rank(collapsed)
    assert full_rank["effective_rank"] > low_rank["effective_rank"]
    assert low_rank["channel_var"] < full_rank["channel_var"]


def test_compare_spectra_flags_grown_last_layer():
    d = 8
    baseline = {
        "layer_00.self_attn.in_proj": torch.eye(d).repeat(3, 1),
        "layer_15.self_attn.q": torch.eye(d),
        "layer_15.self_attn.k": torch.eye(d),
        "layer_15.self_attn.v": torch.eye(d),
        "layer_15.self_attn.out_proj": torch.eye(d),
        "layer_14.linear2": torch.eye(d),
    }
    current = {name: tensor.clone() for name, tensor in baseline.items()}
    current["layer_15.self_attn.q"] = 2.0 * current["layer_15.self_attn.q"]
    report = compare_matrix_spectra(current, baseline)
    assert report["layer_15.self_attn.q"]["spectral_norm_ratio"] == pytest.approx(2.0)
    assert report["layer_00.self_attn.in_proj"]["spectral_norm_ratio"] == pytest.approx(1.0)
    assert report["layer_15.self_attn.q"]["cosine"] == pytest.approx(1.0)


def test_summarize_tail_gain_prefers_joint_weight_and_rank():
    spectra = {
        "layer_15.self_attn.q": {"spectral_norm_ratio": 1.8},
        "layer_15.self_attn.k": {"spectral_norm_ratio": 1.1},
        "layer_15.self_attn.v": {"spectral_norm_ratio": 1.0},
        "layer_15.self_attn.out_proj": {"spectral_norm_ratio": 1.0},
        "layer_14.linear2": {"spectral_norm_ratio": 1.3},
        "layer_00.self_attn.in_proj": {"spectral_norm_ratio": 1.01},
    }
    io = {
        "layer_15": {
            "ln1_in_effective_rank_ratio": 0.5,
            "ln1_in_channel_var_ratio": 0.44,
            "attn_out_grad_rms_ratio": 1.05,
        }
    }
    report = summarize_tail_gain(spectra, io)
    assert report["verdict"] == "weight_gain_and_rank_collapse"
    assert "weight_gain" in report["votes"]
    assert "feature_rank_collapse" in report["votes"]


def test_summarize_tail_gain_quiet_when_nothing_moves():
    spectra = {
        "layer_15.self_attn.q": {"spectral_norm_ratio": 1.02},
        "layer_00.self_attn.in_proj": {"spectral_norm_ratio": 1.01},
    }
    io = {
        "layer_15": {
            "ln1_in_effective_rank_ratio": 0.99,
            "ln1_in_channel_var_ratio": 1.01,
            "attn_out_grad_rms_ratio": 1.03,
        }
    }
    assert summarize_tail_gain(spectra, io)["verdict"] == "quiet_weights"


def test_collect_tail_matrices_reads_packed_qkv_keys():
    d = 4
    state = {
        "body_model.seqTransEncoder.layers.15.self_attn.in_proj_weight": torch.randn(3 * d, d),
        "body_model.seqTransEncoder.layers.15.self_attn.out_proj.weight": torch.randn(d, d),
        "body_model.seqTransEncoder.layers.15.linear2.weight": torch.randn(d, 8),
        "body_model.output_linear.weight": torch.randn(6, d),
    }
    matrices = collect_tail_matrices(state, layers=(15,))
    assert "layer_15.self_attn.q" in matrices
    assert "layer_15.self_attn.out_proj" in matrices
    assert matrices["layer_15.self_attn.q"].shape == (d, d)
    assert "output_linear" in matrices
    assert math.isfinite(matrix_spectrum(matrices["layer_15.self_attn.q"])["spectral_norm"])
