from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kimodo.training.body_jacobian_probe import (
    BodyLayerHooks,
    alignment_ratio,
    compare_checkpoint_rows,
    pairwise_cosines,
    probe_forward_backward,
)
from kimodo.training.config import CurriculumConfig, LossConfig, ModelConfig
from kimodo.training.losses import KimodoLoss
from kimodo.training.modeling import build_trainable_denoiser


def test_pairwise_cosine_identical_and_orthogonal():
    ones = torch.ones(8)
    report = pairwise_cosines([ones, 3 * ones])
    assert report["pair_count"] == 1
    assert report["mean_cosine"] == pytest.approx(1.0)
    assert report["mean_angle_deg"] == pytest.approx(0.0)

    left = torch.tensor([1.0, 0.0, 0.0])
    right = torch.tensor([0.0, 1.0, 0.0])
    ortho = pairwise_cosines([left, right])
    assert ortho["mean_cosine"] == pytest.approx(0.0)
    assert ortho["mean_angle_deg"] == pytest.approx(90.0)


def test_alignment_ratio_collapses_when_directions_agree():
    vector = torch.tensor([3.0, 4.0])
    aligned = alignment_ratio([vector, 2 * vector, 0.5 * vector])
    assert aligned["ratio"] == pytest.approx(1.0)

    opposite = alignment_ratio([vector, -vector])
    assert opposite["mean_vector_norm"] == pytest.approx(0.0)
    assert opposite["ratio"] == pytest.approx(0.0)


def test_body_layer_hooks_record_ln_variance_and_output_rms():
    layer = nn.TransformerEncoderLayer(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        batch_first=True,
        norm_first=False,
        dropout=0.0,
    )
    body = SimpleNamespace(
        seqTransEncoder=nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
    )
    tokens = torch.randn(2, 5, 8)
    with BodyLayerHooks(body) as hooks:
        body.seqTransEncoder(tokens)
    assert len(hooks.layers) == 2
    for slot in hooks.layers:
        assert math.isfinite(slot["ln1_in_var"])
        assert math.isfinite(slot["ln2_in_var"])
        assert slot["ln1_in_var"] >= 0
        assert math.isfinite(slot["output_rms"])
        assert slot["output_rms"] > 0


def test_tiny_denoiser_probe_splits_gain_from_prediction_grad(training_fixture):
    config = ModelConfig(
        skeleton_joints=30,
        stats_path=str(training_fixture["stats"]),
        llm_dim=16,
        num_text_tokens_override=2,
        latent_dim=16,
        ff_size=32,
        num_layers=2,
        num_heads=4,
    )
    curriculum = CurriculumConfig(phase1_steps=1, phase2_steps=1)
    model = build_trainable_denoiser(config, curriculum, torch.device("cpu"))
    model.train()
    frames = 8
    dim = model.motion_rep.motion_rep_dim
    batch = 2
    noisy = torch.randn(batch, frames, dim)
    target = torch.randn(batch, frames, dim)
    valid = torch.ones(batch, frames, dtype=torch.bool)
    text = torch.randn(batch, 2, 16)
    text_mask = torch.ones(batch, 2, dtype=torch.bool)
    timesteps = torch.tensor([12, 40])
    heading = torch.zeros(batch)
    motion_mask = torch.zeros(batch, frames, dim, dtype=torch.bool)
    observed = torch.zeros(batch, frames, dim)
    loss = KimodoLoss(model.motion_rep, LossConfig())

    baseline = probe_forward_backward(
        model,
        noisy=noisy,
        valid_frames=valid,
        text_features=text,
        text_pad_mask=text_mask,
        timesteps=timesteps,
        first_heading_angle=heading,
        motion_mask=motion_mask,
        observed_motion=observed,
        target=target,
        loss_function=loss,
        pair_samples=2,
    )
    assert math.isfinite(baseline["prediction_grad_norm"])
    assert math.isfinite(baseline["body_batch_grad_norm"])
    assert "body.layer_00" in baseline["grad_norms"]
    assert "body.layer_01" in baseline["grad_norms"]
    assert len(baseline["body_activation_layers"]) == 2
    assert baseline["per_sample"]["pairwise"]["pair_count"] == 1
    assert math.isfinite(baseline["per_sample"]["alignment"]["ratio"])

    scaled = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    for name, tensor in scaled.items():
        if name.startswith("body_model.seqTransEncoder"):
            scaled[name] = tensor * 1.5
    model.load_state_dict(scaled)
    grown = probe_forward_backward(
        model,
        noisy=noisy,
        valid_frames=valid,
        text_features=text,
        text_pad_mask=text_mask,
        timesteps=timesteps,
        first_heading_angle=heading,
        motion_mask=motion_mask,
        observed_motion=observed,
        target=target,
        loss_function=loss,
        pair_samples=2,
    )
    comparison = compare_checkpoint_rows(
        [
            {"global_step": 650000, "probe": baseline},
            {"global_step": 696000, "probe": grown},
        ]
    )
    assert comparison[1]["body_batch_grad_norm_ratio"] != pytest.approx(1.0, rel=1e-3)
    assert math.isfinite(comparison[1]["prediction_grad_norm_ratio"])
    assert all(parameter.grad is None for parameter in model.parameters())
