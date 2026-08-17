from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from kimodo.training.body_attention_probe import (
    BodyAttentionHooks,
    attention_pointer_stats,
    compare_attention_rows,
    keyframe_frame_mask,
    probe_body_attention,
    summarize_pointer_grid,
)
from kimodo.training.config import CurriculumConfig, ModelConfig
from kimodo.training.modeling import build_trainable_denoiser


def test_keyframe_frame_mask_any_channel():
    mask = torch.zeros(2, 4, 3, dtype=torch.bool)
    mask[0, 1, 2] = True
    mask[1, :, 0] = True
    frames = keyframe_frame_mask(mask)
    assert frames.tolist() == [[False, True, False, False], [True, True, True, True]]


def test_uniform_attention_mass_matches_keyframe_fraction():
    seq = 6
    prefix = 2
    frames = 4
    weights = torch.full((1, 1, seq, seq), 1.0 / seq)
    valid = torch.ones(1, frames, dtype=torch.bool)
    keyframe = torch.tensor([[True, False, True, False]])
    stats = attention_pointer_stats(weights, valid_frames=valid, keyframe_frames=keyframe)
    unconstrained = stats["unconstrained"]
    assert unconstrained["keyframe_mass"] == pytest.approx(2.0 / 6.0)
    assert unconstrained["prefix_mass"] == pytest.approx(2.0 / 6.0)
    assert unconstrained["uniform_keyframe_mass"] == pytest.approx(0.5)
    assert unconstrained["entropy"] == pytest.approx(math.log(seq), rel=1e-5)
    assert unconstrained["max_prob"] == pytest.approx(1.0 / seq)
    assert unconstrained["normalized_entropy"] == pytest.approx(1.0, rel=1e-4)


def test_one_hot_keyframe_attention_is_peaked():
    seq = 5
    prefix = 1
    frames = 4
    weights = torch.zeros(1, 2, seq, seq)
    key_index = prefix  # first motion token, a keyframe
    weights[:, :, prefix:, key_index] = 1.0
    valid = torch.ones(1, frames, dtype=torch.bool)
    keyframe = torch.tensor([[True, False, False, False]])
    stats = attention_pointer_stats(weights, valid_frames=valid, keyframe_frames=keyframe)
    unconstrained = stats["unconstrained"]
    assert unconstrained["entropy"] == pytest.approx(0.0, abs=1e-6)
    assert unconstrained["max_prob"] == pytest.approx(1.0)
    assert unconstrained["keyframe_mass"] == pytest.approx(1.0)
    assert unconstrained["prefix_mass"] == pytest.approx(0.0)
    assert unconstrained["keyframe_lift"] == pytest.approx(4.0)


def test_body_attention_hooks_capture_transformer_weights():
    layer = nn.TransformerEncoderLayer(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        batch_first=True,
        norm_first=False,
        dropout=0.0,
    )
    body = nn.Module()
    body.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
    tokens = torch.randn(2, 6, 8)
    with BodyAttentionHooks(body) as hooks:
        body.seqTransEncoder(tokens)
    assert set(hooks.weights) == {0, 1}
    for index, maps in hooks.weights.items():
        assert maps.shape[-1] == maps.shape[-2] == 6
        assert maps.shape[0] == 2
        row_sum = maps.float().sum(dim=-1)
        assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4)
    # After exit, a second forward must not keep filling the captured dict.
    captured = dict(hooks.weights)
    body.seqTransEncoder(tokens)
    assert hooks.weights.keys() == captured.keys()
    for index in captured:
        assert hooks.weights[index].data_ptr() == captured[index].data_ptr()


def test_tiny_denoiser_attention_probe_reports_last_layer(training_fixture):
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
    valid = torch.ones(batch, frames, dtype=torch.bool)
    text = torch.randn(batch, 2, 16)
    text_mask = torch.ones(batch, 2, dtype=torch.bool)
    timesteps = torch.tensor([12, 40])
    heading = torch.zeros(batch)
    motion_mask = torch.zeros(batch, frames, dim, dtype=torch.bool)
    motion_mask[:, [0, 4], 0] = True
    observed = torch.zeros(batch, frames, dim)
    report = probe_body_attention(
        model,
        noisy=noisy,
        valid_frames=valid,
        text_features=text,
        text_pad_mask=text_mask,
        timesteps=timesteps,
        first_heading_angle=heading,
        motion_mask=motion_mask,
        observed_motion=observed,
    )
    assert len(report["layers"]) == 2
    last = report["layers"][-1]
    unc = last["unconstrained"]
    assert last["index"] == 1
    assert last["prefix_len"] >= 3
    assert 0.0 <= unc["keyframe_mass"] <= 1.0
    assert 0.0 <= unc["prefix_mass"] <= 1.0
    assert math.isfinite(unc["entropy"])
    assert unc["max_prob"] > 0.0
    assert report["keyframe_frame_fraction"] == pytest.approx(0.25)


def _attn_probe(*, entropy: float, mass: float, inner_entropy: float = 2.0, inner_mass: float = 0.2) -> dict:
    def layer(index: int, ent: float, key_mass: float) -> dict:
        return {
            "index": index,
            "unconstrained": {
                "entropy": ent,
                "normalized_entropy": ent / 4.0,
                "max_prob": 0.4 if index == 15 else 0.1,
                "keyframe_mass": key_mass,
                "keyframe_lift": key_mass / 0.2,
                "prefix_mass": 0.1,
            },
            "all_valid": {"entropy": ent},
        }

    return {"layers": [layer(0, inner_entropy, inner_mass), layer(15, entropy, mass)]}


def _cell(weight_step: int, constraint_step: int, probe: dict) -> dict:
    return {"weight_step": weight_step, "constraint_step": constraint_step, "probe": probe}


def test_pointer_grid_labels_last_layer_collapse_not_clock():
    healthy = _attn_probe(entropy=2.0, mass=0.20)
    takeoff = _attn_probe(entropy=1.2, mass=0.45)
    report = summarize_pointer_grid(
        [
            _cell(795000, 795000, healthy),
            _cell(800000, 795000, takeoff),
            _cell(795000, 800000, healthy),
            _cell(800000, 800000, takeoff),
        ]
    )
    assert report["verdict"] == "weights"
    assert report["weight_verdict"] == "pointer"
    assert report["clock_verdict"] == "not_attention"
    last = report["weight_at_healthy_clock"]
    assert last["unconstrained_entropy_ratio"] == pytest.approx(0.6)
    assert last["unconstrained_keyframe_mass_delta"] == pytest.approx(0.25)


def test_pointer_grid_quiet_when_entropy_stays():
    healthy = _attn_probe(entropy=2.0, mass=0.20)
    report = summarize_pointer_grid(
        [
            _cell(795000, 795000, healthy),
            _cell(800000, 795000, healthy),
            _cell(795000, 800000, healthy),
            _cell(800000, 800000, healthy),
        ]
    )
    assert report["verdict"] == "not_attention"


def test_compare_attention_rows_ratio_on_last_layer():
    rows = [
        {"global_step": 795000, "probe": _attn_probe(entropy=2.0, mass=0.2)},
        {"global_step": 800000, "probe": _attn_probe(entropy=1.0, mass=0.5)},
    ]
    comparison = compare_attention_rows(rows)
    last = comparison[1]["layers"]["layer_15"]
    assert last["unconstrained_entropy_ratio"] == pytest.approx(0.5)
    assert last["unconstrained_keyframe_mass_delta"] == pytest.approx(0.3)
    inner = comparison[1]["layers"]["layer_00"]
    assert inner["unconstrained_entropy_ratio"] == pytest.approx(1.0)
