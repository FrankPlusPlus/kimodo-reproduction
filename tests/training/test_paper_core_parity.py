from __future__ import annotations

import torch
from torch import nn

from kimodo.model.kimodo_model import _resolve_multiprompt_inputs
from kimodo.model.twostage_denoiser import TwostageDenoiser
from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton
from kimodo.training.config import LossConfig
from kimodo.training.losses import KimodoLoss
from kimodo.training.modeling import set_model_dropout


class _RootStage(nn.Module):
    def __init__(self, root: torch.Tensor) -> None:
        super().__init__()
        self.root = nn.Parameter(root)
        self.seen: torch.Tensor | None = None

    def forward(self, x, *_args):
        self.seen = x.detach().clone()
        return self.root.expand(x.shape[0], -1, -1)


class _BodyStage(nn.Module):
    def __init__(self, body_dim: int) -> None:
        super().__init__()
        self.body_dim = body_dim
        self.seen: torch.Tensor | None = None

    def forward(self, x, *_args):
        self.seen = x.detach().clone()
        # Depend on the predicted local root so this also probes the paper's
        # end-to-end gradient path from body prediction into the root stage.
        return x[..., :1].expand(*x.shape[:2], self.body_dim)


def test_two_stage_paper_dataflow_and_default_end_to_end(training_fixture):
    """Fig. 9 / Sec. 4.2: impute -> root -> local root -> body -> concat."""
    motion_rep = KimodoMotionRep(
        build_skeleton(30), fps=30, stats_path=str(training_fixture["stats"])
    )
    model = TwostageDenoiser(
        motion_rep=motion_rep,
        motion_mask_mode="concat",
        llm_shape=[2, 16],
        use_text_mask=False,
        latent_dim=16,
        ff_size=32,
        num_layers=1,
        num_heads=4,
        activation="gelu",
        dropout=0.0,
        pe_dropout=0.0,
        norm_first=False,
        num_text_tokens_override=2,
        input_first_heading_angle=True,
    )
    assert model.detach_root_for_body is False

    batch, frames, dim = 1, 4, motion_rep.motion_rep_dim
    root_prediction = torch.zeros(1, frames, motion_rep.global_root_dim)
    # A non-constant trajectory makes finite-difference local-root features
    # carry gradients to the root prediction.
    root_prediction[0, :, 0] = torch.arange(frames, dtype=torch.float32)
    root_prediction[0, :, 3] = 1.0  # heading [cos, sin] = [1, 0]
    root_stage = _RootStage(root_prediction)
    body_stage = _BodyStage(dim - motion_rep.global_root_dim)
    model.root_model = root_stage
    model.body_model = body_stage

    noisy = torch.randn(batch, frames, dim)
    observed = torch.randn_like(noisy)
    control_mask = torch.zeros_like(noisy, dtype=torch.bool)
    control_mask[..., ::17] = True
    valid = torch.ones(batch, frames, dtype=torch.bool)
    output = model(
        noisy,
        valid,
        torch.zeros(batch, 2, 16),
        torch.ones(batch, 2, dtype=torch.bool),
        torch.tensor([7]),
        first_heading_angle=torch.zeros(batch),
        motion_mask=control_mask,
        observed_motion=observed,
    )

    imputed = torch.where(control_mask, observed, noisy)
    assert root_stage.seen is not None
    assert torch.equal(root_stage.seen[..., :dim], imputed)
    assert torch.equal(root_stage.seen[..., dim:], control_mask.to(noisy.dtype))

    expected_local_root = motion_rep.global_root_to_local_root(
        root_stage.root.expand(batch, -1, -1), normalized=True, lengths=valid.sum(-1)
    )
    expected_body_input = torch.cat(
        [
            expected_local_root,
            imputed[..., motion_rep.body_slice],
            control_mask.to(noisy.dtype),
        ],
        dim=-1,
    )
    assert body_stage.seen is not None
    assert torch.allclose(body_stage.seen, expected_body_input)
    assert torch.equal(output[..., : motion_rep.global_root_dim], root_stage.root)

    output[..., motion_rep.body_slice].sum().backward()
    assert root_stage.root.grad is not None
    assert root_stage.root.grad.abs().sum() > 0


def test_q_sample_is_clean_motion_prediction_forward_process():
    """Sec. 4.3: x_t = sqrt(alpha_bar)x_0 + sqrt(1-alpha_bar)epsilon."""
    from kimodo.model.diffusion import Diffusion

    diffusion = Diffusion(1000)
    clean = torch.randn(2, 3, 5)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([0, 999])
    actual = diffusion.q_sample(clean, timesteps, noise)
    expected = (
        diffusion.sqrt_alphas_cumprod[timesteps, None, None] * clean
        + diffusion.sqrt_one_minus_alphas_cumprod[timesteps, None, None] * noise
    )
    assert torch.equal(actual, expected)


def test_eq1_has_all_six_direct_terms_exact_weights_and_valid_frame_reduction(
    training_fixture,
):
    motion_rep = KimodoMotionRep(
        build_skeleton(30), fps=30, stats_path=str(training_fixture["stats"])
    )
    config = LossConfig(direct_feature_domain="normalized", smooth_l1_beta=1.0)
    criterion = KimodoLoss(motion_rep, config)
    target = torch.zeros(1, 3, motion_rep.motion_rep_dim)
    valid = torch.tensor([[True, True, False]])

    assert {
        "root_position": config.root_position,
        "root_heading": config.root_heading,
        "joint_position": config.joint_position,
        "joint_velocity": config.joint_velocity,
        "joint_rotation": config.joint_rotation,
        "foot_contact": config.foot_contact,
        "forward_kinematics": config.forward_kinematics,
    } == {
        "root_position": 10.0,
        "root_heading": 2.0,
        "joint_position": 10.0,
        "joint_velocity": 3.0,
        "joint_rotation": 10.0,
        "foot_contact": 4.0,
        "forward_kinematics": 5.0,
    }

    # SmoothL1(beta=1) at delta=0.5 is 0.5 * delta^2 = 0.125.
    # The invalid third frame must not affect the per-valid-frame mean.
    for term_name, feature_name in criterion.FEATURE_TERMS:
        prediction = target.clone()
        prediction[..., motion_rep.slice_dict[feature_name]] = 0.5
        losses = criterion(prediction, target, valid)
        assert torch.allclose(losses[term_name], torch.tensor(0.125))


def test_paper_phase_dropout_values_reach_attention_and_embedding_dropout():
    module = nn.Sequential(
        nn.Dropout(0.0),
        nn.MultiheadAttention(8, 2, dropout=0.0, batch_first=True),
    )
    set_model_dropout(module, 0.1)
    assert module[0].p == 0.1
    assert module[1].dropout == 0.1
    set_model_dropout(module, 0.0)
    assert module[0].p == 0.0
    assert module[1].dropout == 0.0


def test_multiprompt_segment_lengths_are_independent_of_sample_batch_size():
    prompts, frames, samples, squeeze = _resolve_multiprompt_inputs(
        ["walk", "jump", "sit"], 30, None
    )
    assert prompts == ["walk", "jump", "sit"]
    assert frames == [30, 30, 30]
    assert samples == 1 and squeeze

    _, frames, samples, squeeze = _resolve_multiprompt_inputs(
        ["walk", "jump", "sit"], 30, 2
    )
    assert frames == [30, 30, 30]
    assert samples == 2 and not squeeze

    import pytest

    with pytest.raises(ValueError, match="one value per prompt"):
        _resolve_multiprompt_inputs(["walk", "jump"], [30], 1)
