from __future__ import annotations

import torch

from kimodo.training.body_sigma_cause_probe import set_branch_cosine, summarize_sigma_cause


def test_set_branch_cosine_preserves_length_and_hits_target():
    residual = torch.ones(1, 2, 4)
    branch = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]])
    patched = set_branch_cosine(residual, branch, -1.0)
    cosine = (residual * patched).sum(-1) / (
        residual.norm(dim=-1) * patched.norm(dim=-1)
    )
    assert torch.allclose(patched.norm(dim=-1), branch.norm(dim=-1), atol=1e-5)
    assert torch.allclose(cosine, torch.full_like(cosine, -1.0), atol=1e-4)


def test_attn_scale_verdict_when_only_scale_recovers():
    rows = [
        {
            "name": "healthy_identity",
            "ln_sigma_mean": 0.96,
            "l15_attn_grad": 900.0,
        },
        {
            "name": "crashed_identity",
            "ln_sigma_mean": 0.44,
            "l15_attn_grad": 2600.0,
        },
        {
            "name": "crashed_attn_scale_to_healthy",
            "ln_sigma_mean": 0.90,
            "l15_attn_grad": 1000.0,
        },
        {
            "name": "crashed_cosine_to_healthy",
            "ln_sigma_mean": 0.50,
            "l15_attn_grad": 2400.0,
        },
        {
            "name": "healthy_attn_scale_to_crashed",
            "ln_sigma_mean": 0.50,
            "l15_attn_grad": 2000.0,
        },
        {
            "name": "healthy_cosine_to_crashed",
            "ln_sigma_mean": 0.90,
            "l15_attn_grad": 1100.0,
        },
    ]
    report = summarize_sigma_cause(rows)
    assert report["verdict"] == "attn_scale_causes_sigma_collapse"
    assert report["config_hint"] == "weight_decay_or_stop_before_attn_rms_growth"
