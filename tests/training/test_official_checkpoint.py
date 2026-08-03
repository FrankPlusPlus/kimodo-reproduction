from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from kimodo.training.modeling import load_official_trainable_denoiser
from kimodo.training.checkpoint import _replace_checkpoint_paths


@pytest.mark.official
def test_official_checkpoint_strict_load_forward_backward():
    bundle_value = os.environ.get("KIMODO_OFFICIAL_BUNDLE")
    if not bundle_value:
        pytest.skip("set KIMODO_OFFICIAL_BUNDLE to run the 1.1 GB official-checkpoint gate")
    bundle = Path(bundle_value).expanduser().resolve()
    model = load_official_trainable_denoiser(bundle, torch.device("cpu"))
    assert sum(parameter.numel() for parameter in model.parameters()) == 283_281_777
    model.train()
    output = model(
        torch.zeros(1, 2, 369),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 1, 4096),
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([999]),
        first_heading_angle=torch.zeros(1),
    )
    assert output.shape == (1, 2, 369)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_official_export_rewrites_weights_and_stats_to_bundle():
    rewritten = _replace_checkpoint_paths(
        {"denoiser": {"ckpt_path": "/old/model.safetensors", "motion_rep": {"stats_path": "/old/stats"}}}
    )
    assert rewritten["denoiser"]["ckpt_path"] == "${checkpoint_dir}/model.pt"
    assert rewritten["denoiser"]["motion_rep"]["stats_path"] == "${checkpoint_dir}/stats"
