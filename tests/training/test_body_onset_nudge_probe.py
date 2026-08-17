from __future__ import annotations

import math

import pytest
import torch

from kimodo.training.body_onset_nudge_probe import (
    shift_branch_cosine,
    summarize_onset_nudge,
)


def test_shift_identity_at_zero_delta():
    residual = torch.randn(2, 3, 8)
    branch = torch.randn(2, 3, 8)
    out = shift_branch_cosine(residual, branch, 0.0)
    assert torch.equal(out, branch)


def test_shift_preserves_branch_magnitude():
    residual = torch.randn(2, 5, 16)
    branch = torch.randn(2, 5, 16)
    before = branch.norm(dim=-1)
    after = shift_branch_cosine(residual, branch, -0.25).norm(dim=-1)
    assert torch.allclose(before, after, rtol=1e-5, atol=1e-5)


def test_negative_delta_lowers_cosine():
    residual = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    # Cosine with x-axis ≈ +0.22
    branch = torch.tensor([[[0.22, math.sqrt(1.0 - 0.22**2), 0.0, 0.0]]])
    before = torch.nn.functional.cosine_similarity(residual, branch, dim=-1)
    out = shift_branch_cosine(residual, branch, -0.25)
    after = torch.nn.functional.cosine_similarity(residual, out, dim=-1)
    assert float(before) == pytest.approx(0.22, abs=1e-5)
    assert float(after) == pytest.approx(-0.03, abs=1e-4)
    assert float(after) < float(before)


def test_positive_delta_raises_cosine():
    residual = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    branch = torch.tensor([[[0.22, math.sqrt(1.0 - 0.22**2), 0.0, 0.0]]])
    out = shift_branch_cosine(residual, branch, 0.25)
    after = torch.nn.functional.cosine_similarity(residual, out, dim=-1)
    assert float(after) == pytest.approx(0.47, abs=1e-4)


def test_shift_works_when_branch_already_parallel():
    residual = torch.ones(1, 2, 4)
    branch = 3 * residual
    out = shift_branch_cosine(residual, branch, -0.25)
    cosine = torch.nn.functional.cosine_similarity(residual, out, dim=-1)
    assert torch.allclose(cosine, torch.full_like(cosine, 0.75), atol=1e-4)


def _row(step: int, delta: float, targets, loss: float, terms=None) -> dict:
    means = {"total": loss}
    if terms:
        means.update(terms)
    return {
        "global_step": step,
        "cosine_delta": delta,
        "targets": list(targets),
        "probe": {"loss_total_mean": loss, "loss_means": means},
    }


def _triplet(step: int, identity: float, cancel: float, add: float) -> list[dict]:
    return [
        _row(step, 0.0, ("attn",), identity),
        _row(step, -0.25, ("attn",), cancel),
        _row(step, 0.25, ("attn",), add),
        _row(step, -0.25, ("ffn",), identity),
        _row(step, 0.25, ("ffn",), identity),
    ]


def test_summarize_rewards_when_preflip_cancel_lowers_loss():
    rows = (
        _triplet(650000, 0.50, 0.50, 0.50)
        + _triplet(690000, 0.50, 0.47, 0.51)
        + _triplet(695000, 0.50, 0.50, 0.50)
    )
    report = summarize_onset_nudge(rows)
    assert report["verdict"] == "loss_rewards_onset"
    assert report["preflip_attn"]["more_cancel_loss_ratio"] == pytest.approx(0.94)


def test_summarize_punishes_when_preflip_cancel_raises_loss():
    rows = (
        _triplet(650000, 0.50, 0.50, 0.50)
        + _triplet(690000, 0.50, 0.53, 0.49)
        + _triplet(695000, 0.50, 0.50, 0.50)
    )
    report = summarize_onset_nudge(rows)
    assert report["verdict"] == "loss_punishes_onset"


def test_summarize_indifferent_when_loss_barely_moves():
    rows = (
        _triplet(650000, 0.50, 0.5001, 0.5001)
        + _triplet(690000, 0.50, 0.501, 0.5005)
        + _triplet(695000, 0.50, 0.5002, 0.5002)
    )
    report = summarize_onset_nudge(rows)
    assert report["verdict"] == "loss_indifferent_at_onset"


def test_summarize_any_nudge_hurts_is_not_a_direction():
    rows = (
        _triplet(650000, 0.50, 0.56, 0.55)
        + _triplet(690000, 0.50, 0.56, 0.55)
        + _triplet(695000, 0.50, 0.56, 0.55)
    )
    report = summarize_onset_nudge(rows)
    assert report["verdict"] == "any_nudge_hurts"
