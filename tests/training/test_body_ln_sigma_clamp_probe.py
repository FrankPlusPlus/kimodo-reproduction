from __future__ import annotations

from kimodo.training.body_ln_sigma_clamp_probe import summarize_sigma_clamp


def _row(step: int, mode: str, attn_grad: float, *, floors=None, ffn_cosine=0.3, loss=0.24) -> dict:
    return {
        "global_step": step,
        "clamp_mode": mode,
        "sigma_floors": dict(floors or {}),
        "probe": {
            "loss_total_mean": loss,
            "ffn_cosine": ffn_cosine,
            "grad_norms": {"body.layer_15.self_attn": attn_grad},
        },
    }


def test_summarize_flags_sigma_when_clamp_drops_grads():
    rows = [
        _row(750000, "none", 1.0, ffn_cosine=0.4),
        _row(800000, "none", 2.0, ffn_cosine=-0.2),
        _row(800000, "backward", 1.1, floors={"norm1": 0.96, "norm2": 1.0}),
    ]
    report = summarize_sigma_clamp(rows)
    assert report["verdict"] == "sigma_amplifies_grads"
    assert report["backward_over_takeoff_attn_grad"] == 1.1 / 2.0


def test_summarize_not_sigma_when_clamp_does_nothing():
    rows = [
        _row(750000, "none", 1.0),
        _row(800000, "none", 2.0),
        _row(800000, "backward", 1.95, floors={"norm1": 0.96, "norm2": 1.0}),
    ]
    report = summarize_sigma_clamp(rows)
    assert report["verdict"] == "not_sigma"
