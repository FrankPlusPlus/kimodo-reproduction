from __future__ import annotations

from kimodo.training.body_firstcause_config import (
    summarize_multibatch_drift,
    summarize_official_vs_ours,
)
from kimodo.training.body_onset_path_probe import _variant_optimizer_knobs


def test_official_stays_aligned_ours_flips():
    rows = [
        {
            "label": "official",
            "probe": {"layers": [{"index": 15, "mean_token_cosine": 0.22}]},
        },
        {
            "label": "ours-790k",
            "global_step": 790000,
            "probe": {"layers": [{"index": 15, "mean_token_cosine": -0.77}]},
        },
    ]
    report = summarize_official_vs_ours(rows)
    assert report["verdict"] == "our_recipe_flips_official_stays_aligned"


def test_official_also_flipped():
    rows = [
        {"label": "official", "probe": {"layers": [{"index": 15, "mean_token_cosine": -0.4}]}},
        {"label": "ours", "global_step": 790000, "probe": {"layers": [{"index": 15, "mean_token_cosine": -0.7}]}},
    ]
    assert summarize_official_vs_ours(rows)["verdict"] == "official_also_flipped"


def test_our_knobs_ranked_most_negative():
    rows = [
        {"variant": "atan2", "precision": "bf16", "deltas": [-0.002, -0.001]},
        {"variant": "adam", "precision": "bf16", "deltas": [0.0001, 0.0]},
        {"variant": "atan2_wd03", "precision": "bf16", "deltas": [-0.0002, 0.0]},
    ]
    report = summarize_multibatch_drift(rows, drift_cut=1e-4)
    assert report["verdict"] == "our_filled_knobs_drive_flip"
    assert report["ranked"][0]["variant"] == "atan2"


def test_new_virtual_variants_parse():
    assert _variant_optimizer_knobs("atan2_wd03", default_lambda=8.0)["weight_decay"] == 0.3
    assert _variant_optimizer_knobs("atan2_lambda1", default_lambda=8.0)["atan2_lambda"] == 1.0
