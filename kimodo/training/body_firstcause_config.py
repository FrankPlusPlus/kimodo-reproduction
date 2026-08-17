"""Summarize official-vs-ours L15 geometry and multi-batch config drift.

Does not treat cosine as a cause. Ranks which *our filled-in knobs*
(wd, clip, λ, bf16, atan2 vs Adam) produce a more negative one-step
Δcos on fresh batches at a pre-flip checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _f(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def layer15(probe: dict[str, Any]) -> dict[str, Any]:
    for layer in (probe.get("layers") or []):
        if int(layer.get("index") or -1) == 15:
            return layer
    return {}


def summarize_official_vs_ours(
    rows: Sequence[dict[str, Any]],
    *,
    cosine_flip: float = 0.0,
) -> dict[str, Any]:
    official = next((row for row in rows if row.get("label") == "official"), None)
    ours = [row for row in rows if row.get("label") != "official"]
    official_cos = _f((layer15((official or {}).get("probe") or {})).get("mean_token_cosine"))
    flipped_ours = []
    for row in ours:
        cosine = _f(layer15(row.get("probe") or {}).get("mean_token_cosine"))
        if math.isfinite(cosine) and cosine <= cosine_flip:
            flipped_ours.append({"label": row.get("label"), "step": row.get("global_step"), "cosine": cosine})
    if not math.isfinite(official_cos):
        verdict = "incomplete"
    elif official_cos > cosine_flip and flipped_ours:
        verdict = "our_recipe_flips_official_stays_aligned"
    elif official_cos <= cosine_flip and flipped_ours:
        verdict = "official_also_flipped"
    elif official_cos > cosine_flip and not flipped_ours:
        verdict = "neither_flipped"
    else:
        verdict = "official_flipped_ours_aligned"
    return {
        "verdict": verdict,
        "official_l15_attn_cosine": official_cos,
        "ours_flipped": flipped_ours,
    }


def summarize_multibatch_drift(
    variant_rows: Sequence[dict[str, Any]],
    *,
    drift_cut: float = 1e-4,
) -> dict[str, Any]:
    ranked = []
    for row in variant_rows:
        deltas = [ _f(item) for item in (row.get("deltas") or []) ]
        finite = [value for value in deltas if math.isfinite(value)]
        mean = float(sum(finite) / len(finite)) if finite else float("nan")
        ranked.append(
            {
                "variant": row.get("variant"),
                "precision": row.get("precision"),
                "clip_norm": row.get("clip_norm"),
                "n": len(finite),
                "mean_delta": mean,
                "sum_delta": float(sum(finite)) if finite else float("nan"),
            }
        )
    ranked.sort(key=lambda item: (item["mean_delta"] if math.isfinite(item["mean_delta"]) else 0.0))
    ours = next(
        (
            item
            for item in ranked
            if item.get("variant") == "atan2" and str(item.get("precision")) == "bf16"
        ),
        ranked[0] if ranked else {},
    )
    controls = [item for item in ranked if item is not ours]
    more_negative = [
        item["variant"]
        for item in ranked
        if math.isfinite(item["mean_delta"])
        and math.isfinite(_f(ours.get("mean_delta")))
        and item["mean_delta"] < float(ours["mean_delta"]) - abs(drift_cut)
    ]
    ours_mean = _f(ours.get("mean_delta"))
    if not math.isfinite(ours_mean):
        verdict = "incomplete"
    elif ours_mean <= -drift_cut and not more_negative:
        verdict = "our_filled_knobs_drive_flip"
    elif ours_mean <= -drift_cut and more_negative:
        verdict = "several_knobs_drive_flip"
    elif ours_mean > drift_cut:
        verdict = "our_knobs_do_not_drive_flip_on_one_step"
    else:
        verdict = "one_step_drift_below_cut"
    return {
        "verdict": verdict,
        "ranked": ranked,
        "ours_mean_delta": ours_mean,
        "more_negative_than_ours": more_negative,
        "n_controls": len(controls),
    }
