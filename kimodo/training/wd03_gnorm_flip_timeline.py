"""Experiment U: align wd03 train.jsonl gnorm with L15 attn cosine timeline."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def bucket_train_jsonl(
    rows: list[dict[str, Any]],
    *,
    step_start: int,
    step_every: int,
    step_end: int | None = None,
) -> list[dict[str, Any]]:
    """Aggregate jsonl windows into step buckets (default 10k)."""
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            step = int(row.get("global_step", 0))
        except (TypeError, ValueError):
            continue
        if step < int(step_start):
            continue
        if step_end is not None and step > int(step_end):
            continue
        key = (step // int(step_every)) * int(step_every)
        buckets.setdefault(key, []).append(row)

    timeline: list[dict[str, Any]] = []
    for key in sorted(buckets):
        group = buckets[key]
        gnorms = [
            float(row["optimizer/gradient_norm_before_clip"])
            for row in group
            if row.get("optimizer/gradient_norm_before_clip") is not None
        ]
        clips = [
            float(row.get("optimizer/gradient_clip_fraction") or 0.0) for row in group
        ]
        l15 = [
            float(row["body_layer_15_grad_norm"])
            for row in group
            if row.get("body_layer_15_grad_norm") is not None
        ]
        entry: dict[str, Any] = {
            "step": key,
            "n_rows": len(group),
            "gnorm_median": statistics.median(gnorms) if gnorms else None,
            "gnorm_p10": (
                statistics.quantiles(gnorms, n=10)[0] if len(gnorms) >= 10 else (min(gnorms) if gnorms else None)
            ),
            "clip_fraction_mean": statistics.mean(clips) if clips else None,
            "l15_grad_median": statistics.median(l15) if l15 else None,
        }
        timeline.append(entry)
    return timeline


def cosine_points_from_full_stack(full_stack: list[dict[str, Any]], *, layer: int = 15) -> list[dict[str, Any]]:
    prefix = f"L{int(layer):02d}"
    points: list[dict[str, Any]] = []
    for row in full_stack:
        step = row.get("global_step")
        attn = row.get(f"{prefix}_attn_cosine")
        ffn = row.get(f"{prefix}_ffn_cosine")
        if step is None:
            continue
        points.append(
            {
                "step": int(step),
                "attn_cosine": attn,
                "ffn_cosine": ffn,
                "attn_sigma": row.get(f"{prefix}_attn_sigma"),
                "ffn_sigma": row.get(f"{prefix}_ffn_sigma"),
            }
        )
    return sorted(points, key=lambda item: int(item["step"]))


def merge_gnorm_cosine(
    gnorm_timeline: list[dict[str, Any]],
    cosine_points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cos_by_step = {int(item["step"]): item for item in cosine_points}
    merged: list[dict[str, Any]] = []
    for row in gnorm_timeline:
        step = int(row["step"])
        cos = cos_by_step.get(step, {})
        merged.append({**row, **cos})
    for step, cos in cos_by_step.items():
        if not any(int(row["step"]) == step for row in merged):
            merged.append({"step": step, **cos})
    return sorted(merged, key=lambda item: int(item["step"]))


def _first_step(timeline: list[dict[str, Any]], predicate) -> int | None:
    for row in timeline:
        if predicate(row):
            try:
                return int(row["step"])
            except (TypeError, ValueError, KeyError):
                continue
    return None


def summarize_gnorm_flip_relation(
    merged: list[dict[str, Any]],
    *,
    gnorm_p10_baseline: float = 0.41,
    gnorm_p10_rise: float = 0.55,
    cosine_flip: float = 0.0,
    event_step: int = 696_600,
) -> dict[str, Any]:
    """Decide whether 696.6k gnorm wall precedes L15 attn flip."""
    gnorm_rise = _first_step(
        merged,
        lambda row: isinstance(row.get("gnorm_p10"), (int, float))
        and float(row["gnorm_p10"]) >= gnorm_p10_rise
        and int(row["step"]) >= 650_000,
    )
    attn_flip = _first_step(
        merged,
        lambda row: isinstance(row.get("attn_cosine"), (int, float))
        and float(row["attn_cosine"]) <= cosine_flip,
    )
    near_696 = None
    for row in merged:
        step = int(row["step"])
        if abs(step - int(event_step)) <= 20_000:
            p10 = row.get("gnorm_p10")
            cos = row.get("attn_cosine")
            if isinstance(p10, (int, float)) and float(p10) >= gnorm_p10_rise:
                near_696 = step
                break
            if isinstance(cos, (int, float)) and float(cos) <= cosine_flip:
                near_696 = step
                break

    verdict = "incomplete"
    if gnorm_rise is not None and attn_flip is not None:
        if gnorm_rise < attn_flip - 10_000:
            verdict = "gnorm_rise_before_flip"
        elif attn_flip < gnorm_rise - 10_000:
            verdict = "flip_before_gnorm_rise"
        else:
            verdict = "gnorm_and_flip_same_era"
    elif gnorm_rise is not None:
        verdict = "gnorm_rise_without_measured_flip"
    elif attn_flip is not None:
        verdict = "flip_without_gnorm_rise"

    return {
        "verdict": verdict,
        "gnorm_p10_rise_step": gnorm_rise,
        "attn_flip_step": attn_flip,
        "near_696k_signal_step": near_696,
        "gnorm_p10_baseline": gnorm_p10_baseline,
        "gnorm_p10_rise_threshold": gnorm_p10_rise,
        "cosine_flip_threshold": cosine_flip,
        "event_step_hint": int(event_step),
    }


def build_report(
    *,
    train_jsonl: Path,
    full_stack_timeline: list[dict[str, Any]] | None = None,
    step_start: int = 650_000,
    step_every: int = 10_000,
    step_end: int = 800_000,
) -> dict[str, Any]:
    rows = load_jsonl_rows(train_jsonl)
    gnorm_timeline = bucket_train_jsonl(
        rows, step_start=step_start, step_every=step_every, step_end=step_end
    )
    cosine_points = cosine_points_from_full_stack(full_stack_timeline or [])
    merged = merge_gnorm_cosine(gnorm_timeline, cosine_points)
    summary = summarize_gnorm_flip_relation(merged)
    return {
        "train_jsonl": str(train_jsonl),
        "gnorm_timeline": gnorm_timeline,
        "cosine_points": cosine_points,
        "merged": merged,
        "summary": summary,
    }
