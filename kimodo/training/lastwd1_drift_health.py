"""Health snapshot for the 750k last-layer wd=1 run.

Combines train.jsonl gradient windows with the residual-cancel cosine probe
photographed at the same checkpoint. Thresholds come from the wd03 parent:
750k was quality-best (L15≈0.13, cosine≈−0.02, clip=0); 790k had already
flipped (L15≈0.50, cosine≈−0.78); 800k locked (clip=1, L15≈15).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


PARENT_STEP = 750_000
PARENT_L15 = 0.13
PARENT_COSINE = -0.02
PARENT_CLIP = 0.0
PARENT_GNORM = 0.47

L15_DRIFT = 0.25
L15_EXPLODE = 0.50
CLIP_DRIFT = 0.05
CLIP_EXPLODE = 0.50
COSINE_DRIFT = -0.30
COSINE_EXPLODE = -0.60
GNORM_DRIFT = 0.80
GNORM_EXPLODE = 2.0


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def nearest_jsonl_row(rows: list[dict[str, Any]], step: int) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_distance: int | None = None
    for row in rows:
        current = row.get("global_step")
        try:
            current_step = int(current)
        except (TypeError, ValueError):
            continue
        distance = abs(current_step - int(step))
        if best_distance is None or distance < best_distance:
            best = row
            best_distance = distance
    return best


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


def cosine_from_probe(payload: dict[str, Any], step: int, layer: int = 15) -> float | None:
    key = f"L{int(layer):02d}_mean_token_cosine"
    timeline = payload.get("timeline") or []
    if not isinstance(timeline, list):
        return None
    for row in timeline:
        if not isinstance(row, dict):
            continue
        try:
            if int(row.get("global_step")) != int(step):
                continue
        except (TypeError, ValueError):
            continue
        value = _finite(row.get(key))
        if value is not None:
            return value
    return None


def classify_health(
    *,
    l15: float | None,
    clip: float | None,
    gnorm: float | None,
    cosine: float | None,
) -> str:
    exploding = (
        (clip is not None and clip >= CLIP_EXPLODE)
        or (l15 is not None and l15 >= L15_EXPLODE)
        or (gnorm is not None and gnorm >= GNORM_EXPLODE)
        or (cosine is not None and cosine <= COSINE_EXPLODE)
    )
    if exploding:
        return "exploding"
    drifting = (
        (clip is not None and clip >= CLIP_DRIFT)
        or (l15 is not None and l15 >= L15_DRIFT)
        or (gnorm is not None and gnorm >= GNORM_DRIFT)
        or (cosine is not None and cosine <= COSINE_DRIFT)
    )
    if drifting:
        return "drifting"
    return "healthy"


def build_health_record(
    *,
    step: int,
    jsonl_row: dict[str, Any] | None,
    probe_payload: dict[str, Any] | None,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    row = jsonl_row or {}
    l15 = _finite(row.get("optimizer/body_layer_15_grad_norm"))
    clip = _finite(row.get("optimizer/gradient_clip_fraction"))
    gnorm = _finite(row.get("optimizer/gradient_norm_before_clip"))
    cosine = cosine_from_probe(probe_payload or {}, step)
    parent_cosine = cosine_from_probe(probe_payload or {}, PARENT_STEP)
    status = classify_health(l15=l15, clip=clip, gnorm=gnorm, cosine=cosine)
    return {
        "step": int(step),
        "checkpoint": checkpoint,
        "jsonl_step": row.get("global_step"),
        "health": status,
        "optimizer": {
            "learning_rate": _finite(row.get("optimizer/learning_rate")),
            "weight_decay": _finite(row.get("optimizer/weight_decay")),
            "last_layer_weight_decay": _finite(row.get("optimizer/last_layer_weight_decay")),
            "clip_fraction": clip,
            "grad_norm_before_clip": gnorm,
            "l15_grad_norm": l15,
            "l00_grad_norm": _finite(row.get("optimizer/body_layer_00_grad_norm")),
        },
        "drift": {
            "l15_attn_cosine": cosine,
            "parent_750k_attn_cosine": parent_cosine if parent_cosine is not None else PARENT_COSINE,
            "probe_verdict": ((probe_payload or {}).get("verdict") or {}).get("verdict"),
        },
        "parent_750k": {
            "l15_grad_norm": PARENT_L15,
            "attn_cosine": PARENT_COSINE,
            "clip_fraction": PARENT_CLIP,
            "grad_norm_before_clip": PARENT_GNORM,
        },
        "thresholds": {
            "l15_drift": L15_DRIFT,
            "l15_explode": L15_EXPLODE,
            "cosine_drift": COSINE_DRIFT,
            "cosine_explode": COSINE_EXPLODE,
        },
    }
