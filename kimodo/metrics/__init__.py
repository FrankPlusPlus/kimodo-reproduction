# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation metrics for motion quality (foot skate, contact consistency, constraint following)."""

from .base import (
    Metric,
    aggregate_metrics,
    clear_metrics,
    compute_metrics,
)
from .constraints import (
    PAPER_END_EFFECTOR_POSITION,
    PAPER_END_EFFECTOR_ROTATION,
    PAPER_FULLBODY_POSITION,
    PAPER_PELVIS_TO_SMOOTH_ROOT_2D,
    PAPER_SMOOTH_ROOT_2D,
    ContraintFollow,
    PaperConstraintFollow,
    compute_paper_constraint_errors,
    rotation_geodesic_degrees,
)
from .foot_skate import (
    FootContactConsistency,
    FootSkateFromContacts,
    FootSkateFromHeight,
    FootSkateRatio,
)
from .tmr import (
    TMR_EmbeddingMetric,
    TMR_Metric,
    compute_tmr_per_sample_retrieval,
    compute_tmr_retrieval_metrics,
)

__all__ = [
    "Metric",
    "ContraintFollow",
    "PaperConstraintFollow",
    "PAPER_FULLBODY_POSITION",
    "PAPER_END_EFFECTOR_POSITION",
    "PAPER_END_EFFECTOR_ROTATION",
    "PAPER_SMOOTH_ROOT_2D",
    "PAPER_PELVIS_TO_SMOOTH_ROOT_2D",
    "FootContactConsistency",
    "FootSkateFromContacts",
    "FootSkateFromHeight",
    "FootSkateRatio",
    "TMR_EmbeddingMetric",
    "TMR_Metric",
    "aggregate_metrics",
    "clear_metrics",
    "compute_metrics",
    "compute_tmr_per_sample_retrieval",
    "compute_tmr_retrieval_metrics",
    "compute_paper_constraint_errors",
    "rotation_geodesic_degrees",
]
