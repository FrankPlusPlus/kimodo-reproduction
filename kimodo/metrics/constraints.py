# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constraint-following metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch import Tensor

from kimodo.constraints import (
    EndEffectorConstraintSet,
    FullBodyConstraintSet,
    Root2DConstraintSet,
)
from kimodo.tools import ensure_batched

from .base import Metric


PAPER_FULLBODY_POSITION = "paper_constraint_fullbody_position_m"
PAPER_END_EFFECTOR_POSITION = "paper_constraint_end_effector_position_m"
PAPER_END_EFFECTOR_ROTATION = "paper_constraint_end_effector_rotation_deg"
PAPER_SMOOTH_ROOT_2D = "paper_constraint_smooth_root_2d_m"
PAPER_PELVIS_TO_SMOOTH_ROOT_2D = "paper_constraint_pelvis_to_smooth_root_2d_m"


def rotation_geodesic_degrees(pred: Tensor, target: Tensor) -> Tensor:
    """Return the SO(3) geodesic angle between rotation matrices, in degrees.

    Kimodo Sec. 6.1 reports end-effector rotation error in degrees.  This is
    the coordinate-invariant angle of ``pred @ target.T`` rather than an
    element-wise matrix or 6D-representation distance.
    """
    if pred.shape != target.shape or pred.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected matching rotation matrices [..., 3, 3], "
            f"got {tuple(pred.shape)} and {tuple(target.shape)}."
        )
    relative = pred @ target.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def compute_paper_constraint_errors(
    *,
    posed_joints: Tensor,
    constraints_lst: List,
    root_idx: int,
    global_rot_mats: Optional[Tensor] = None,
    smooth_root_pos: Optional[Tensor] = None,
    length: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Compute the raw constraint-point errors explicitly described in Sec. 6.1.

    Values are intentionally *not* reduced per motion.  Keeping every
    constrained frame/joint lets the suite evaluator compute one mean (and
    one pelvis p95) over the complete constraint test suite.  The legacy
    :class:`ContraintFollow` metric remains unchanged for public-benchmark
    compatibility.
    """
    if posed_joints.ndim != 3:
        raise ValueError(f"Expected posed_joints [T, J, 3], got {tuple(posed_joints.shape)}")
    valid_length = posed_joints.shape[0] if length is None else int(length)
    output: defaultdict[str, list[Tensor]] = defaultdict(list)

    for constraint in constraints_lst:
        frame_idx = constraint.frame_indices.to(device=posed_joints.device, dtype=torch.long)
        if frame_idx.numel() == 0:
            continue
        if int(frame_idx.min()) < 0 or int(frame_idx.max()) >= valid_length:
            raise ValueError("Constraint frame index lies outside the valid motion length.")

        if isinstance(constraint, Root2DConstraintSet):
            if smooth_root_pos is None:
                raise ValueError(
                    "Sec. 6.1 smooth-root error requires generated 'smooth_root_pos'; "
                    "pelvis positions are not a valid substitute."
                )
            smooth_root_pos = smooth_root_pos.to(posed_joints.device)
            if smooth_root_pos.ndim != 2 or smooth_root_pos.shape[-1] not in (2, 3):
                raise ValueError(
                    "Expected smooth_root_pos [T, 2] or [T, 3], "
                    f"got {tuple(smooth_root_pos.shape)}."
                )
            pred_smooth_2d = (
                smooth_root_pos[frame_idx]
                if smooth_root_pos.shape[-1] == 2
                else smooth_root_pos[frame_idx][:, [0, 2]]
            )
            target = constraint.smooth_root_2d.to(posed_joints.device)
            pelvis_2d = posed_joints[frame_idx, root_idx][:, [0, 2]]
            output[PAPER_SMOOTH_ROOT_2D].append(torch.linalg.vector_norm(pred_smooth_2d - target, dim=-1))
            output[PAPER_PELVIS_TO_SMOOTH_ROOT_2D].append(torch.linalg.vector_norm(pelvis_2d - target, dim=-1))

        elif isinstance(constraint, FullBodyConstraintSet):
            pred = posed_joints[frame_idx]
            target = constraint.global_joints_positions.to(posed_joints.device)
            output[PAPER_FULLBODY_POSITION].append(torch.linalg.vector_norm(pred - target, dim=-1).reshape(-1))

        elif isinstance(constraint, EndEffectorConstraintSet):
            pos_idx = constraint.pos_indices.to(device=posed_joints.device, dtype=torch.long)
            rot_idx = constraint.rot_indices.to(device=posed_joints.device, dtype=torch.long)
            if pos_idx.numel():
                pred_pos = posed_joints[frame_idx].index_select(1, pos_idx)
                target_pos = constraint.global_joints_positions.to(posed_joints.device).index_select(1, pos_idx)
                output[PAPER_END_EFFECTOR_POSITION].append(
                    torch.linalg.vector_norm(pred_pos - target_pos, dim=-1).reshape(-1)
                )
            if rot_idx.numel():
                if global_rot_mats is None:
                    raise ValueError(
                        "Sec. 6.1 end-effector rotation error requires generated 'global_rot_mats'."
                    )
                pred_rot = global_rot_mats.to(posed_joints.device)[frame_idx].index_select(1, rot_idx)
                target_rot = constraint.global_joints_rots.to(posed_joints.device).index_select(1, rot_idx)
                output[PAPER_END_EFFECTOR_ROTATION].append(
                    rotation_geodesic_degrees(pred_rot, target_rot).reshape(-1)
                )

    return {key: torch.cat(values) for key, values in output.items() if values}


class PaperConstraintFollow(Metric):
    """Raw Sec. 6.1 constraint errors for exact suite-level aggregation."""

    def __init__(self, skeleton, **kwargs):
        super().__init__(**kwargs)
        self.skeleton = skeleton

    def _compute(
        self,
        posed_joints: Tensor,
        constraints_lst: Optional[List],
        lengths: Optional[Tensor] = None,
        global_rot_mats: Optional[Tensor] = None,
        smooth_root_pos: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        if not constraints_lst:
            return {}

        if posed_joints.ndim == 3:
            posed_joints = posed_joints.unsqueeze(0)
            constraints_lst = [constraints_lst]
            if global_rot_mats is not None:
                global_rot_mats = global_rot_mats.unsqueeze(0)
            if smooth_root_pos is not None:
                smooth_root_pos = smooth_root_pos.unsqueeze(0)
            if lengths is not None and lengths.ndim == 0:
                lengths = lengths.unsqueeze(0)
        elif posed_joints.ndim != 4:
            raise ValueError(f"Expected posed_joints [T,J,3] or [B,T,J,3], got {tuple(posed_joints.shape)}")
        if len(constraints_lst) != posed_joints.shape[0]:
            raise ValueError("constraints_lst must contain one list per motion in the batch.")

        batch_output: defaultdict[str, list[Tensor]] = defaultdict(list)
        for batch_idx, (posed, constraints) in enumerate(zip(posed_joints, constraints_lst)):
            if not constraints:
                continue
            errors = compute_paper_constraint_errors(
                posed_joints=posed,
                constraints_lst=constraints,
                root_idx=self.skeleton.root_idx,
                global_rot_mats=None if global_rot_mats is None else global_rot_mats[batch_idx],
                smooth_root_pos=None if smooth_root_pos is None else smooth_root_pos[batch_idx],
                length=None if lengths is None else int(lengths[batch_idx]),
            )
            for key, values in errors.items():
                batch_output[key].append(values)
        return {key: torch.cat(values) for key, values in batch_output.items() if values}


class ContraintFollow(Metric):
    """Constraint-following metric dispatcher for kimodo constraint sets."""

    def __init__(
        self,
        skeleton,
        root_threshold: float = 0.10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.skeleton = skeleton
        self.root_threshold = root_threshold

    @ensure_batched(posed_joints=4, constraints_lst=2, lengths=1)
    def _compute(
        self,
        posed_joints: Tensor,
        constraints_lst: Optional[List],
        lengths: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict:
        if not constraints_lst:
            return {}

        root_idx = self.skeleton.root_idx
        output = defaultdict(list)

        for posed_joints_s, constraint_lst_s, lengths_s in zip(posed_joints, constraints_lst, lengths):
            output_seq = defaultdict(list)
            for constraint in constraint_lst_s:
                frame_idx = constraint.frame_indices.to(device=posed_joints_s.device, dtype=torch.long)
                assert frame_idx.max() < lengths_s, "The constraint is defined outsite the lenght of the motion."
                if frame_idx.numel() == 0:
                    continue

                if isinstance(constraint, Root2DConstraintSet):
                    pred_root2d = posed_joints_s[frame_idx, root_idx][:, [0, 2]]
                    target = constraint.smooth_root_2d.to(posed_joints_s.device)

                    dist = torch.norm(pred_root2d - target, dim=-1)
                    output_seq["constraint_root2d_err"].append(dist)
                    hit = (dist <= self.root_threshold).float()
                    output_seq["constraint_root2d_acc"].append(hit)

                elif isinstance(constraint, FullBodyConstraintSet):
                    pred = posed_joints_s[frame_idx]
                    target = constraint.global_joints_positions.to(posed_joints_s.device)
                    err = torch.norm(pred - target, dim=-1)
                    output_seq["constraint_fullbody_keyframe"].append(err)

                elif isinstance(constraint, EndEffectorConstraintSet):
                    pos_idx = constraint.pos_indices.to(device=posed_joints_s.device, dtype=torch.long)
                    pred = posed_joints_s[frame_idx].index_select(1, pos_idx)
                    target = constraint.global_joints_positions.to(posed_joints_s.device).index_select(1, pos_idx)
                    err = torch.norm(pred - target, dim=-1)
                    output_seq["constraint_end_effector"].append(err)

            # in case we have several same constraints in the list
            for key, val in output_seq.items():
                output[key].append(torch.cat(val).mean())

        reduced = {}
        for key, vals in output.items():
            reduced[key] = torch.stack(vals, dim=0)
        return reduced
