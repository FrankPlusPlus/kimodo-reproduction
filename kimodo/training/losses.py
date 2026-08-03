# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seven-term clean-motion prediction loss from Kimodo Eq. (1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kimodo.geometry import cont6d_to_matrix
from kimodo.skeleton.transforms import global_rots_to_local_rots


def _masked_smooth_l1_frame_sum(
    prediction: torch.Tensor,
    target: torch.Tensor,
    frame_mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    raw = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    expanded = frame_mask
    while expanded.ndim < raw.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(raw)
    components_per_frame = raw[0, 0].numel()
    return (raw * expanded).sum() / components_per_frame


@dataclass
class LossOutput:
    means: dict[str, torch.Tensor]
    frame_sums: dict[str, torch.Tensor]
    valid_frame_count: torch.Tensor

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.means[key]

    def items(self):
        return self.means.items()

    def values(self):
        return self.means.values()


class KimodoLoss:
    """Compute the paper's six representation losses plus differentiable FK."""

    FEATURE_TERMS = (
        ("root_position", "smooth_root_pos"),
        ("root_heading", "global_root_heading"),
        ("joint_position", "local_joints_positions"),
        ("joint_rotation", "global_rot_data"),
        ("joint_velocity", "velocities"),
        ("foot_contact", "foot_contacts"),
    )

    def __init__(self, motion_rep, config) -> None:
        self.motion_rep = motion_rep
        self.skeleton = motion_rep.skeleton
        self.config = config

    def _target_positions(self, target_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode only the target tensors consumed by the FK loss.

        ``KimodoMotionRep.inverse(..., posed_joints_from="positions")`` also
        converts target 6D rotations to matrices and then to local rotations,
        but those rotations are discarded by this loss.  Reconstructing the
        root and posed joints directly is exactly the position branch of that
        inverse method.
        """
        smooth_root = target_raw[..., self.motion_rep.slice_dict["smooth_root_pos"]]
        local_positions = target_raw[
            ..., self.motion_rep.slice_dict["local_joints_positions"]
        ].reshape(*target_raw.shape[:2], self.skeleton.nbjoints, 3)
        posed_joints = local_positions.clone()
        posed_joints[..., 0] += smooth_root[..., None, 0]
        posed_joints[..., 2] += smooth_root[..., None, 2]
        root_positions = posed_joints[..., self.skeleton.root_idx, :]
        return root_positions, posed_joints

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> LossOutput:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have identical shapes")
        if valid_frames.shape != prediction.shape[:2] or valid_frames.dtype != torch.bool:
            raise ValueError("valid_frames must be bool [B,T]")

        if self.config.direct_feature_domain == "physical":
            direct_prediction = self.motion_rep.unnormalize(prediction)
            direct_target = self.motion_rep.unnormalize(target)
        else:
            direct_prediction = prediction
            direct_target = target

        valid_frame_count = valid_frames.sum().to(dtype=prediction.dtype).clamp_min(1)
        frame_sums: dict[str, torch.Tensor] = {}
        for term_name, feature_name in self.FEATURE_TERMS:
            feature_slice = self.motion_rep.slice_dict[feature_name]
            frame_sums[term_name] = _masked_smooth_l1_frame_sum(
                direct_prediction[..., feature_slice],
                direct_target[..., feature_slice],
                valid_frames,
                self.config.smooth_l1_beta,
            )

        # FK is always evaluated after unnormalization because it operates in
        # meters and valid rotation coordinates.
        prediction_raw = (
            direct_prediction
            if self.config.direct_feature_domain == "physical"
            else self.motion_rep.unnormalize(prediction)
        )
        target_raw = (
            direct_target
            if self.config.direct_feature_domain == "physical"
            else self.motion_rep.unnormalize(target)
        )
        rotation_data = prediction_raw[..., self.motion_rep.slice_dict["global_rot_data"]]
        rotation_data = rotation_data.reshape(*rotation_data.shape[:2], self.skeleton.nbjoints, 6)
        global_rotations = cont6d_to_matrix(rotation_data)
        local_rotations = global_rots_to_local_rots(global_rotations, self.skeleton)

        target_root_positions, target_posed_joints = self._target_positions(target_raw)
        _, predicted_fk_positions, _ = self.skeleton.fk(
            local_rotations,
            target_root_positions,
        )
        frame_sums["forward_kinematics"] = _masked_smooth_l1_frame_sum(
            predicted_fk_positions,
            target_posed_joints,
            valid_frames,
            self.config.smooth_l1_beta,
        )

        total_sum = prediction.new_zeros(())
        for term_name, value in frame_sums.items():
            total_sum = total_sum + float(getattr(self.config, term_name)) * value
        frame_sums["total"] = total_sum
        means = {name: value / valid_frame_count for name, value in frame_sums.items()}
        return LossOutput(means, frame_sums, valid_frame_count)
