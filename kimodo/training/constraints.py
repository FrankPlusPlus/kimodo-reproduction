# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-2 constraint curriculum reconstructed from the Kimodo report."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ConstraintBatch:
    observed_motion: torch.Tensor
    motion_mask: torch.Tensor
    pattern_names: list[list[str]]
    maximum_sparse_keyframes: int


class ConstraintCurriculumSampler:
    """Sample paper-described constraint patterns directly in motion-feature space.

    The report discloses the pattern families and top-level mixing rates, but not
    the probability of each family nor the precise distribution that biases
    sparse counts low.  This implementation uses uniform family selection and a
    configurable power-law count sampler; both choices are explicitly marked as
    inferred in the reproduction documentation.
    """

    PATTERNS = (
        "full_body_sparse",
        "end_effector_sparse",
        "root_sparse",
        "root_dense",
        "foot_contact_sparse",
    )

    def __init__(self, motion_rep, config) -> None:
        self.motion_rep = motion_rep
        self.skeleton = motion_rep.skeleton
        self.config = config

    def _phase2_progress(self, global_step: int) -> float:
        offset = global_step - self.config.phase1_steps
        denominator = max(1, self.config.phase2_steps - 1)
        return min(1.0, max(0.0, offset / denominator))

    def maximum_sparse_keyframes(self, global_step: int) -> int:
        progress = self._phase2_progress(global_step)
        low = self.config.sparse_keyframes_min
        high = self.config.sparse_keyframes_max
        return int(round(low + progress * (high - low)))

    @staticmethod
    def _rand(generator: torch.Generator) -> float:
        return float(torch.rand((), generator=generator).item())

    @staticmethod
    def _randint(high: int, generator: torch.Generator) -> int:
        if high <= 0:
            raise ValueError("randint high must be positive")
        return int(torch.randint(high, (), generator=generator).item())

    def _keyframes(self, length: int, maximum: int, generator: torch.Generator) -> torch.Tensor:
        maximum = max(1, min(length, maximum))
        minimum = min(self.config.sparse_keyframes_min, maximum)
        choices = torch.arange(minimum, maximum + 1, dtype=torch.float32)
        weights = choices.pow(-float(self.config.sparse_count_power))
        selected = int(torch.multinomial(weights, 1, generator=generator).item())
        count = minimum + selected
        return torch.randperm(length, generator=generator)[:count].sort().values

    @staticmethod
    def _mark_feature(mask: torch.Tensor, frames: torch.Tensor, feature_slice: slice) -> None:
        frames = frames.to(mask.device)
        mask[frames, feature_slice] = True

    def _mark_joint_features(
        self,
        mask: torch.Tensor,
        frames: torch.Tensor,
        feature_name: str,
        joint_indices: list[int],
        width: int,
    ) -> None:
        frames = frames.to(mask.device)
        start = self.motion_rep.slice_dict[feature_name].start
        for joint_index in joint_indices:
            mask[frames, start + joint_index * width : start + (joint_index + 1) * width] = True

    def _full_body_sparse(self, mask, length, maximum, generator) -> None:
        frames = self._keyframes(length, maximum, generator)
        self._mark_feature(mask, frames, self.motion_rep.slice_dict["smooth_root_pos"])
        self._mark_feature(mask, frames, self.motion_rep.slice_dict["global_root_heading"])
        self._mark_feature(mask, frames, self.motion_rep.slice_dict["local_joints_positions"])

    def _end_effector_sparse(self, mask, length, maximum, generator) -> None:
        frames = self._keyframes(length, maximum, generator)
        semantic_groups = ["LeftFoot", "RightFoot", "LeftHand", "RightHand"]
        selected_count = 1 + self._randint(len(semantic_groups), generator)
        order = torch.randperm(len(semantic_groups), generator=generator)[:selected_count].tolist()
        selected = [semantic_groups[index] for index in order]
        rotation_names, position_names = self.skeleton.expand_joint_names(selected)
        rotation_indices = [self.skeleton.bone_index[name] for name in rotation_names]
        position_indices = [self.skeleton.bone_index[name] for name in position_names]
        # Global joint positions are represented relative to the smooth root;
        # matching the released constraint builder also supplies root and heading.
        self._mark_feature(mask, frames, self.motion_rep.slice_dict["smooth_root_pos"])
        self._mark_feature(mask, frames, self.motion_rep.slice_dict["global_root_heading"])
        self._mark_joint_features(mask, frames, "local_joints_positions", position_indices, 3)
        self._mark_joint_features(mask, frames, "global_rot_data", rotation_indices, 6)

    def _root_sparse(self, mask, length, maximum, generator) -> None:
        frames = self._keyframes(length, maximum, generator).to(mask.device)
        root_pos = self.motion_rep.slice_dict["smooth_root_pos"]
        mask[frames, root_pos.start] = True
        mask[frames, root_pos.start + 2] = True
        if self._rand(generator) < self.config.root_heading_probability:
            self._mark_feature(mask, frames, self.motion_rep.slice_dict["global_root_heading"])

    def _root_dense(self, mask, length, maximum, generator) -> None:
        del maximum
        low = max(1, int(round(length * self.config.dense_path_min_fraction)))
        high = max(low, int(round(length * self.config.dense_path_max_fraction)))
        span = low + self._randint(high - low + 1, generator)
        start = self._randint(length - span + 1, generator)
        frames = torch.arange(start, start + span, device=mask.device)
        root_pos = self.motion_rep.slice_dict["smooth_root_pos"]
        mask[frames, root_pos.start] = True
        mask[frames, root_pos.start + 2] = True
        if self._rand(generator) < self.config.root_heading_probability:
            self._mark_feature(mask, frames, self.motion_rep.slice_dict["global_root_heading"])

    def _foot_contact_sparse(self, mask, length, maximum, generator) -> None:
        frames = self._keyframes(length, maximum, generator)
        self._mark_feature(mask, frames, self.motion_rep.slice_dict["foot_contacts"])

    def _apply_pattern(self, name, mask, length, maximum, generator) -> None:
        getattr(self, f"_{name}")(mask, length, maximum, generator)

    def sample(
        self,
        clean_motion: torch.Tensor,
        lengths: torch.Tensor,
        global_step: int,
        generator: torch.Generator,
    ) -> ConstraintBatch:
        if clean_motion.ndim != 3:
            raise ValueError("clean_motion must have shape [B,T,D]")
        batch_size, max_time, _ = clean_motion.shape
        mask = torch.zeros_like(clean_motion, dtype=torch.bool)
        names: list[list[str]] = []
        maximum = self.maximum_sparse_keyframes(global_step)

        in_phase2 = global_step >= self.config.phase1_steps
        for batch_index in range(batch_size):
            length = int(lengths[batch_index].item())
            if not 1 <= length <= max_time:
                raise ValueError(f"Invalid sequence length {length} for T={max_time}")
            selected: list[str] = []
            choice = self._rand(generator)
            if in_phase2 and choice >= self.config.no_constraint_probability:
                count = 2 if choice < (
                    self.config.no_constraint_probability + self.config.mix_two_probability
                ) else 1
                order = torch.randperm(len(self.PATTERNS), generator=generator)[:count].tolist()
                selected = [self.PATTERNS[index] for index in order]
                for name in selected:
                    self._apply_pattern(name, mask[batch_index], length, maximum, generator)
            names.append(selected)

        observed = torch.where(mask, clean_motion, torch.zeros_like(clean_motion))
        return ConstraintBatch(observed, mask, names, maximum)
