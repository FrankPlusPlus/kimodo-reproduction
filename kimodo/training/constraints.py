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
    sampling_lanes: list[str]
    component_counts: list[int]


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

    # Leaves mirror the public benchmark's capability taxonomy.  Their training
    # probability is not disclosed, so V2 samples these uniformly inside an
    # explicitly configured coverage branch while retaining the original
    # paper-derived branch above.
    BENCHMARK_PATTERNS = (
        "benchmark_full_body_inbetweening",
        "benchmark_full_body_random",
        "benchmark_ee_feet_posrot",
        "benchmark_ee_hands_posrot",
        "benchmark_ee_hands_feet_posrot",
        "benchmark_root_path_2dpos",
        "benchmark_root_path_2dposrot",
        "benchmark_root_waypoint_2dpos",
        "benchmark_root_waypoint_2dposrot",
        "benchmark_mix_root_ee_hands_posrot",
        "benchmark_mix_root_ee_hands_posrot_fullbody",
        "benchmark_mix_root_ee_hands_feet_posrot_fullbody",
        "benchmark_mix_root_path_fullbody",
    )
    ALL_PATTERNS = PATTERNS + BENCHMARK_PATTERNS
    BENCHMARK_TWO_COMPONENT = frozenset(
        {
            "benchmark_mix_root_ee_hands_posrot",
            "benchmark_mix_root_path_fullbody",
        }
    )
    BENCHMARK_THREE_COMPONENT = frozenset(
        {
            "benchmark_mix_root_ee_hands_posrot_fullbody",
            "benchmark_mix_root_ee_hands_feet_posrot_fullbody",
        }
    )

    def __init__(self, motion_rep, config) -> None:
        self.motion_rep = motion_rep
        self.skeleton = motion_rep.skeleton
        self.config = config
        if len(self.ALL_PATTERNS) != len(set(self.ALL_PATTERNS)):
            raise RuntimeError("Constraint pattern registry contains duplicate names")

    def _phase2_progress(self, global_step: int) -> float:
        offset = global_step - self.config.phase1_steps
        denominator = max(1, self.config.phase2_steps - 1)
        return min(1.0, max(0.0, offset / denominator))

    def maximum_sparse_keyframes(self, global_step: int) -> int:
        progress = self._phase2_progress(global_step)
        low = self.config.sparse_keyframes_min
        high = self.config.sparse_keyframes_max
        return round(low + progress * (high - low))

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
        low = max(1, round(length * self.config.dense_path_min_fraction))
        high = max(low, round(length * self.config.dense_path_max_fraction))
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

    def _benchmark_sparse_frames(self, length, maximum, generator) -> torch.Tensor:
        maximum = max(
            1,
            min(length, maximum, int(self.config.benchmark_sparse_keyframes_max)),
        )
        minimum = min(self.config.sparse_keyframes_min, maximum)
        choices = torch.arange(minimum, maximum + 1, dtype=torch.float32)
        weights = choices.pow(-float(self.config.benchmark_sparse_count_power))
        selected = int(torch.multinomial(weights, 1, generator=generator).item())
        count = minimum + selected
        return torch.randperm(length, generator=generator)[:count].sort().values

    def _benchmark_mark_full_body(self, mask, frames) -> None:
        self._mark_feature(mask, frames, self.motion_rep.slice_dict["smooth_root_pos"])
        self._mark_feature(mask, frames, self.motion_rep.slice_dict["global_root_heading"])
        self._mark_feature(mask, frames, self.motion_rep.slice_dict["local_joints_positions"])

    def _benchmark_mark_end_effectors(self, mask, frames, groups: list[str]) -> None:
        rotation_names, position_names = self.skeleton.expand_joint_names(groups)
        rotation_indices = [self.skeleton.bone_index[name] for name in rotation_names]
        position_indices = [self.skeleton.bone_index[name] for name in position_names]
        self._mark_feature(mask, frames, self.motion_rep.slice_dict["smooth_root_pos"])
        self._mark_feature(mask, frames, self.motion_rep.slice_dict["global_root_heading"])
        self._mark_joint_features(mask, frames, "local_joints_positions", position_indices, 3)
        self._mark_joint_features(mask, frames, "global_rot_data", rotation_indices, 6)

    def _benchmark_mark_root(self, mask, frames, *, heading: bool) -> None:
        frames = frames.to(mask.device)
        root_pos = self.motion_rep.slice_dict["smooth_root_pos"]
        mask[frames, root_pos.start] = True
        mask[frames, root_pos.start + 2] = True
        if heading:
            self._mark_feature(mask, frames, self.motion_rep.slice_dict["global_root_heading"])

    def _benchmark_full_body_inbetweening(self, mask, length, maximum, generator) -> None:
        del maximum, generator
        self._benchmark_mark_full_body(mask, torch.tensor([0, length - 1]))

    def _benchmark_full_body_random(self, mask, length, maximum, generator) -> None:
        self._benchmark_mark_full_body(
            mask, self._benchmark_sparse_frames(length, maximum, generator)
        )

    def _benchmark_ee(self, mask, length, maximum, generator, groups) -> None:
        self._benchmark_mark_end_effectors(
            mask,
            self._benchmark_sparse_frames(length, maximum, generator),
            groups,
        )

    def _benchmark_ee_feet_posrot(self, mask, length, maximum, generator) -> None:
        self._benchmark_ee(mask, length, maximum, generator, ["LeftFoot", "RightFoot"])

    def _benchmark_ee_hands_posrot(self, mask, length, maximum, generator) -> None:
        self._benchmark_ee(mask, length, maximum, generator, ["LeftHand", "RightHand"])

    def _benchmark_ee_hands_feet_posrot(self, mask, length, maximum, generator) -> None:
        self._benchmark_ee(
            mask,
            length,
            maximum,
            generator,
            ["LeftHand", "RightHand", "LeftFoot", "RightFoot"],
        )

    def _benchmark_root_path(self, mask, length, *, heading: bool) -> None:
        self._benchmark_mark_root(mask, torch.arange(length), heading=heading)

    def _benchmark_root_waypoint(self, mask, length, maximum, generator, *, heading: bool) -> None:
        self._benchmark_mark_root(
            mask,
            self._benchmark_sparse_frames(length, maximum, generator),
            heading=heading,
        )

    def _benchmark_root_path_2dpos(self, mask, length, maximum, generator) -> None:
        del maximum, generator
        self._benchmark_root_path(mask, length, heading=False)

    def _benchmark_root_path_2dposrot(self, mask, length, maximum, generator) -> None:
        del maximum, generator
        self._benchmark_root_path(mask, length, heading=True)

    def _benchmark_root_waypoint_2dpos(self, mask, length, maximum, generator) -> None:
        self._benchmark_root_waypoint(mask, length, maximum, generator, heading=False)

    def _benchmark_root_waypoint_2dposrot(self, mask, length, maximum, generator) -> None:
        self._benchmark_root_waypoint(mask, length, maximum, generator, heading=True)

    def _benchmark_mix_root_ee_hands_posrot(self, mask, length, maximum, generator) -> None:
        root_frames = self._benchmark_sparse_frames(length, maximum, generator)
        ee_frames = self._benchmark_sparse_frames(length, maximum, generator)
        self._benchmark_mark_root(mask, root_frames, heading=False)
        self._benchmark_mark_end_effectors(mask, ee_frames, ["LeftHand", "RightHand"])

    def _benchmark_mix_root_ee_hands_posrot_fullbody(
        self, mask, length, maximum, generator
    ) -> None:
        self._benchmark_mix_root_ee_hands_posrot(mask, length, maximum, generator)
        self._benchmark_mark_full_body(
            mask, self._benchmark_sparse_frames(length, maximum, generator)
        )

    def _benchmark_mix_root_ee_hands_feet_posrot_fullbody(
        self, mask, length, maximum, generator
    ) -> None:
        # The benchmark's mixture leaf named hands_feet uses RightHand+LeftFoot,
        # unlike the standalone four-end-effector leaf.
        self._benchmark_root_path(mask, length, heading=False)
        ee_frames = self._benchmark_sparse_frames(length, maximum, generator)
        self._benchmark_mark_end_effectors(mask, ee_frames, ["RightHand", "LeftFoot"])
        self._benchmark_mark_full_body(
            mask, self._benchmark_sparse_frames(length, maximum, generator)
        )

    def _benchmark_mix_root_path_fullbody(self, mask, length, maximum, generator) -> None:
        self._benchmark_root_path(mask, length, heading=False)
        self._benchmark_mark_full_body(
            mask, self._benchmark_sparse_frames(length, maximum, generator)
        )

    def _apply_pattern(self, name, mask, length, maximum, generator) -> None:
        getattr(self, f"_{name}")(mask, length, maximum, generator)

    def _select_patterns(self, generator: torch.Generator) -> tuple[list[str], str, int]:
        """Select one stable top-level lane while preserving paper mixture mass."""
        choice = self._rand(generator)
        no_constraint = float(self.config.no_constraint_probability)
        paper_two = float(self.config.mix_two_probability)
        benchmark_mass = (
            (1.0 - no_constraint) * float(self.config.benchmark_coverage_probability)
        )
        if choice < no_constraint:
            return [], "none", 0
        if choice < no_constraint + paper_two:
            order = torch.randperm(len(self.PATTERNS), generator=generator)[:2].tolist()
            return [self.PATTERNS[index] for index in order], "paper_two", 2
        if choice < no_constraint + paper_two + benchmark_mass:
            name = self.BENCHMARK_PATTERNS[
                self._randint(len(self.BENCHMARK_PATTERNS), generator)
            ]
            components = (
                3
                if name in self.BENCHMARK_THREE_COMPONENT
                else 2
                if name in self.BENCHMARK_TWO_COMPONENT
                else 1
            )
            return [name], "benchmark", components
        order = torch.randperm(len(self.PATTERNS), generator=generator)[:1].tolist()
        return [self.PATTERNS[order[0]]], "paper_single", 1

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
        lengths_list = lengths.detach().cpu().tolist()
        if any(not 1 <= int(length) <= max_time for length in lengths_list):
            raise ValueError(f"Invalid sequence lengths for T={max_time}: {lengths_list}")

        in_phase2 = global_step >= self.config.phase1_steps
        if not in_phase2:
            return ConstraintBatch(
                torch.zeros_like(clean_motion),
                mask,
                [[] for _ in range(batch_size)],
                maximum,
                ["phase1_none" for _ in range(batch_size)],
                [0 for _ in range(batch_size)],
            )

        lanes: list[str] = []
        component_counts: list[int] = []
        for batch_index, length_value in enumerate(lengths_list):
            length = int(length_value)
            selected, lane, component_count = self._select_patterns(generator)
            if selected:
                for name in selected:
                    self._apply_pattern(name, mask[batch_index], length, maximum, generator)
            names.append(selected)
            lanes.append(lane)
            component_counts.append(component_count)

        observed = torch.where(mask, clean_motion, torch.zeros_like(clean_motion))
        return ConstraintBatch(observed, mask, names, maximum, lanes, component_counts)
