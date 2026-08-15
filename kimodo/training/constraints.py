# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-2 constraint curriculum reconstructed from the Kimodo report."""

from __future__ import annotations

import math
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
    scheduled_sparse_keyframes: float = 0.0
    sampled_sparse_keyframe_cap_mean: float = 0.0
    sampled_sparse_keyframe_count_mean: float = 0.0
    sparse_constraint_load_mean: float = 0.0
    mask_channel_load_mean: float = 0.0
    mask_channel_load_max: float = 0.0


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
    BENCHMARK_EE_GROUPS = {
        "benchmark_ee_feet_posrot": ("LeftFoot", "RightFoot"),
        "benchmark_ee_hands_posrot": ("LeftHand", "RightHand"),
        "benchmark_ee_hands_feet_posrot": ("LeftHand", "RightHand", "LeftFoot", "RightFoot"),
        "benchmark_mix_root_ee_hands_posrot": ("LeftHand", "RightHand"),
        "benchmark_mix_root_ee_hands_posrot_fullbody": ("LeftHand", "RightHand"),
        "benchmark_mix_root_ee_hands_feet_posrot_fullbody": ("RightHand", "LeftFoot"),
    }
    # Sparse keyframe draws that consume the sample-level K budget. Dense path,
    # full 2D root path, and inbetweening are excluded: they ignore Kmax and
    # were already present in the stable K≤7 window.
    SPARSE_DRAW_KINDS = {
        "full_body_sparse": ("full_body",),
        "end_effector_sparse": ("end_effector",),
        "root_sparse": ("root",),
        "root_dense": (),
        "foot_contact_sparse": ("foot",),
        "benchmark_full_body_inbetweening": (),
        "benchmark_full_body_random": ("full_body",),
        "benchmark_ee_feet_posrot": ("end_effector",),
        "benchmark_ee_hands_posrot": ("end_effector",),
        "benchmark_ee_hands_feet_posrot": ("end_effector",),
        "benchmark_root_path_2dpos": (),
        "benchmark_root_path_2dposrot": (),
        "benchmark_root_waypoint_2dpos": ("root",),
        "benchmark_root_waypoint_2dposrot": ("root",),
        "benchmark_mix_root_ee_hands_posrot": ("root", "end_effector"),
        "benchmark_mix_root_ee_hands_posrot_fullbody": ("root", "end_effector", "full_body"),
        "benchmark_mix_root_ee_hands_feet_posrot_fullbody": ("end_effector", "full_body"),
        "benchmark_mix_root_path_fullbody": ("full_body",),
    }

    def __init__(self, motion_rep, config) -> None:
        self.motion_rep = motion_rep
        self.skeleton = motion_rep.skeleton
        self.config = config
        if len(self.ALL_PATTERNS) != len(set(self.ALL_PATTERNS)):
            raise RuntimeError("Constraint pattern registry contains duplicate names")
        if set(self.SPARSE_DRAW_KINDS) != set(self.ALL_PATTERNS):
            raise RuntimeError("SPARSE_DRAW_KINDS must cover every registered pattern")
        # Power-law multinomial weights depend only on (min, max, power); cache them
        # so Phase-2 hot path does not rebuild arange/pow every sparse draw.
        self._sparse_weight_cache: dict[tuple[int, int, float], torch.Tensor] = {}
        self._benchmark_weight_cache: dict[tuple[int, int, float], torch.Tensor] = {}
        self._constant_load_cache: dict[tuple, torch.Tensor] = {}
        self._count_budget: list[int] | None = None
        self._frame_budget: list[torch.Tensor] | None = None
        self._pending_ee_groups: list[str] | None = None
        self._count_budget_index = 0
        root = self.motion_rep.slice_dict["smooth_root_pos"]
        heading = self.motion_rep.slice_dict["global_root_heading"]
        joints = self.motion_rep.slice_dict["local_joints_positions"]
        feet = self.motion_rep.slice_dict["foot_contacts"]
        self._channel_cost = {
            "full_body": (root.stop - root.start) + (heading.stop - heading.start) + (joints.stop - joints.start),
            "end_effector": 40,
            # Worst-case root sparse cost (XZ + heading). Heading is sampled
            # after the budget draw; billing 4 keeps the 800 cap closed.
            "root": 4,
            "foot": feet.stop - feet.start,
        }

    def _phase2_progress(self, global_step: int) -> float:
        offset = global_step - self.config.phase1_steps
        denominator = max(1, self.config.phase2_steps - 1)
        return min(1.0, max(0.0, offset / denominator))

    def scheduled_sparse_keyframes(self, global_step: int) -> float:
        progress = self._phase2_progress(global_step)
        low = float(self.config.sparse_keyframes_min)
        high = float(self.config.sparse_keyframes_max)
        return low + progress * (high - low)

    def _scheduled_for_sampling(self, global_step: int) -> float:
        scheduled = self.scheduled_sparse_keyframes(global_step)
        hard = int(self.config.sparse_keyframes_hard_cap)
        if hard > 0:
            return min(scheduled, float(hard))
        return scheduled

    def maximum_sparse_keyframes(self, global_step: int) -> int:
        scheduled = self._scheduled_for_sampling(global_step)
        low = int(self.config.sparse_keyframes_min)
        high = int(self.config.sparse_keyframes_max)
        hard = int(self.config.sparse_keyframes_hard_cap)
        if hard > 0:
            high = min(high, hard)
        return min(high, max(low, int(round(scheduled))))

    def sample_sparse_keyframe_cap(self, global_step: int, generator: torch.Generator) -> int:
        mode = str(self.config.sparse_keyframe_cap_mode)
        if mode == "round":
            return self.maximum_sparse_keyframes(global_step)
        if mode != "adjacent_mix":
            raise ValueError("sparse_keyframe_cap_mode must be 'round' or 'adjacent_mix'")
        scheduled = self._scheduled_for_sampling(global_step)
        low = max(int(self.config.sparse_keyframes_min), int(math.floor(scheduled)))
        high = min(int(self.config.sparse_keyframes_max), int(math.ceil(scheduled)))
        hard = int(self.config.sparse_keyframes_hard_cap)
        if hard > 0:
            high = min(high, hard)
            low = min(low, high)
        if high <= low:
            return low
        fraction = scheduled - float(low)
        return high if self._rand(generator) < fraction else low

    @staticmethod
    def _rand(generator: torch.Generator) -> float:
        return float(torch.rand((), generator=generator).item())

    @staticmethod
    def _randint(high: int, generator: torch.Generator) -> int:
        if high <= 0:
            raise ValueError("randint high must be positive")
        return int(torch.randint(high, (), generator=generator).item())

    @staticmethod
    def _power_law_weights(
        minimum: int,
        maximum: int,
        power: float,
        cache: dict[tuple[int, int, float], torch.Tensor],
    ) -> torch.Tensor:
        key = (int(minimum), int(maximum), float(power))
        weights = cache.get(key)
        if weights is None:
            choices = torch.arange(key[0], key[1] + 1, dtype=torch.float32)
            weights = choices.pow(-key[2])
            cache[key] = weights
        return weights

    def _constant_load_weights(
        self,
        minimum: int,
        maximum: int,
        power: float,
    ) -> torch.Tensor:
        baseline = int(self.config.sparse_load_baseline)
        tail_power = float(self.config.sparse_tail_power)
        key = (int(minimum), int(maximum), float(power), baseline, tail_power)
        weights = self._constant_load_cache.get(key)
        if weights is not None:
            return weights
        if maximum <= baseline or minimum > baseline:
            raw = self._power_law_weights(
                minimum, maximum, power, self._sparse_weight_cache
            )
            weights = raw / raw.sum()
            self._constant_load_cache[key] = weights
            return weights
        base_choices = torch.arange(minimum, baseline + 1, dtype=torch.float32)
        base_weights = base_choices.pow(-float(power))
        base_z = base_weights.sum()
        prefix = baseline - minimum
        combined = torch.empty(maximum - minimum + 1, dtype=torch.float32)
        combined[:prefix] = base_weights[:prefix] / base_z
        tail_mass = base_weights[prefix] / base_z
        tail_choices = torch.arange(baseline, maximum + 1, dtype=torch.float32)
        tail_weights = tail_choices.pow(-tail_power)
        combined[prefix:] = tail_mass * tail_weights / tail_weights.sum()
        self._constant_load_cache[key] = combined
        return combined

    def _sample_sparse_count(
        self,
        maximum: int,
        generator: torch.Generator,
        power: float,
        cache: dict[tuple[int, int, float], torch.Tensor],
        *,
        constant_load: bool,
    ) -> int:
        maximum = max(1, maximum)
        minimum = min(int(self.config.sparse_keyframes_min), maximum)
        if constant_load:
            weights = self._constant_load_weights(minimum, maximum, power)
        else:
            weights = self._power_law_weights(minimum, maximum, power, cache)
        selected = int(torch.multinomial(weights, 1, generator=generator).item())
        return minimum + selected

    def _split_positive(self, total: int, parts: int, generator: torch.Generator) -> list[int]:
        if parts <= 0:
            return []
        total = max(total, parts)
        if parts == 1:
            return [total]
        if total == parts:
            return [1] * parts
        cuts = torch.randperm(total - 1, generator=generator)[: parts - 1].sort().values
        edges = [0, *[int(value) + 1 for value in cuts.tolist()], total]
        return [edges[index + 1] - edges[index] for index in range(parts)]

    def _shrink_to_channel_budget(
        self,
        counts: list[int],
        kinds: list[str],
        total: int,
        kind_costs: list[int] | None = None,
    ) -> list[int]:
        if not counts:
            return counts
        counts = list(counts)
        budget = self._channel_budget_units(total)
        widths = (
            [int(cost) for cost in kind_costs]
            if kind_costs is not None
            else [int(self._channel_cost[kind]) for kind in kinds]
        )

        def cost() -> int:
            return sum(count * width for count, width in zip(counts, widths, strict=True))

        while cost() > budget and any(count > 1 for count in counts):
            index = max(
                (item for item in range(len(counts)) if counts[item] > 1),
                key=lambda item: widths[item],
            )
            counts[index] -= 1
        return counts

    def _channel_budget_units(self, sampled_total: int) -> int:
        budget = int(self.config.sparse_channel_budget)
        if budget > 0:
            return budget
        return int(sampled_total) * int(self._channel_cost["full_body"])

    def _prepare_count_budget(
        self,
        selected: list[str],
        cap: int,
        length: int,
        generator: torch.Generator,
        lane: str,
    ) -> tuple[int, int]:
        kinds = [kind for name in selected for kind in self.SPARSE_DRAW_KINDS[name]]
        self._count_budget = None
        self._frame_budget = None
        self._count_budget_index = 0
        if not kinds:
            return 0, 0
        if int(self.config.sparse_channel_budget) > 0:
            return self._prepare_affordable_budget(
                selected, kinds, cap, length, generator, lane
            )
        if cap <= int(self.config.sparse_load_baseline):
            return 0, 0
        power, cache, maximum = self._sparse_draw_params(cap, length, lane)
        total = self._sample_sparse_count(
            maximum, generator, power, cache, constant_load=True
        )
        total = max(total, len(kinds))
        counts = self._split_positive(total, len(kinds), generator)
        counts = self._shrink_to_channel_budget(counts, kinds, total)
        self._count_budget = counts
        load = sum(
            count * int(self._channel_cost[kind])
            for count, kind in zip(counts, kinds, strict=True)
        )
        return sum(counts), load

    def _sparse_draw_params(
        self, cap: int, length: int, lane: str
    ) -> tuple[float, dict[tuple[int, int, float], torch.Tensor], int]:
        if lane == "benchmark":
            return (
                float(self.config.benchmark_sparse_count_power),
                self._benchmark_weight_cache,
                max(
                    1,
                    min(cap, length, int(self.config.benchmark_sparse_keyframes_max)),
                ),
            )
        return (
            float(self.config.sparse_count_power),
            self._sparse_weight_cache,
            max(1, min(cap, length)),
        )

    def _end_effector_channel_cost(self, groups: list[str]) -> int:
        rotation_names, position_names = self.skeleton.expand_joint_names(groups)
        root = int(self.motion_rep.slice_dict["smooth_root_pos"].stop
                   - self.motion_rep.slice_dict["smooth_root_pos"].start)
        heading = int(
            self.motion_rep.slice_dict["global_root_heading"].stop
            - self.motion_rep.slice_dict["global_root_heading"].start
        )
        return root + heading + 3 * len(position_names) + 6 * len(rotation_names)

    def _sample_paper_ee_groups(self, generator: torch.Generator) -> list[str]:
        semantic_groups = ["LeftFoot", "RightFoot", "LeftHand", "RightHand"]
        selected_count = 1 + self._randint(len(semantic_groups), generator)
        order = torch.randperm(len(semantic_groups), generator=generator)[
            :selected_count
        ].tolist()
        return [semantic_groups[index] for index in order]

    def _resolve_end_effector_groups(
        self, selected: list[str], generator: torch.Generator
    ) -> list[str] | None:
        if "end_effector_sparse" in selected:
            return self._sample_paper_ee_groups(generator)
        for name in selected:
            groups = self.BENCHMARK_EE_GROUPS.get(name)
            if groups is not None:
                return list(groups)
        return None

    def _kind_costs_for(
        self, kinds: list[str], ee_groups: list[str] | None
    ) -> list[int]:
        costs: list[int] = []
        for kind in kinds:
            if kind == "end_effector" and ee_groups:
                costs.append(self._end_effector_channel_cost(ee_groups))
            else:
                costs.append(int(self._channel_cost[kind]))
        return costs

    def _prepare_affordable_budget(
        self,
        selected: list[str],
        kinds: list[str],
        cap: int,
        length: int,
        generator: torch.Generator,
        lane: str,
    ) -> tuple[int, int]:
        power, cache, maximum = self._sparse_draw_params(cap, length, lane)
        ee_groups = self._resolve_end_effector_groups(selected, generator)
        self._pending_ee_groups = ee_groups
        costs = self._kind_costs_for(kinds, ee_groups)
        counts = self._sample_affordable_counts(
            kinds, maximum, generator, power, cache, costs
        )
        counts = self._shrink_to_channel_budget(
            counts, kinds, sum(counts), kind_costs=costs
        )
        overlap = (
            lane == "paper_two"
            and len(kinds) >= 2
            and self._rand(generator)
            < float(self.config.sparse_same_frame_overlap_probability)
        )
        self._count_budget = counts
        self._frame_budget = self._allocate_kind_frames(
            counts, length, overlap, generator
        )
        load = sum(
            count * cost for count, cost in zip(counts, costs, strict=True)
        )
        return sum(counts), load

    def _sample_affordable_counts(
        self,
        kinds: list[str],
        maximum: int,
        generator: torch.Generator,
        power: float,
        cache: dict[tuple[int, int, float], torch.Tensor],
        costs: list[int],
    ) -> list[int]:
        del kinds
        order = torch.randperm(len(costs), generator=generator).tolist()
        remaining = int(self.config.sparse_channel_budget)
        counts = [1] * len(costs)
        for position, index in enumerate(order):
            reserved = sum(costs[later] for later in order[position + 1 :])
            affordable = (remaining - reserved) // costs[index]
            max_k = min(maximum, max(1, affordable))
            count = self._sample_sparse_count(
                max_k, generator, power, cache, constant_load=True
            )
            count = max(1, min(count, max_k))
            counts[index] = count
            remaining -= count * costs[index]
        return counts

    def _allocate_kind_frames(
        self,
        counts: list[int],
        length: int,
        overlap: bool,
        generator: torch.Generator,
    ) -> list[torch.Tensor]:
        if not overlap or len(counts) < 2:
            return [
                self._frames_from_count(length, count, generator) for count in counts
            ]
        shared_n = min(counts)
        shared_n = max(1, min(length, shared_n))
        perm = torch.randperm(length, generator=generator)
        shared = perm[:shared_n]
        rest = perm[shared_n:]
        cursor = 0
        frames: list[torch.Tensor] = []
        for count in counts:
            extra_n = max(0, min(count - shared_n, rest.numel() - cursor))
            extra = rest[cursor : cursor + extra_n]
            cursor += extra_n
            if extra.numel():
                frames.append(torch.cat([shared, extra]).sort().values)
            else:
                frames.append(shared.sort().values)
        return frames

    def _take_budget_slot(self) -> tuple[int | None, torch.Tensor | None]:
        if self._count_budget is None:
            return None, None
        if self._count_budget_index >= len(self._count_budget):
            raise RuntimeError("sparse count budget exhausted")
        index = self._count_budget_index
        self._count_budget_index += 1
        frames = None if self._frame_budget is None else self._frame_budget[index]
        return self._count_budget[index], frames

    def _frames_from_count(self, length: int, count: int, generator: torch.Generator) -> torch.Tensor:
        count = max(1, min(length, count))
        return torch.randperm(length, generator=generator)[:count].sort().values

    def _keyframes(self, length: int, maximum: int, generator: torch.Generator) -> torch.Tensor:
        budget_count, budget_frames = self._take_budget_slot()
        if budget_frames is not None:
            return budget_frames
        maximum = max(1, min(length, maximum))
        if budget_count is not None:
            return self._frames_from_count(length, min(budget_count, maximum), generator)
        count = self._sample_sparse_count(
            maximum,
            generator,
            float(self.config.sparse_count_power),
            self._sparse_weight_cache,
            constant_load=False,
        )
        return self._frames_from_count(length, count, generator)

    @staticmethod
    def _mark_feature(mask: torch.Tensor, frames: torch.Tensor, feature_slice: slice) -> None:
        if frames.device != mask.device:
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
        if frames.device != mask.device:
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
        selected = self._pending_ee_groups
        self._pending_ee_groups = None
        if selected is None:
            selected = self._sample_paper_ee_groups(generator)
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
        budget_count, budget_frames = self._take_budget_slot()
        if budget_frames is not None:
            return budget_frames
        if budget_count is not None:
            return self._frames_from_count(length, min(budget_count, maximum), generator)
        count = self._sample_sparse_count(
            maximum,
            generator,
            float(self.config.benchmark_sparse_count_power),
            self._benchmark_weight_cache,
            constant_load=False,
        )
        return self._frames_from_count(length, count, generator)

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
        if frames.device != mask.device:
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
        batch_size, max_time, feature_dim = clean_motion.shape
        names: list[list[str]] = []
        scheduled = self.scheduled_sparse_keyframes(global_step)
        maximum = self.maximum_sparse_keyframes(global_step)
        lengths_list = lengths.detach().cpu().tolist()
        if any(not 1 <= int(length) <= max_time for length in lengths_list):
            raise ValueError(f"Invalid sequence lengths for T={max_time}: {lengths_list}")

        in_phase2 = global_step >= self.config.phase1_steps
        if not in_phase2:
            mask = torch.zeros(
                batch_size,
                max_time,
                feature_dim,
                dtype=torch.bool,
                device=clean_motion.device,
            )
            return ConstraintBatch(
                torch.zeros_like(clean_motion),
                mask,
                [[] for _ in range(batch_size)],
                maximum,
                ["phase1_none" for _ in range(batch_size)],
                [0 for _ in range(batch_size)],
                scheduled,
                float(maximum),
                0.0,
                0.0,
            )

        # Build the boolean mask on CPU: all RNG draws already use a CPU
        # generator, so staying on host avoids per-pattern GPU transfers while
        # preserving the exact multinomial / randperm / randint order.
        mask_cpu = torch.zeros(batch_size, max_time, feature_dim, dtype=torch.bool)
        lanes: list[str] = []
        component_counts: list[int] = []
        sampled_caps: list[int] = []
        sampled_counts: list[int] = []
        sampled_loads: list[int] = []
        for batch_index, length_value in enumerate(lengths_list):
            length = int(length_value)
            selected, lane, component_count = self._select_patterns(generator)
            cap = self.sample_sparse_keyframe_cap(global_step, generator)
            sampled_caps.append(cap)
            count_sum, load = self._prepare_count_budget(
                selected, cap, length, generator, lane
            )
            sampled_counts.append(count_sum)
            sampled_loads.append(load)
            if selected:
                for name in selected:
                    self._apply_pattern(name, mask_cpu[batch_index], length, cap, generator)
            self._count_budget = None
            self._frame_budget = None
            self._pending_ee_groups = None
            names.append(selected)
            lanes.append(lane)
            component_counts.append(component_count)

        mask = mask_cpu.to(device=clean_motion.device)
        observed = torch.where(mask, clean_motion, torch.zeros_like(clean_motion))
        sampled_mean = float(sum(sampled_caps) / len(sampled_caps))
        mask_loads = mask_cpu.reshape(batch_size, -1).sum(dim=1).to(dtype=torch.float32)
        return ConstraintBatch(
            observed,
            mask,
            names,
            maximum,
            lanes,
            component_counts,
            scheduled,
            sampled_mean,
            float(sum(sampled_counts) / len(sampled_counts)),
            float(sum(sampled_loads) / len(sampled_loads)),
            float(mask_loads.mean().item()),
            float(mask_loads.max().item()),
        )
