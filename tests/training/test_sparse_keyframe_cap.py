"""Adjacent-integer sparse-keyframe cap mixing."""

from __future__ import annotations

import pytest
import torch

from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton
from kimodo.training.config import CurriculumConfig
from kimodo.training.constraints import ConstraintCurriculumSampler


def _sampler(**overrides) -> ConstraintCurriculumSampler:
    rep = KimodoMotionRep(build_skeleton(30), fps=30, stats_path=None)
    config = CurriculumConfig(
        phase1_steps=0,
        phase2_steps=11,
        sparse_keyframes_min=7,
        sparse_keyframes_max=8,
        sparse_keyframe_cap_mode="adjacent_mix",
        **overrides,
    )
    return ConstraintCurriculumSampler(rep, config)


def test_adjacent_mix_uses_fractional_part_as_probability():
    sampler = _sampler()
    assert sampler.scheduled_sparse_keyframes(0) == 7.0
    assert sampler.scheduled_sparse_keyframes(1) == pytest.approx(7.1)
    assert sampler.scheduled_sparse_keyframes(10) == 8.0

    generator = torch.Generator().manual_seed(0)
    before = generator.get_state().clone()
    assert sampler.sample_sparse_keyframe_cap(0, generator) == 7
    assert torch.equal(generator.get_state(), before)
    assert sampler.sample_sparse_keyframe_cap(10, generator) == 8
    assert torch.equal(generator.get_state(), before)

    caps = [sampler.sample_sparse_keyframe_cap(1, generator) for _ in range(8_000)]
    assert set(caps) <= {7, 8}
    assert sum(cap == 8 for cap in caps) / len(caps) == pytest.approx(0.1, abs=0.02)


def test_round_mode_still_jumps_at_half_integers():
    sampler = _sampler(sparse_keyframe_cap_mode="round")
    assert sampler.maximum_sparse_keyframes(4) == 7
    assert sampler.maximum_sparse_keyframes(5) == 8
    generator = torch.Generator().manual_seed(3)
    before = generator.get_state().clone()
    assert sampler.sample_sparse_keyframe_cap(5, generator) == 8
    assert torch.equal(generator.get_state(), before)


def test_constant_load_freezes_high_density_mass_when_support_grows():
    sampler = ConstraintCurriculumSampler(
        KimodoMotionRep(build_skeleton(30), fps=30, stats_path=None),
        CurriculumConfig(
            phase1_steps=0,
            phase2_steps=20,
            sparse_keyframes_min=1,
            sparse_keyframes_max=20,
            sparse_count_power=1.0,
        ),
    )
    baseline = sampler._constant_load_weights(1, 7, 1.0)
    opened_eight = sampler._constant_load_weights(1, 8, 1.0)
    opened_twenty = sampler._constant_load_weights(1, 20, 1.0)
    frozen_tail = float(baseline[6].item())
    assert float(opened_eight[6:].sum()) == pytest.approx(frozen_tail, abs=1e-6)
    assert float(opened_twenty[6:].sum()) == pytest.approx(frozen_tail, abs=1e-6)
    assert frozen_tail == pytest.approx(0.0551, abs=1e-3)

    renormalized_eight = torch.arange(1, 9, dtype=torch.float32).pow(-1.0)
    renormalized_eight = renormalized_eight / renormalized_eight.sum()
    assert float(opened_eight[6:].sum()) < float(renormalized_eight[6:].sum()) - 0.03


def test_k7_does_not_share_or_freeze_budget():
    sampler = _paper_sampler()
    generator = torch.Generator().manual_seed(0)
    count, load = sampler._prepare_count_budget(
        ["full_body_sparse", "end_effector_sparse"],
        cap=7,
        length=32,
        generator=generator,
        lane="paper_two",
    )
    assert count == 0
    assert load == 0
    assert sampler._count_budget is None


def test_k8_shares_one_sample_budget_across_two_sparse_patterns():
    sampler = _paper_sampler()
    generator = torch.Generator().manual_seed(1)
    count, load = sampler._prepare_count_budget(
        ["full_body_sparse", "end_effector_sparse"],
        cap=8,
        length=32,
        generator=generator,
        lane="paper_two",
    )
    assert sampler._count_budget is not None
    assert len(sampler._count_budget) == 2
    assert sum(sampler._count_budget) == count
    assert count >= 2
    assert count <= 8
    assert load == (
        sampler._count_budget[0] * sampler._channel_cost["full_body"]
        + sampler._count_budget[1] * sampler._channel_cost["end_effector"]
    )


def test_path_and_inbetweening_do_not_consume_sparse_budget():
    sampler = _paper_sampler()
    generator = torch.Generator().manual_seed(2)
    count, load = sampler._prepare_count_budget(
        ["benchmark_full_body_inbetweening"],
        cap=20,
        length=32,
        generator=generator,
        lane="benchmark",
    )
    assert count == 0
    assert load == 0
    assert sampler._count_budget is None
    count, load = sampler._prepare_count_budget(
        ["benchmark_root_path_2dpos"],
        cap=20,
        length=32,
        generator=generator,
        lane="benchmark",
    )
    assert count == 0
    assert sampler._count_budget is None


def test_v2_mix_shares_budget_and_ignores_the_uncapped_root_path():
    sampler = _paper_sampler()
    generator = torch.Generator().manual_seed(3)
    count, _load = sampler._prepare_count_budget(
        ["benchmark_mix_root_path_fullbody"],
        cap=9,
        length=32,
        generator=generator,
        lane="benchmark",
    )
    assert sampler._count_budget is not None
    assert len(sampler._count_budget) == 1
    assert count == sampler._count_budget[0]
    assert count <= 9


def _paper_sampler() -> ConstraintCurriculumSampler:
    return ConstraintCurriculumSampler(
        KimodoMotionRep(build_skeleton(30), fps=30, stats_path=None),
        CurriculumConfig(
            phase1_steps=0,
            phase2_steps=20,
            sparse_keyframes_min=1,
            sparse_keyframes_max=20,
            benchmark_coverage_probability=0.25,
        ),
    )


def test_v2_1m_blend_reaches_k8_and_k9_later_than_the_hard_jump():
    sampler = ConstraintCurriculumSampler(
        KimodoMotionRep(build_skeleton(30), fps=30, stats_path=None),
        CurriculumConfig(
            phase1_steps=500_000,
            phase2_steps=500_000,
            sparse_keyframes_min=1,
            sparse_keyframes_max=20,
            sparse_keyframe_cap_mode="adjacent_mix",
        ),
    )
    assert sampler.maximum_sparse_keyframes(500_000) == 1
    assert sampler.maximum_sparse_keyframes(999_999) == 20
    # Old round() entered Kmax=8 at 671053; mixing starts leaking K=8 once
    # the continuous cap exceeds 7, and is fully K=8 only at 8.0.
    assert sampler.scheduled_sparse_keyframes(657_894) < 7.0
    assert sampler.scheduled_sparse_keyframes(657_895) > 7.0
    assert sampler.scheduled_sparse_keyframes(671_053) == pytest.approx(7.5, abs=0.01)
    assert sampler.scheduled_sparse_keyframes(684_210) < 8.0
    assert sampler.scheduled_sparse_keyframes(684_211) >= 8.0
    assert sampler.scheduled_sparse_keyframes(710_526) == pytest.approx(9.0, abs=0.01)


def test_hard_cap_freezes_sampled_support_without_rescaling_the_ramp():
    sampler = ConstraintCurriculumSampler(
        KimodoMotionRep(build_skeleton(30), fps=30, stats_path=None),
        CurriculumConfig(
            phase1_steps=500_000,
            phase2_steps=500_000,
            sparse_keyframes_min=1,
            sparse_keyframes_max=20,
            sparse_keyframe_cap_mode="adjacent_mix",
            sparse_keyframes_hard_cap=7,
        ),
    )
    assert sampler.scheduled_sparse_keyframes(695_000) == pytest.approx(8.41, abs=0.01)
    assert sampler.maximum_sparse_keyframes(695_000) == 7
    generator = torch.Generator().manual_seed(0)
    caps = [sampler.sample_sparse_keyframe_cap(695_000, generator) for _ in range(200)]
    assert caps == [7] * 200
    uncapped = ConstraintCurriculumSampler(
        KimodoMotionRep(build_skeleton(30), fps=30, stats_path=None),
        CurriculumConfig(
            phase1_steps=500_000,
            phase2_steps=500_000,
            sparse_keyframes_min=1,
            sparse_keyframes_max=20,
            sparse_keyframe_cap_mode="adjacent_mix",
        ),
    )
    mixed = [
        uncapped.sample_sparse_keyframe_cap(695_000, torch.Generator().manual_seed(seed))
        for seed in range(40)
    ]
    assert set(mixed) <= {8, 9}
    assert 8 in mixed and 9 in mixed


def test_legacy_total_times_full_body_budget_does_not_cut_nine_full_body():
    sampler = _paper_sampler()
    assert sampler.config.sparse_channel_budget == 0
    assert sampler._channel_cost["full_body"] == 95
    assert sampler._channel_cost["end_effector"] == 40
    assert sampler._shrink_to_channel_budget([9], ["full_body"], 9) == [9]


def _affordable_sampler(**overrides) -> ConstraintCurriculumSampler:
    return ConstraintCurriculumSampler(
        KimodoMotionRep(build_skeleton(30), fps=30, stats_path=None),
        CurriculumConfig(
            phase1_steps=0,
            phase2_steps=20,
            sparse_keyframes_min=1,
            sparse_keyframes_max=20,
            sparse_channel_budget=800,
            sparse_same_frame_overlap_probability=0.0,
            **overrides,
        ),
    )


def test_affordable_sampling_never_draws_nine_full_body_and_fits_800():
    sampler = _affordable_sampler()
    fb = sampler._channel_cost["full_body"]
    assert 800 // fb == 8
    for seed in range(80):
        generator = torch.Generator().manual_seed(seed)
        count, load = sampler._prepare_count_budget(
            ["full_body_sparse"],
            cap=20,
            length=64,
            generator=generator,
            lane="paper_single",
        )
        assert sampler._count_budget == [count]
        assert count <= 8
        assert load == count * fb
        assert load <= 800

        generator = torch.Generator().manual_seed(seed + 1_000)
        count, load = sampler._prepare_count_budget(
            ["end_effector_sparse"],
            cap=20,
            length=64,
            generator=generator,
            lane="paper_single",
        )
        ee_cost = sampler._end_effector_channel_cost(sampler._pending_ee_groups)
        assert count <= 800 // ee_cost
        assert load == count * ee_cost
        assert load <= 800

        generator = torch.Generator().manual_seed(seed + 2_000)
        count, load = sampler._prepare_count_budget(
            ["full_body_sparse", "end_effector_sparse"],
            cap=20,
            length=64,
            generator=generator,
            lane="paper_two",
        )
        fb_count, ee_count = sampler._count_budget
        ee_cost = sampler._end_effector_channel_cost(sampler._pending_ee_groups)
        assert fb_count >= 1
        assert ee_count >= 1
        assert fb_count <= 8
        assert ee_count <= 800 // ee_cost
        assert load == fb_count * fb + ee_count * ee_cost
        assert load <= 800
        assert count == fb_count + ee_count

        generator = torch.Generator().manual_seed(seed + 3_000)
        count, load = sampler._prepare_count_budget(
            ["full_body_sparse", "root_sparse"],
            cap=20,
            length=64,
            generator=generator,
            lane="paper_two",
        )
        fb_count, root_count = sampler._count_budget
        root_cost = sampler._channel_cost["root"]
        assert root_cost == 4
        assert fb_count <= 8
        assert load == fb_count * fb + root_count * root_cost
        assert load <= 800
        assert count == fb_count + root_count


def test_four_group_ee_max_count_fits_real_cost():
    sampler = _affordable_sampler()
    groups = ["LeftFoot", "RightFoot", "LeftHand", "RightHand"]
    cost = sampler._end_effector_channel_cost(groups)
    assert cost == 53
    assert 800 // cost == 15
    generator = torch.Generator().manual_seed(0)
    counts = sampler._sample_affordable_counts(
        ["end_effector"],
        20,
        generator,
        1.0,
        sampler._sparse_weight_cache,
        [cost],
    )
    assert counts[0] <= 15
    assert counts[0] * cost <= 800


def test_paper_two_overlap_puts_smaller_set_on_shared_frames():
    sampler = _affordable_sampler(sparse_same_frame_overlap_probability=1.0)
    generator = torch.Generator().manual_seed(11)
    sampler._prepare_count_budget(
        ["full_body_sparse", "end_effector_sparse"],
        cap=20,
        length=64,
        generator=generator,
        lane="paper_two",
    )
    left, right = sampler._frame_budget
    left_set = set(left.tolist())
    right_set = set(right.tolist())
    shared = left_set & right_set
    smaller = left_set if len(left_set) <= len(right_set) else right_set
    assert shared == smaller
