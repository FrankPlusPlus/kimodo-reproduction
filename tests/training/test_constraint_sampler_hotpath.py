"""Parity tests for Phase-2 constraint hot-path optimizations."""

from __future__ import annotations

import torch

from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton
from kimodo.training.config import CurriculumConfig
from kimodo.training.constraints import ConstraintCurriculumSampler


def _sampler(**overrides) -> ConstraintCurriculumSampler:
    rep = KimodoMotionRep(build_skeleton(30), fps=30, stats_path=None)
    config = CurriculumConfig(
        phase1_steps=10,
        phase2_steps=100,
        sparse_keyframes_min=1,
        sparse_keyframes_max=4,
        benchmark_coverage_probability=0.25,
        **overrides,
    )
    return ConstraintCurriculumSampler(rep, config)


def _sample_legacy_device_mask(
    sampler: ConstraintCurriculumSampler,
    clean_motion: torch.Tensor,
    lengths: torch.Tensor,
    global_step: int,
    generator: torch.Generator,
):
    """Pre-optimization sample path: build the mask on clean_motion.device."""
    batch_size, max_time, _ = clean_motion.shape
    mask = torch.zeros_like(clean_motion, dtype=torch.bool)
    names: list[list[str]] = []
    maximum = sampler.maximum_sparse_keyframes(global_step)
    lengths_list = lengths.detach().cpu().tolist()
    lanes: list[str] = []
    component_counts: list[int] = []
    for batch_index, length_value in enumerate(lengths_list):
        length = int(length_value)
        selected, lane, component_count = sampler._select_patterns(generator)
        if selected:
            for name in selected:
                sampler._apply_pattern(name, mask[batch_index], length, maximum, generator)
        names.append(selected)
        lanes.append(lane)
        component_counts.append(component_count)
    observed = torch.where(mask, clean_motion, torch.zeros_like(clean_motion))
    return observed, mask, names, lanes, component_counts


def test_power_law_weight_cache_matches_fresh_tensor():
    sampler = _sampler()
    weights = sampler._power_law_weights(1, 4, 1.5, sampler._sparse_weight_cache)
    fresh = torch.arange(1, 5, dtype=torch.float32).pow(-1.5)
    assert torch.equal(weights, fresh)
    cached = sampler._power_law_weights(1, 4, 1.5, sampler._sparse_weight_cache)
    assert cached is weights


def test_cached_keyframes_match_uncached_rng_draws():
    sampler = _sampler()
    cached_gen = torch.Generator().manual_seed(91)
    fresh_gen = torch.Generator().manual_seed(91)
    frames_cached = []
    frames_fresh = []
    for length in (8, 12, 16, 8, 20):
        frames_cached.append(sampler._keyframes(length, 4, cached_gen).clone())
        maximum = max(1, min(length, 4))
        minimum = min(sampler.config.sparse_keyframes_min, maximum)
        choices = torch.arange(minimum, maximum + 1, dtype=torch.float32)
        weights = choices.pow(-float(sampler.config.sparse_count_power))
        selected = int(torch.multinomial(weights, 1, generator=fresh_gen).item())
        count = minimum + selected
        frames_fresh.append(torch.randperm(length, generator=fresh_gen)[:count].sort().values)
    for left, right in zip(frames_cached, frames_fresh, strict=True):
        assert torch.equal(left, right)
    assert torch.equal(cached_gen.get_state(), fresh_gen.get_state())


def test_phase2_cpu_mask_matches_legacy_device_mask_and_rng():
    sampler = _sampler()
    clean = torch.randn(6, 24, sampler.motion_rep.motion_rep_dim)
    lengths = torch.tensor([24, 18, 12, 20, 16, 8], dtype=torch.long)
    global_step = sampler.config.phase1_steps + 5

    optimized_gen = torch.Generator().manual_seed(20260810)
    legacy_gen = torch.Generator().manual_seed(20260810)
    optimized = sampler.sample(clean, lengths, global_step, optimized_gen)
    legacy_observed, legacy_mask, legacy_names, legacy_lanes, legacy_components = (
        _sample_legacy_device_mask(sampler, clean, lengths, global_step, legacy_gen)
    )

    assert optimized.pattern_names == legacy_names
    assert optimized.sampling_lanes == legacy_lanes
    assert optimized.component_counts == legacy_components
    assert torch.equal(optimized.motion_mask, legacy_mask)
    assert torch.equal(optimized.observed_motion, legacy_observed)
    assert torch.equal(optimized_gen.get_state(), legacy_gen.get_state())


def test_phase2_sample_identical_on_cpu_and_cuda_when_available():
    if not torch.cuda.is_available():
        return
    sampler = _sampler()
    clean_cpu = torch.randn(4, 20, sampler.motion_rep.motion_rep_dim)
    clean_cuda = clean_cpu.cuda()
    lengths = torch.tensor([20, 16, 12, 8], dtype=torch.long)
    global_step = sampler.config.phase1_steps + 1

    cpu_gen = torch.Generator().manual_seed(17)
    cuda_gen = torch.Generator().manual_seed(17)
    cpu_batch = sampler.sample(clean_cpu, lengths, global_step, cpu_gen)
    cuda_batch = sampler.sample(clean_cuda, lengths.cuda(), global_step, cuda_gen)

    assert cpu_batch.pattern_names == cuda_batch.pattern_names
    assert cpu_batch.sampling_lanes == cuda_batch.sampling_lanes
    assert torch.equal(cpu_batch.motion_mask, cuda_batch.motion_mask.cpu())
    assert torch.allclose(cpu_batch.observed_motion, cuda_batch.observed_motion.cpu())
    assert torch.equal(cpu_gen.get_state(), cuda_gen.get_state())


def test_phase1_sample_skips_pattern_rng():
    sampler = _sampler()
    clean = torch.randn(2, 10, sampler.motion_rep.motion_rep_dim)
    lengths = torch.tensor([10, 8], dtype=torch.long)
    generator = torch.Generator().manual_seed(3)
    before = generator.get_state().clone()
    batch = sampler.sample(clean, lengths, global_step=0, generator=generator)
    assert batch.sampling_lanes == ["phase1_none", "phase1_none"]
    assert not batch.motion_mask.any()
    assert torch.equal(generator.get_state(), before)
