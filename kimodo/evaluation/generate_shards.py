# SPDX-License-Identifier: Apache-2.0
"""Disjoint example sharding for multi-GPU benchmark generation."""

from __future__ import annotations

from pathlib import Path


def resolve_generation_shard(
    shard_index: int | None,
    shard_count: int | None,
    *,
    environ: dict[str, str] | None = None,
) -> tuple[int, int]:
    """Resolve a disjoint example shard from CLI flags or torchrun RANK/WORLD_SIZE."""
    env = environ if environ is not None else {}
    env_rank = env.get("RANK")
    env_world = env.get("WORLD_SIZE")
    count = shard_count if shard_count is not None else (int(env_world) if env_world not in (None, "") else 1)
    index = shard_index if shard_index is not None else (int(env_rank) if env_rank not in (None, "") else 0)
    if count < 1:
        raise ValueError(f"shard count must be >= 1, got {count}")
    if not 0 <= index < count:
        raise ValueError(f"shard index {index} is outside [0, {count})")
    return index, count


def select_example_shard(
    examples: list[tuple[Path, Path]],
    shard_index: int,
    shard_count: int,
) -> list[tuple[Path, Path]]:
    """Keep every Nth discovered example so ranks cover the tree without overlap."""
    if shard_count == 1:
        return list(examples)
    return [item for index, item in enumerate(examples) if index % shard_count == shard_index]
