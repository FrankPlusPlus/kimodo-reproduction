# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point for the reconstructed Kimodo trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_training_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML training config")
    parser.add_argument(
        "--paths",
        help=(
            "Optional schema-v1 YAML containing only data/model/run path fields; "
            "merged after the base config"
        ),
    )
    parser.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="Strict training/hardware YAML overlay; may be repeated in merge order",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict OmegaConf dot-list override; may be repeated",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print config without importing training runtime")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Validate the complete manifest/inventory contract, then construct one representative "
            "CPU batch without allocating the denoiser"
        ),
    )
    return parser


def _run_data_preflight(config) -> dict:
    from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
    from kimodo.skeleton.registry import build_skeleton

    from .data import MotionManifestDataset, collate_motion_batch
    from .reference_inventory import load_inventory_summary

    if config.model.checkpoint_dir is not None and not config.model.stats_path:
        raise ValueError(
            "data-only preflight requires model.stats_path; point it at the official bundle stats"
        )
    if config.data.reference_verification == "inventory":
        load_inventory_summary(config.data.manifest, config.data.reference_inventory)
    motion_rep = KimodoMotionRep(
        skeleton=build_skeleton(config.model.skeleton_joints),
        fps=config.model.fps,
        stats_path=config.model.stats_path,
    )
    dataset = MotionManifestDataset(
        config.data.manifest,
        config.data.split,
        motion_rep,
        max_seconds=config.data.max_seconds,
        min_frames=config.data.min_frames,
        seed=config.runtime.seed,
        require_cached_text=config.data.require_cached_text,
        require_paper_data_parity=config.data.require_paper_data_parity,
        normalize=True,
        augment=True,
    )
    sample_count = min(config.runtime.batch_size, len(dataset))
    batch = collate_motion_batch([dataset[index] for index in range(sample_count)])
    if batch["text_features"] is None:
        raise ValueError("preflight requires cached text embeddings")
    if int(batch["text_features"].shape[-1]) != config.model.llm_dim:
        raise ValueError("preflight text embedding width does not match model.llm_dim")
    return {
        "event": "kimodo_full_data_preflight_passed",
        "manifest_entries_validated": dataset.manifest_entries,
        "dataset_entries": len(dataset),
        "excluded_short_entries": dataset.excluded_short_entries,
        "excluded_short_temporal_entries": dataset.excluded_short_temporal_entries,
        "excluded_short_full_entries": dataset.excluded_short_full_entries,
        "sampled_entries": sample_count,
        "motion_shape": list(batch["clean_motion"].shape),
        "text_shape": list(batch["text_features"].shape),
        "mixture_sources": list(dataset.mixture_sources),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run and args.preflight:
        raise SystemExit("--dry-run and --preflight are mutually exclusive")
    overrides = list(args.set)
    if args.dry_run:
        overrides.append("runtime.dry_run=true")
    config = load_training_config(
        args.config,
        overrides,
        paths=args.paths,
        overlays=args.overlay,
    )
    if args.dry_run:
        config.validate(require_paths=False)
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return
    if args.preflight:
        print(json.dumps(_run_data_preflight(config), indent=2, sort_keys=True))
        return

    from .engine import KimodoTrainer

    project_root = Path(__file__).resolve().parents[2]
    try:
        trainer = KimodoTrainer(config, project_root)
        trainer.train()
    finally:
        # torchrun expects the application to tear down NCCL explicitly.  The
        # trainer itself does not do this because library callers may run
        # several trainers inside one process group (the exact-resume tests do).
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
