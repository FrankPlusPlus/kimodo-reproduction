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
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict OmegaConf dot-list override; may be repeated",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print config without importing training runtime")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    overrides = list(args.set)
    if args.dry_run:
        overrides.append("runtime.dry_run=true")
    config = load_training_config(args.config, overrides)
    if args.dry_run:
        config.validate(require_paths=False)
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return

    from .engine import KimodoTrainer

    project_root = Path(__file__).resolve().parents[2]
    trainer = KimodoTrainer(config, project_root)
    trainer.train()


if __name__ == "__main__":
    main()

