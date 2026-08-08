#!/usr/bin/env python3
"""Refresh checkpoint provenance so exact resume accepts the current code tree."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kimodo.training.config import load_training_config
from kimodo.training.provenance import collect_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/kimodo_soma_seed_public.yaml")
    parser.add_argument("--paths", required=True)
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    overlays = [Path(value) for value in args.overlay]
    config = load_training_config(
        root / args.config,
        list(args.overrides),
        paths=Path(args.paths),
        overlays=overlays,
    )
    config.validate(require_paths=False)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["provenance"] = collect_provenance(config, root)
    # Keep resume-critical config aligned with the current training tree so exact
    # resume accepts max_steps_override and other allowed runtime-only changes.
    state["config"] = config.to_dict()
    torch.save(state, checkpoint)
    print(f"patched provenance and config: {checkpoint}")


if __name__ == "__main__":
    main()
