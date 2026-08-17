#!/usr/bin/env python3
"""Read-only layer-weight RMS timeline. No forward, no data, no optimizer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from kimodo.training.body_layer_weight_probe import (
    checkpoint_layer_rms,
    summarize_layer_weight_timeline,
)


def _load_checkpoint_weights(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    weights = state.get("model")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Checkpoint has no model weights: {path}")
    return weights, {"global_step": int(state["global_step"])}


def diagnose(args) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "running.json").write_text(
            json.dumps({"status": "starting"}, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "starting", "output_dir": str(output_dir)}), flush=True)
    layers = [int(index) for index in (args.layers or [0, 7, 14, 15])]
    rows = []
    for checkpoint in args.checkpoints:
        path = Path(checkpoint).expanduser().resolve()
        print(json.dumps({"status": "loading_checkpoint", "path": str(path)}), flush=True)
        weights, summary = _load_checkpoint_weights(path)
        probe = checkpoint_layer_rms(weights, layers=layers)
        row = {
            "checkpoint": str(path),
            "global_step": int(summary["global_step"]),
            "probe": probe,
        }
        rows.append(row)
        if output_dir is not None:
            name = f"partial-step-{int(summary['global_step']):09d}.json"
            (output_dir / name).write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        l15 = probe.get("layer_15") or {}
        print(
            json.dumps(
                {
                    "status": "probed",
                    "global_step": int(summary["global_step"]),
                    "l00_in_proj": (probe.get("layer_00") or {}).get("in_proj_rms"),
                    "l15_in_proj": l15.get("in_proj_rms"),
                    "l15_ffn_out": l15.get("ffn_out_rms"),
                }
            ),
            flush=True,
        )
        del weights
    verdict = summarize_layer_weight_timeline(
        rows,
        start_step=int(args.start_step),
        end_step=int(args.end_step),
    )
    payload = {
        "rank": int(os.environ.get("RANK", "0")),
        "rows": rows,
        "verdict": verdict,
    }
    print(json.dumps({"verdict": verdict}, indent=2), flush=True)
    if output_dir is not None:
        (output_dir / "rank-00.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "verdict.json").write_text(
            json.dumps({"verdict": verdict}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        running = output_dir / "running.json"
        if running.is_file():
            running.unlink()
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--layers", nargs="*", type=int)
    parser.add_argument("--start-step", type=int, default=650000)
    parser.add_argument("--end-step", type=int, default=695000)
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    diagnose(build_parser().parse_args())


if __name__ == "__main__":
    main()
