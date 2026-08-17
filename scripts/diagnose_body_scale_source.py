#!/usr/bin/env python3
"""Read-only scale-source probe: why pre-LN σ shrinks.

Weights-only by default (CPU-fast). Optionally ingest residual-cancel JSON
dirs already on disk so activation decomposition does not require a second
forward. No optimizer step, no weight write.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from kimodo.training.body_scale_source_probe import (
    checkpoint_scale_weights,
    compare_weight_scale,
    decompose_sigma_change,
    layer_named,
    residual_stack_ratios,
    summarize_scale_source,
)


def _load_checkpoint_weights(path: Path) -> tuple[dict[str, torch.Tensor], int]:
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    weights = state.get("model")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Checkpoint has no model weights: {path}")
    return weights, int(state["global_step"])


def _load_activation_rows(directories: list[Path]) -> dict[int, dict[str, Any]]:
    by_step: dict[int, dict] = {}
    for directory in directories:
        for path in sorted(directory.glob("partial-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            try:
                step = int(payload.get("global_step"))
            except (TypeError, ValueError):
                continue
            if step not in by_step:
                by_step[step] = payload
    return by_step


def diagnose(args) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "running.json").write_text(
            json.dumps({"status": "starting"}, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "starting", "output_dir": str(output_dir)}), flush=True)

    layers = list(range(16)) if args.all_layers else [int(index) for index in (args.layers or [0, 7, 14, 15])]
    weight_rows = []
    for checkpoint in args.checkpoints or []:
        path = Path(checkpoint).expanduser().resolve()
        print(json.dumps({"status": "loading_checkpoint", "path": str(path)}), flush=True)
        weights, step = _load_checkpoint_weights(path)
        probe = checkpoint_scale_weights(weights, layers=layers)
        row = {"checkpoint": str(path), "global_step": step, "probe": probe}
        weight_rows.append(row)
        if output_dir is not None:
            (output_dir / f"weights-step-{step:09d}.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(
            json.dumps(
                {
                    "status": "weighed",
                    "global_step": step,
                    "input_linear_rms": probe.get("input_linear_rms"),
                    "mean_ln1_gamma_rms": probe.get("mean_ln1_gamma_rms"),
                    "l15_in_proj": (probe.get("layers") or {}).get("layer_15", {}).get("in_proj_rms"),
                }
            ),
            flush=True,
        )
        del weights

    activations = _load_activation_rows([Path(item).expanduser().resolve() for item in (args.activation_dirs or [])])
    activation_pairs = []
    if args.base_step is not None and args.end_step is not None:
        pair_list = [(int(args.base_step), int(args.end_step))]
    else:
        steps = sorted(activations)
        pair_list = [(steps[0], step) for step in steps[1:]] if len(steps) >= 2 else []
        if args.windows:
            pair_list = []
            for window in args.windows:
                start_s, end_s = window.split(":")
                pair_list.append((int(start_s), int(end_s)))

    comparisons = []
    for start_step, end_step in pair_list:
        base_act = activations.get(start_step)
        other_act = activations.get(end_step)
        sigma = {}
        stack = {}
        if base_act is not None and other_act is not None:
            base_layers = layer_named(base_act.get("probe") or {})
            other_layers = layer_named(other_act.get("probe") or {})
            if 15 in base_layers and 15 in other_layers:
                sigma = decompose_sigma_change(base_layers[15], other_layers[15])
            stack = residual_stack_ratios(base_layers, other_layers)
        weights_cmp = None
        base_w = next((row for row in weight_rows if int(row["global_step"]) == start_step), None)
        other_w = next((row for row in weight_rows if int(row["global_step"]) == end_step), None)
        if base_w is not None and other_w is not None:
            weights_cmp = compare_weight_scale(base_w["probe"], other_w["probe"])
        summary = summarize_scale_source(sigma=sigma, stack_ratios=stack, weights=weights_cmp)
        item = {
            "start_step": start_step,
            "end_step": end_step,
            **summary,
        }
        comparisons.append(item)
        activation_pairs.append(item)
        print(json.dumps({"status": "compared", **item}, default=str)[:2000], flush=True)

    payload = {
        "weight_rows": weight_rows,
        "comparisons": comparisons,
        "verdict": comparisons[-1] if comparisons else {"source": "incomplete"},
    }
    print(json.dumps({"verdict": payload["verdict"]}, indent=2, default=str), flush=True)
    if output_dir is not None:
        (output_dir / "rank-00.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        (output_dir / "verdict.json").write_text(
            json.dumps({"verdict": payload["verdict"]}, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        running = output_dir / "running.json"
        if running.is_file():
            running.unlink()
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--activation-dirs", nargs="*", default=[])
    parser.add_argument("--layers", nargs="*", type=int)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--base-step", type=int)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--windows", nargs="*", help="start:end pairs, e.g. 500000:550000 780000:790000")
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    diagnose(build_parser().parse_args())


if __name__ == "__main__":
    main()
