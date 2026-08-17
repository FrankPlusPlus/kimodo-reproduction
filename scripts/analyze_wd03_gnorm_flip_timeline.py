#!/usr/bin/env python3
"""Experiment U: merge wd03 jsonl gnorm with full-stack attn cosine timeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kimodo.training.wd03_gnorm_flip_timeline import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-jsonl",
        default="/home/share/yezitao-kimodo-reproduction/runs/v2-1m-hostnet-wd03-from650k/train.jsonl",
    )
    parser.add_argument(
        "--full-stack-json",
        help="verdict.json or rank-00.json from experiment R (uses full_stack_timeline key)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--step-start", type=int, default=650_000)
    parser.add_argument("--step-every", type=int, default=10_000)
    parser.add_argument("--step-end", type=int, default=800_000)
    args = parser.parse_args()

    full_stack: list[dict] = []
    if args.full_stack_json:
        payload = json.loads(Path(args.full_stack_json).read_text(encoding="utf-8"))
        full_stack = payload.get("full_stack_timeline") or payload.get("timeline") or []

    report = build_report(
        train_jsonl=Path(args.train_jsonl).expanduser().resolve(),
        full_stack_timeline=full_stack,
        step_start=int(args.step_start),
        step_every=int(args.step_every),
        step_end=int(args.step_end),
    )
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "verdict.json").write_text(
        json.dumps(report["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
