#!/usr/bin/env python3
"""Write a 20k drift-health snapshot from train.jsonl plus residual-cancel probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kimodo.training.lastwd1_drift_health import (
    build_health_record,
    load_jsonl_rows,
    nearest_jsonl_row,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--probe-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    probe_payload = None
    if args.probe_dir:
        verdict = Path(args.probe_dir) / "verdict.json"
        rank = Path(args.probe_dir) / "rank-00.json"
        path = rank if rank.is_file() else verdict
        if path.is_file():
            probe_payload = json.loads(path.read_text(encoding="utf-8"))
    rows = load_jsonl_rows(Path(args.jsonl))
    record = build_health_record(
        step=int(args.step),
        jsonl_row=nearest_jsonl_row(rows, int(args.step)),
        probe_payload=probe_payload,
        checkpoint=args.checkpoint,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "wrote", "health": record["health"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
