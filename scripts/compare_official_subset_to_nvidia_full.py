#!/usr/bin/env python3
"""Compare Official SEED summary on a local subset against NVIDIA published full-suite tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NVIDIA_FULL = {
    "content": {
        "text_following": {
            "overview": {"R@3": 81.13, "FID": 0.035, "Skate": 4.077, "Contact": 0.977},
            "timeline_single": {"R@3": 73.17, "FID": 0.028, "Skate": 3.873, "Contact": 0.980},
            "timeline_multi": {"R@3": 80.10, "FID": 0.032, "Skate": 3.685, "Contact": 0.981},
        },
        "constraints_withtext": {
            "FB": 3.421,
            "EE": 3.817,
            "Root2D": 4.979,
            "Pelvis95": 9.14,
        },
        "constraints_notext": {
            "FB": 3.320,
            "EE": 3.664,
            "Root2D": 4.797,
            "Pelvis95": 9.03,
        },
    },
    "repetition": {
        "text_following": {
            "overview": {"R@3": 90.92, "FID": 0.004, "Skate": 4.573, "Contact": 0.972},
            "timeline_single": {"R@3": 80.38, "FID": 0.007, "Skate": 4.442, "Contact": 0.976},
            "timeline_multi": {"R@3": 92.58, "FID": 0.006, "Skate": 4.199, "Contact": 0.974},
        },
        "constraints_withtext": {
            "FB": 3.187,
            "EE": 3.852,
            "Root2D": 4.734,
            "Pelvis95": 9.19,
        },
        "constraints_notext": {
            "FB": 3.120,
            "EE": 3.510,
            "Root2D": 4.264,
            "Pelvis95": 7.89,
        },
    },
}

ROW_LABELS = {
    "overview": "Overview",
    "timeline_single": "Timeline single",
    "timeline_multi": "Timeline multi",
}


def _load_tables(summary_path: Path) -> dict:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload["tables"]


def _pct_delta(local: float | None, ref: float | None) -> float | None:
    if local is None or ref in (None, 0):
        return None
    return (local - ref) / ref * 100.0


def compare(summary_path: Path) -> dict:
    tables = _load_tables(summary_path)
    report: dict = {"summary_path": str(summary_path), "splits": {}}
    for split, nvidia_split in NVIDIA_FULL.items():
        split_report: dict = {"text_following": [], "constraints": []}
        for key, label in ROW_LABELS.items():
            row = next(item for item in tables[split]["text_following"] if item["row"] == label)
            ref = nvidia_split["text_following"][key]
            local = {
                "R@3": row.get("R@3 (gen)"),
                "FID": row.get("FID gen-GT"),
                "Skate": row.get("Skate (gen, cm/s)"),
                "Contact": row.get("Contact (gen)"),
            }
            split_report["text_following"].append(
                {
                    "category": key,
                    "local": local,
                    "nvidia_full": ref,
                    "delta_pct": {
                        metric: _pct_delta(local[metric], ref[metric]) for metric in local
                    },
                }
            )
        for bucket, ref in (
            ("Constraints with text", "constraints_withtext"),
            ("Constraints without text", "constraints_notext"),
        ):
            row = next(item for item in tables[split]["constraints"] if item["row"] == bucket)
            local = {
                "FB": row.get("Full-Body Pos (gen, cm)"),
                "EE": row.get("End-Effector Pos (gen, cm)"),
                "Root2D": row.get("2D Root Pos (gen, cm)"),
                "Pelvis95": row.get("2D Pelvis Pos@95% (gen, cm)"),
            }
            split_report["constraints"].append(
                {
                    "bucket": ref,
                    "local": local,
                    "nvidia_full": nvidia_split[ref],
                    "delta_pct": {
                        metric: _pct_delta(local[metric], nvidia_split[ref][metric])
                        for metric in local
                    },
                }
            )
        report["splits"][split] = split_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_rows", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = compare(args.summary_rows)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
