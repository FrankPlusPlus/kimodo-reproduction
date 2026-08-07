# SPDX-License-Identifier: Apache-2.0
"""Watch EMA exports and run the public Kimodo benchmark in a sidecar process."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from kimodo.training.checkpoint import atomic_text_write

_EXPORT_NAME = re.compile(r"^step-(\d{9})$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_inventory_sha256(root: Path) -> str:
    """Fingerprint the immutable inputs used by a benchmark subset."""
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name in {"meta.json", "constraints.json", "gt_motion.npz"}
    )
    if not paths:
        raise ValueError(f"No benchmark inputs found under {root}")
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _benchmark_inventory_stat_signature(root: Path) -> str:
    """Cheaply detect proxy mutations without rereading every large GT NPZ."""
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name in {"meta.json", "constraints.json", "gt_motion.npz"}
    )
    if not paths:
        raise ValueError(f"No benchmark inputs found under {root}")
    for path in paths:
        stat = path.stat()
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def resolve_benchmark_inventory_sha256(root: Path, output_root: Path) -> str:
    """Hash immutable proxy contents once, then verify a lightweight signature."""
    cache_path = output_root / "benchmark_inventory.json"
    stat_signature = _benchmark_inventory_stat_signature(root)
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("root") != str(root) or cached.get("stat_signature") != stat_signature:
            raise RuntimeError(
                "Benchmark proxy changed after monitoring started; use a new output root"
            )
        return str(cached["sha256"])
    content_hash = benchmark_inventory_sha256(root)
    atomic_text_write(
        json.dumps(
            {
                "root": str(root),
                "stat_signature": stat_signature,
                "sha256": content_hash,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        cache_path,
    )
    return content_hash


def discover_exports(run_dir: Path, minimum_step: int = 0) -> list[tuple[int, Path]]:
    exports = run_dir / "exports"
    if not exports.is_dir():
        return []
    found = []
    for path in exports.iterdir():
        match = _EXPORT_NAME.fullmatch(path.name)
        if not match or not path.is_dir():
            continue
        step = int(match.group(1))
        if step < minimum_step:
            continue
        required = (path / "model.pt", path / "config.yaml", path / "stats")
        if all(item.exists() for item in required):
            found.append((step, path))
    return sorted(found)


def _flatten_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            output.update(_flatten_numbers(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            output.update(_flatten_numbers(item, f"{prefix}/{index}"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            output[prefix] = number
    return output


def _direction(name: str) -> str | None:
    lowered = name.lower()
    if any(token in lowered for token in ("fid", "skate", "error", "err", "medr", "pos (gen")):
        return "lower"
    if any(token in lowered for token in ("r@", "r0", "contact", "accuracy", "acc", "sim")):
        return "higher"
    return None


def _significant_worsening(name: str, current: float, reference: float) -> bool:
    direction = _direction(name)
    if direction is None:
        return False
    lowered = name.lower()
    if "r@" in lowered or "/r0" in lowered:
        margin = 2.0
    elif "contact" in lowered:
        margin = 0.005
    elif "cm" in lowered:
        margin = max(0.5, abs(reference) * 0.10)
    else:
        margin = abs(reference) * 0.10
    return current > reference + margin if direction == "lower" else current < reference - margin


def trend_alerts(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Warn only after the same metric worsens across two consecutive intervals."""
    if len(summaries) < 3:
        return []
    older = _flatten_numbers(summaries[-3].get("summary", {}))
    previous = _flatten_numbers(summaries[-2].get("summary", {}))
    current = _flatten_numbers(summaries[-1].get("summary", {}))
    alerts = []
    for name in sorted(current.keys() & previous.keys() & older.keys()):
        if _significant_worsening(name, previous[name], older[name]) and _significant_worsening(
            name, current[name], previous[name]
        ):
            alerts.append(
                {
                    "metric": name,
                    "values": [older[name], previous[name], current[name]],
                    "reason": "significant worsening in two consecutive benchmark intervals",
                }
            )
    return alerts


def _run(command: list[str], *, cwd: Path, log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def _load_completed(output_root: Path) -> list[dict[str, Any]]:
    completed = []
    for path in sorted(output_root.glob("step-*/complete.json")):
        completed.append(json.loads(path.read_text(encoding="utf-8")))
    return completed


def evaluate_export(args: argparse.Namespace, step: int, bundle: Path) -> Path | None:
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / f"step-{step:09d}"
    if (final / "complete.json").is_file():
        return final

    claims = output_root / ".claims"
    claims.mkdir(exist_ok=True)
    claim = claims / f"step-{step:09d}"
    try:
        claim.mkdir()
    except FileExistsError:
        return None

    building = output_root / f".step-{step:09d}.building-{os.getpid()}"
    if building.exists():
        raise FileExistsError(building)
    building.mkdir()
    generated = building / "generated"
    log_path = building / "evaluation.log"
    project_root = Path(__file__).resolve().parents[2]
    python = args.python
    benchmark_root = Path(args.benchmark).expanduser().resolve()
    try:
        generate = [
            python,
            "benchmark/generate_eval.py",
            "--benchmark",
            str(benchmark_root),
            "--output",
            str(generated),
            "--checkpoint-bundle",
            str(bundle),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--diffusion_steps",
            str(args.diffusion_steps),
        ]
        if args.text_encoder_fp32:
            generate.append("--text_encoder_fp32")
        _run(generate, cwd=project_root, log_path=log_path)

        embed = [
            python,
            "benchmark/embed_folder.py",
            str(generated),
            "--device",
            args.device,
        ]
        if args.text_encoder_fp32:
            embed.append("--text_encoder_fp32")
        _run(embed, cwd=project_root, log_path=log_path)

        evaluate = [
            python,
            "benchmark/evaluate_folder.py",
            str(generated),
            "--device",
            args.device,
        ]
        if args.paper_protocol:
            evaluate.append("--paper-protocol")
        _run(evaluate, cwd=project_root, log_path=log_path)

        summary_path = building / "summary_rows.json"
        _run(
            [
                python,
                "benchmark/parse_folder.py",
                str(generated),
                "--output",
                str(summary_path),
            ],
            cwd=project_root,
            log_path=log_path,
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        record: dict[str, Any] = {
            "schema_version": 1,
            "event": "kimodo_public_benchmark_monitor",
            "step": step,
            "bundle": str(bundle),
            "bundle_model_sha256": _sha256(bundle / "model.pt"),
            "benchmark": str(benchmark_root),
            "benchmark_inventory_sha256": resolve_benchmark_inventory_sha256(
                benchmark_root, output_root
            ),
            "diffusion_steps": args.diffusion_steps,
            "generation_batch_size": args.batch_size,
            "text_encoder_precision": "fp32" if args.text_encoder_fp32 else "bf16",
            "paper_protocol": bool(args.paper_protocol),
            "summary": summary,
        }
        if args.baseline_summary:
            baseline_path = Path(args.baseline_summary).expanduser().resolve()
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            current_values = _flatten_numbers(summary)
            baseline_values = _flatten_numbers(baseline)
            record["official_baseline"] = {
                "path": str(baseline_path),
                "sha256": _sha256(baseline_path),
                "deltas": {
                    name: current_values[name] - baseline_values[name]
                    for name in sorted(current_values.keys() & baseline_values.keys())
                    if _direction(name) is not None
                },
            }
        prior = _load_completed(output_root)
        record["alerts"] = trend_alerts([*prior, record])
        atomic_text_write(json.dumps(record, indent=2, sort_keys=True) + "\n", building / "complete.json")
        os.replace(building, final)
        history_record = {
            key: record[key]
            for key in (
                "schema_version",
                "event",
                "step",
                "bundle_model_sha256",
                "benchmark_inventory_sha256",
                "alerts",
            )
        }
        history_record["metrics"] = {
            name: value
            for name, value in sorted(_flatten_numbers(summary).items())
            if _direction(name) is not None
        }
        with (output_root / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history_record, sort_keys=True) + "\n")
        atomic_text_write(
            json.dumps(history_record, indent=2, sort_keys=True) + "\n",
            output_root / "latest.json",
        )
        return final
    except Exception as error:
        failure = {
            "step": step,
            "bundle": str(bundle),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        atomic_text_write(json.dumps(failure, indent=2) + "\n", building / "failed.json")
        failed = output_root / f"failed-step-{step:09d}-{int(time.time())}"
        os.replace(building, failed)
        raise
    finally:
        shutil.rmtree(claim, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Training run containing exports/step-*")
    parser.add_argument("--benchmark", required=True, help="Fixed public benchmark proxy tree")
    parser.add_argument("--output-root", required=True, help="Independent evaluation output directory")
    parser.add_argument("--baseline-summary", help="Official SEED model summary_rows.json on this proxy")
    parser.add_argument("--minimum-step", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--text-encoder-fp32", action="store_true")
    parser.add_argument("--paper-protocol", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.poll_seconds < 1 or args.batch_size < 1 or args.diffusion_steps < 1:
        raise SystemExit("poll-seconds, batch-size and diffusion-steps must be positive")
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    while True:
        completed_steps = {
            int(path.parent.name.removeprefix("step-"))
            for path in output_root.glob("step-*/complete.json")
        }
        for step, bundle in discover_exports(run_dir, args.minimum_step):
            if step not in completed_steps:
                result = evaluate_export(args, step, bundle)
                if result is not None:
                    print(
                        json.dumps(
                            {"event": "benchmark_complete", "step": step, "path": str(result)}
                        ),
                        flush=True,
                    )
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
