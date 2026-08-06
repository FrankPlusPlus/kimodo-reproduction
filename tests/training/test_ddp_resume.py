from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_ddp_resume(training_fixture, output: Path, mode: str | None = None):
    worker = Path(__file__).with_name("_ddp_resume_worker.py")
    environment = dict(os.environ)
    # The DDP worker is intentionally self-contained and must not inherit a
    # heavyweight official-checkpoint gate from the parent pytest invocation.
    environment.pop("KIMODO_OFFICIAL_BUNDLE", None)
    environment["OMP_NUM_THREADS"] = "1"
    if sys.platform == "darwin":
        environment.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    command = [
        sys.executable,
        str(worker),
        str(training_fixture["root"]),
        str(output),
    ]
    if mode is not None:
        command.append(mode)
    subprocess.run(
        command,
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        timeout=120,
    )


def test_two_rank_cpu_ddp_checkpoint_exact_resume(training_fixture, tmp_path):
    _run_ddp_resume(training_fixture, tmp_path / "ddp")


def test_two_rank_cpu_ddp_benchmark_lane_exact_resume(training_fixture, tmp_path):
    _run_ddp_resume(training_fixture, tmp_path / "ddp-benchmark", "benchmark")
