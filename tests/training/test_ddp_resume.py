from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_two_rank_cpu_ddp_checkpoint_exact_resume(training_fixture, tmp_path):
    worker = Path(__file__).with_name("_ddp_resume_worker.py")
    environment = dict(os.environ)
    # The DDP worker is intentionally self-contained and must not inherit a
    # heavyweight official-checkpoint gate from the parent pytest invocation.
    environment.pop("KIMODO_OFFICIAL_BUNDLE", None)
    environment["OMP_NUM_THREADS"] = "1"
    if sys.platform == "darwin":
        environment.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    subprocess.run(
        [
            sys.executable,
            str(worker),
            str(training_fixture["root"]),
            str(tmp_path / "ddp"),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        timeout=120,
    )
