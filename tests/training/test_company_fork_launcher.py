from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "scripts/company_start_hostnet_fork_600k.sh"


class CompanyForkLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.code_root = self.root / "code"
        self.storage_root = self.root / "storage"
        scripts = self.code_root / "scripts"
        overlay = self.code_root / "configs/overlays/v2_1m_from600k_kf_smooth.yaml"
        scripts.mkdir(parents=True)
        shutil.copy2(PROJECT_ROOT / "scripts/stage_startup_file.py", scripts)
        overlay.parent.mkdir(parents=True)
        overlay.write_text("optimizer:\n  learning_rate: 1.0e-5\n", encoding="utf-8")
        self.marker = self.root / "launcher.marker"
        downstream = scripts / "train_company.sh"
        downstream.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'printf "%s\\n" "$KIMODO_NODE_RANK|$MASTER_ADDR|$KIMODO_RUN_DIR|'
            '$KIMODO_LOCAL_MANIFEST_READ_PATH|$KIMODO_LOCAL_FEATURE_INDEX_READ_PATH" '
            '> "$FORK_TEST_MARKER"\n'
            'printf "%s\\n" "$@" >> "$FORK_TEST_MARKER"\n',
            encoding="utf-8",
        )
        downstream.chmod(0o755)
        cache = self.storage_root / "feature-cache/v1"
        cache.mkdir(parents=True)
        (cache / "meta.json").touch()
        (cache / "index.jsonl").touch()
        manifest = self.storage_root / "benchmark-v2-soma30-v2.2/train.cached.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.touch()
        checkpoint = (
            self.storage_root
            / "runs/v2-1m-hostnet/checkpoints/step-000600000.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        for name in (
            "KIMODO_NODE_RANK",
            "PET_NODE_RANK",
            "NODE_RANK",
            "JOB_COMPLETION_INDEX",
            "PET_MASTER_ADDR",
            "NNODES",
            "PET_NNODES",
        ):
            environment.pop(name, None)
        environment.update(
            {
                "KIMODO_CODE_ROOT": str(self.code_root),
                "KIMODO_STORAGE_ROOT": str(self.storage_root),
                "KIMODO_NNODES": "2",
                "KIMODO_NPROC_PER_NODE": "8",
                "MASTER_ADDR": "master.test",
                "FORK_TEST_MARKER": str(self.marker),
                "KIMODO_STARTUP_CACHE_ROOT": str(self.root / "startup-cache"),
                "KIMODO_PYTHON": sys.executable,
            }
        )
        return environment

    def test_posix_entry_accepts_job_completion_index(self) -> None:
        environment = self.environment()
        environment["JOB_COMPLETION_INDEX"] = "1"
        subprocess.run(
            ["/bin/sh", str(LAUNCHER)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            timeout=5,
        )
        content = self.marker.read_text(encoding="utf-8")
        self.assertIn("1|master.test|", content)
        self.assertIn("runtime.resume_mode=fork", content)
        self.assertIn("step-000600000.pt", content)
        fields = content.splitlines()[0].split("|")
        self.assertTrue(Path(fields[3]).is_file())
        self.assertTrue(Path(fields[4]).is_file())
        self.assertTrue(Path(fields[2]).is_dir())

    def test_explicit_kimodo_rank_is_accepted(self) -> None:
        environment = self.environment()
        environment["KIMODO_NODE_RANK"] = "0"
        subprocess.run(
            ["/bin/sh", str(LAUNCHER)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            timeout=5,
        )
        self.assertTrue(self.marker.read_text(encoding="utf-8").startswith("0|master.test|"))

    def test_missing_rank_fails_before_downstream_launcher(self) -> None:
        result = subprocess.run(
            ["/bin/sh", str(LAUNCHER)],
            cwd=PROJECT_ROOT,
            env=self.environment(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("No node rank was injected", result.stderr)
        self.assertFalse(self.marker.exists())

    def test_partial_child_run_fails_without_deleting_it(self) -> None:
        child = self.storage_root / "runs/v2-1m-hostnet-kf-smooth-lr1e5"
        child.mkdir(parents=True)
        partial = child / "train.jsonl"
        partial.write_text("partial\n", encoding="utf-8")
        environment = self.environment()
        environment["JOB_COMPLETION_INDEX"] = "0"
        result = subprocess.run(
            ["/bin/sh", str(LAUNCHER)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("child run exists but has no checkpoint", result.stderr)
        self.assertEqual(partial.read_text(encoding="utf-8"), "partial\n")


if __name__ == "__main__":
    unittest.main()
