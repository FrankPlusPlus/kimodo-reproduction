from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "scripts/company_run_body_jacobian_probe.sh"


class BodyJacobianLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.code_root = self.root / "code"
        self.storage_root = self.root / "storage"
        scripts = self.code_root / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(LAUNCHER, scripts / LAUNCHER.name)
        (scripts / "diagnose_body_jacobian.py").write_text("print('probe')\n", encoding="utf-8")
        kf = self.storage_root / "runs/v2-1m-hostnet-kf-smooth-lr1e5"
        k7 = self.storage_root / "runs/v2-1m-hostnet-k7-from690k"
        (kf / "checkpoints").mkdir(parents=True)
        (k7 / "checkpoints").mkdir(parents=True)
        (kf / "config.resolved.yaml").write_text("schema_version: 1\n", encoding="utf-8")
        (kf / "checkpoints/step-000650000.pt").write_bytes(b"x")
        (kf / "checkpoints/step-000690000.pt").write_bytes(b"x")
        (k7 / "checkpoints/step-000696000.pt").write_bytes(b"x")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _env(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            {
                "KIMODO_CODE_ROOT": str(self.code_root),
                "KIMODO_STORAGE_ROOT": str(self.storage_root),
                "KIMODO_PYTHON": "true",
            }
        )
        return environment

    def test_nonzero_rank_skips_without_running_python(self) -> None:
        environment = self._env()
        environment["JOB_COMPLETION_INDEX"] = "1"
        result = subprocess.run(
            ["/bin/bash", str(self.code_root / "scripts" / LAUNCHER.name)],
            cwd=self.code_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("skipping node rank 1", result.stdout)

    def test_rank0_requires_checkpoints(self) -> None:
        environment = self._env()
        environment["JOB_COMPLETION_INDEX"] = "0"
        (self.storage_root / "runs/v2-1m-hostnet-k7-from690k/checkpoints/step-000696000.pt").unlink()
        result = subprocess.run(
            ["/bin/bash", str(self.code_root / "scripts" / LAUNCHER.name)],
            cwd=self.code_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing readable path", result.stderr)


if __name__ == "__main__":
    unittest.main()
