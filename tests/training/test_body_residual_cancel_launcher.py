from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "scripts/company_run_body_residual_cancel_probe_wd03_795_800.sh"


class BodyResidualCancelLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.code_root = self.root / "code"
        self.storage_root = self.root / "storage"
        scripts = self.code_root / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(LAUNCHER, scripts / LAUNCHER.name)
        (scripts / "diagnose_body_residual_cancel.py").write_text(
            "# constraint-step\nprint('probe')\n",
            encoding="utf-8",
        )
        training = self.code_root / "kimodo/training"
        training.mkdir(parents=True)
        (training / "body_residual_cancel_probe.py").write_text(
            "def summarize_residual_cancel():\n    return {}\n",
            encoding="utf-8",
        )
        rescue = self.storage_root / "runs/v2-1m-hostnet-wd03-from780k-lr3e6"
        (rescue / "checkpoints").mkdir(parents=True)
        (rescue / "config.resolved.yaml").write_text("schema_version: 1\n", encoding="utf-8")
        (rescue / "checkpoints/step-000795000.pt").write_bytes(b"x")
        (rescue / "checkpoints/step-000800000.pt").write_bytes(b"x")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _env(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            {
                "KIMODO_CODE_ROOT": str(self.code_root),
                "KIMODO_STORAGE_ROOT": str(self.storage_root),
                "KIMODO_PYTHON": "python3",
                "KIMODO_PROBE_SKIP_TORCHRUN": "1",
            }
        )
        return environment

    def test_skip_torchrun_passes_two_checkpoints(self) -> None:
        diagnose = self.code_root / "scripts/diagnose_body_residual_cancel.py"
        diagnose.write_text(
            "import sys\nprint('diagnose', ' '.join(sys.argv[1:]))\n# constraint-step\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            ["/bin/bash", str(self.code_root / "scripts" / LAUNCHER.name)],
            cwd=self.code_root,
            env=self._env(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("step-000795000.pt", result.stdout)
        self.assertIn("step-000800000.pt", result.stdout)
        self.assertIn("body-residual-cancel-wd03-795-800", result.stdout)
        self.assertIn("--constraint-step 795000", result.stdout)

    def test_stale_module_fails(self) -> None:
        (self.code_root / "kimodo/training/body_residual_cancel_probe.py").write_text("x = 1\n")
        result = subprocess.run(
            ["/bin/bash", str(self.code_root / "scripts" / LAUNCHER.name)],
            cwd=self.code_root,
            env=self._env(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("summarize_residual_cancel", result.stderr)


if __name__ == "__main__":
    unittest.main()
