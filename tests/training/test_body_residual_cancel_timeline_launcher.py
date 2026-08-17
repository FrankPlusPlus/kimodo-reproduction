from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "scripts/company_run_body_residual_cancel_probe_wd03_timeline.sh"


class BodyResidualCancelTimelineLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.code_root = self.root / "code"
        self.storage_root = self.root / "storage"
        scripts = self.code_root / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(LAUNCHER, scripts / LAUNCHER.name)
        (scripts / "diagnose_body_residual_cancel.py").write_text(
            "# constraint-steps\nprint('probe')\n",
            encoding="utf-8",
        )
        training = self.code_root / "kimodo/training"
        training.mkdir(parents=True)
        (training / "body_residual_cancel_probe.py").write_text(
            "def residual_timeline():\n    return []\n",
            encoding="utf-8",
        )
        kf = self.storage_root / "runs/v2-1m-hostnet-kf-smooth-lr1e5"
        parent = self.storage_root / "runs/v2-1m-hostnet-wd03-from650k"
        hostnet = self.storage_root / "runs/v2-1m-hostnet"
        (kf / "checkpoints").mkdir(parents=True)
        (parent / "checkpoints").mkdir(parents=True)
        (hostnet / "checkpoints").mkdir(parents=True)
        (parent / "config.resolved.yaml").write_text("schema_version: 1\n", encoding="utf-8")
        (kf / "checkpoints/step-000650000.pt").write_bytes(b"x")
        (hostnet / "checkpoints/step-000400000.pt").write_bytes(b"x")
        (hostnet / "checkpoints/step-000500000.pt").write_bytes(b"x")
        for step in (700000, 750000, 780000, 790000):
            (parent / "checkpoints" / f"step-{step:09d}.pt").write_bytes(b"x")

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

    def test_passes_phase1_and_phase2_checkpoints_and_two_clocks(self) -> None:
        diagnose = self.code_root / "scripts/diagnose_body_residual_cancel.py"
        diagnose.write_text(
            "import sys\nprint('diagnose', ' '.join(sys.argv[1:]))\n# constraint-steps\n",
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
        self.assertIn("step-000400000.pt", result.stdout)
        self.assertIn("step-000500000.pt", result.stdout)
        self.assertIn("step-000650000.pt", result.stdout)
        self.assertIn("step-000700000.pt", result.stdout)
        self.assertIn("step-000750000.pt", result.stdout)
        self.assertIn("step-000780000.pt", result.stdout)
        self.assertIn("step-000790000.pt", result.stdout)
        self.assertIn("--constraint-steps 400000 750000", result.stdout)
        self.assertIn("body-residual-cancel-wd03-timeline", result.stdout)

    def test_falls_back_to_preserved_790k(self) -> None:
        (
            self.storage_root
            / "runs/v2-1m-hostnet-wd03-from650k/checkpoints/step-000790000.pt"
        ).unlink()
        preserved = self.storage_root / "preserved-pre-collapse/v2-1m-hostnet-wd03-from650k"
        preserved.mkdir(parents=True)
        (preserved / "step-000790000.pt").write_bytes(b"x")
        diagnose = self.code_root / "scripts/diagnose_body_residual_cancel.py"
        diagnose.write_text(
            "import sys\nprint('diagnose', ' '.join(sys.argv[1:]))\n# constraint-steps\n",
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
        self.assertIn("preserved-pre-collapse", result.stdout)


if __name__ == "__main__":
    unittest.main()
