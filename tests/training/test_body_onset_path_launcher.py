from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "scripts/company_run_body_onset_path_probe_kfsmooth.sh"


class BodyOnsetPathLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.code_root = self.root / "code"
        self.storage_root = self.root / "storage"
        scripts = self.code_root / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(LAUNCHER, scripts / LAUNCHER.name)
        (scripts / "diagnose_body_onset_path.py").write_text(
            "# virtual-steps\nprint('probe')\n",
            encoding="utf-8",
        )
        training = self.code_root / "kimodo/training"
        training.mkdir(parents=True)
        (training / "body_onset_path_probe.py").write_text(
            "def summarize_onset_path():\n    return {}\n",
            encoding="utf-8",
        )
        kf = self.storage_root / "runs/v2-1m-hostnet-kf-smooth-lr1e5"
        (kf / "checkpoints").mkdir(parents=True)
        (kf / "config.resolved.yaml").write_text("schema_version: 1\n", encoding="utf-8")
        (kf / "checkpoints/step-000650000.pt").write_bytes(b"x")
        (kf / "checkpoints/step-000690000.pt").write_bytes(b"x")
        (kf / "checkpoints/step-000695000.pt").write_bytes(b"x")

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

    def test_passes_preflip_checkpoints_and_virtual_steps(self) -> None:
        diagnose = self.code_root / "scripts/diagnose_body_onset_path.py"
        diagnose.write_text(
            "import sys\nprint('diagnose', ' '.join(sys.argv[1:]))\n# virtual-steps\n",
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
        self.assertIn("step-000650000.pt", result.stdout)
        self.assertIn("step-000690000.pt", result.stdout)
        self.assertIn("step-000695000.pt", result.stdout)
        self.assertIn("--preflip-step 690000", result.stdout)
        self.assertIn("--virtual-steps 20", result.stdout)
        self.assertIn("body-onset-path-kfsmooth-690-695", result.stdout)
        self.assertNotIn("wd03-from650k", result.stdout)

    def test_stale_module_fails(self) -> None:
        (self.code_root / "kimodo/training/body_onset_path_probe.py").write_text("x = 1\n")
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
        self.assertIn("summarize_onset_path", result.stderr)


if __name__ == "__main__":
    unittest.main()
