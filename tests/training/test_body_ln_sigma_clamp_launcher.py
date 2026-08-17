from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "scripts/company_run_body_ln_sigma_clamp_probe_wd03.sh"


class BodyLnSigmaClampLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.code_root = self.root / "code"
        self.storage_root = self.root / "storage"
        scripts = self.code_root / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(LAUNCHER, scripts / LAUNCHER.name)
        (scripts / "diagnose_body_ln_sigma_clamp.py").write_text(
            "# takeoff-step\nprint('probe')\n",
            encoding="utf-8",
        )
        training = self.code_root / "kimodo/training"
        training.mkdir(parents=True)
        (training / "body_ln_sigma_clamp_probe.py").write_text(
            "def summarize_sigma_clamp():\n    return {}\n",
            encoding="utf-8",
        )
        parent = self.storage_root / "runs/v2-1m-hostnet-wd03-from650k"
        rescue = self.storage_root / "runs/v2-1m-hostnet-wd03-from780k-lr3e6"
        (parent / "checkpoints").mkdir(parents=True)
        (rescue / "checkpoints").mkdir(parents=True)
        (parent / "config.resolved.yaml").write_text("schema_version: 1\n", encoding="utf-8")
        (parent / "checkpoints/step-000750000.pt").write_bytes(b"x")
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

    def test_passes_750_795_800_and_takeoff_step(self) -> None:
        diagnose = self.code_root / "scripts/diagnose_body_ln_sigma_clamp.py"
        diagnose.write_text(
            "import sys\nprint('diagnose', ' '.join(sys.argv[1:]))\n# takeoff-step\n",
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
        self.assertIn("step-000750000.pt", result.stdout)
        self.assertIn("step-000795000.pt", result.stdout)
        self.assertIn("step-000800000.pt", result.stdout)
        self.assertIn("--takeoff-step 800000", result.stdout)
        self.assertIn("body-ln-sigma-clamp-wd03", result.stdout)


if __name__ == "__main__":
    unittest.main()
