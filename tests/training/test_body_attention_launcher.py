from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "scripts/company_run_body_attention_probe_wd03_795_800.sh"


class BodyAttentionLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.code_root = self.root / "code"
        self.storage_root = self.root / "storage"
        scripts = self.code_root / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(LAUNCHER, scripts / LAUNCHER.name)
        (scripts / "diagnose_body_attention.py").write_text(
            "# constraint-steps\nprint('probe')\n",
            encoding="utf-8",
        )
        training = self.code_root / "kimodo/training"
        training.mkdir(parents=True)
        (training / "body_attention_probe.py").write_text(
            "def summarize_pointer_grid():\n    return {}\n",
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
                "KIMODO_PYTHON": "true",
            }
        )
        return environment

    def test_single_gpu_skips_master_addr(self) -> None:
        environment = self._env()
        environment["JOB_COMPLETION_INDEX"] = "0"
        environment["KIMODO_NNODES"] = "1"
        environment["KIMODO_NPROC_PER_NODE"] = "1"
        environment["KIMODO_PYTHON"] = "python3"
        environment.pop("MASTER_ADDR", None)
        environment.pop("PET_MASTER_ADDR", None)
        diagnose = self.code_root / "scripts/diagnose_body_attention.py"
        diagnose.write_text(
            "import sys\n"
            "print('diagnose', ' '.join(sys.argv[1:]))\n"
            "# constraint-steps\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            ["/bin/bash", str(self.code_root / "scripts" / LAUNCHER.name)],
            cwd=self.code_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--constraint-steps 795000 800000", result.stdout)
        self.assertIn("step-000795000.pt", result.stdout)
        self.assertIn("step-000800000.pt", result.stdout)
        self.assertIn("--output-dir", result.stdout)
        self.assertIn("body-attention-wd03-795-800", result.stdout)

    def test_requires_master_addr_for_two_nodes(self) -> None:
        environment = self._env()
        environment["JOB_COMPLETION_INDEX"] = "0"
        environment["KIMODO_NNODES"] = "2"
        environment["KIMODO_NPROC_PER_NODE"] = "8"
        environment.pop("MASTER_ADDR", None)
        environment.pop("PET_MASTER_ADDR", None)
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
        self.assertIn("MASTER_ADDR", result.stderr)

    def test_falls_back_to_preserved_800k(self) -> None:
        environment = self._env()
        environment["KIMODO_PROBE_SKIP_TORCHRUN"] = "1"
        environment["JOB_COMPLETION_INDEX"] = "0"
        environment["KIMODO_PYTHON"] = "python3"
        (
            self.storage_root
            / "runs/v2-1m-hostnet-wd03-from780k-lr3e6/checkpoints/step-000800000.pt"
        ).unlink()
        preserved = (
            self.storage_root
            / "preserved-pre-collapse/v2-1m-hostnet-wd03-from780k-lr3e6"
        )
        preserved.mkdir(parents=True)
        (preserved / "step-000800000.pt").write_bytes(b"x")
        diagnose = self.code_root / "scripts/diagnose_body_attention.py"
        diagnose.write_text(
            "import sys\n"
            "print('diagnose', ' '.join(sys.argv[1:]))\n"
            "# constraint-steps\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            ["/bin/bash", str(self.code_root / "scripts" / LAUNCHER.name)],
            cwd=self.code_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("preserved-pre-collapse", result.stdout)

    def test_optional_layer_filter_is_forwarded(self) -> None:
        environment = self._env()
        environment["KIMODO_PROBE_SKIP_TORCHRUN"] = "1"
        environment["KIMODO_PYTHON"] = "python3"
        environment["KIMODO_ATTENTION_LAYERS"] = "0 7 14 15"
        diagnose = self.code_root / "scripts/diagnose_body_attention.py"
        diagnose.write_text(
            "import sys\n"
            "print('diagnose', ' '.join(sys.argv[1:]))\n"
            "# constraint-steps\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            ["/bin/bash", str(self.code_root / "scripts" / LAUNCHER.name)],
            cwd=self.code_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--layers 0 7 14 15", result.stdout)

    def test_stale_probe_module_fails(self) -> None:
        environment = self._env()
        environment["KIMODO_PROBE_SKIP_TORCHRUN"] = "1"
        (self.code_root / "kimodo/training/body_attention_probe.py").write_text(
            "def other():\n    return 1\n",
            encoding="utf-8",
        )
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
        self.assertIn("summarize_pointer_grid", result.stderr)


if __name__ == "__main__":
    unittest.main()
