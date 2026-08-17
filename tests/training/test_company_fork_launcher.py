from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_600K = PROJECT_ROOT / "scripts/company_start_hostnet_fork_600k.sh"
LAUNCHER_695K = PROJECT_ROOT / "scripts/company_start_hostnet_fork_695k.sh"
LAUNCHER_695K_K7 = PROJECT_ROOT / "scripts/company_start_hostnet_fork_695k_k7.sh"
LAUNCHER_690K_K7 = PROJECT_ROOT / "scripts/company_start_hostnet_fork_690k_k7.sh"
LAUNCHER_696K_K7_RESEED = PROJECT_ROOT / "scripts/company_start_hostnet_fork_696k_k7_reseed.sh"
LAUNCHER_650K_WD03 = PROJECT_ROOT / "scripts/company_start_hostnet_fork_650k_wd03.sh"
LAUNCHER_780K_LR3E6 = PROJECT_ROOT / "scripts/company_start_hostnet_fork_780k_lr3e6.sh"
LAUNCHER_750K_LASTWD = PROJECT_ROOT / "scripts/company_start_hostnet_fork_750k_lastwd.sh"


class _ForkLauncherTestBase(unittest.TestCase):
    launcher = LAUNCHER_600K
    parent_checkpoint = "runs/v2-1m-hostnet/checkpoints/step-000600000.pt"
    expected_resume_step = "step-000600000.pt"
    expected_run_dir_token = "v2-1m-hostnet-kf-smooth-lr1e5"
    partial_child = "runs/v2-1m-hostnet-kf-smooth-lr1e5"

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.code_root = self.root / "code"
        self.storage_root = self.root / "storage"
        scripts = self.code_root / "scripts"
        overlay_dir = self.code_root / "configs/overlays"
        scripts.mkdir(parents=True)
        shutil.copy2(PROJECT_ROOT / "scripts/stage_startup_file.py", scripts)
        overlay_dir.mkdir(parents=True)
        (overlay_dir / "v2_1m_from600k_kf_smooth.yaml").write_text(
            "curriculum:\n  sparse_channel_budget: 800\n",
            encoding="utf-8",
        )
        (overlay_dir / "v2_1m_k7_from695k.yaml").write_text(
            "curriculum:\n  sparse_keyframes_hard_cap: 7\n",
            encoding="utf-8",
        )
        (overlay_dir / "v2_1m_k7_reseed696k.yaml").write_text(
            "curriculum:\n  sparse_keyframes_hard_cap: 7\n"
            "runtime:\n  seed: 4321\n  max_steps_override: 698000\n",
            encoding="utf-8",
        )
        (overlay_dir / "v2_1m_wd03_from650k.yaml").write_text(
            "optimizer:\n  weight_decay: 0.3\n  warmup_steps: 2000\n"
            "runtime:\n  reset_optimizer: true\n"
            "curriculum:\n  sparse_keyframes_max: 20\n",
            encoding="utf-8",
        )
        (overlay_dir / "v2_1m_wd03_from780k_lr3e6.yaml").write_text(
            "optimizer:\n  weight_decay: 0.3\n  warmup_steps: 2000\n"
            "  learning_rate: 3.0e-6\n  lr_schedule_start_step: 780000\n"
            "runtime:\n  reset_optimizer: true\n"
            "curriculum:\n  sparse_keyframes_max: 20\n",
            encoding="utf-8",
        )
        (overlay_dir / "v2_1m_lastwd1_from750k.yaml").write_text(
            "optimizer:\n  weight_decay: 0.3\n  last_layer_weight_decay: 1.0\n"
            "  warmup_steps: 2000\n  lr_schedule_start_step: 650000\n"
            "runtime:\n  reset_optimizer: true\n"
            "curriculum:\n  sparse_keyframes_max: 20\n",
            encoding="utf-8",
        )
        training = self.code_root / "kimodo/training"
        training.mkdir(parents=True)
        (training / "config.py").write_text(
            "sparse_channel_budget = 0\nsparse_keyframes_hard_cap = 0\n"
            "warmup_steps = 0\nreset_optimizer = False\n"
            "last_layer_weight_decay = None\n",
            encoding="utf-8",
        )
        (training / "constraints.py").write_text(
            "def _prepare_affordable_budget():\n    return 0\n"
            "def _scheduled_for_sampling():\n    return 0\n",
            encoding="utf-8",
        )
        (training / "optim.py").write_text(
            "def scheduled_learning_rate():\n    return 0\n"
            "def _body_last_layer_parameters():\n    return []\n"
            "track_update_stats = False\n",
            encoding="utf-8",
        )
        (training / "engine.py").write_text(
            "def _weight_decay_for_group():\n    return 0.3\n",
            encoding="utf-8",
        )
        (training / "checkpoint.py").write_text(
            "reset_optimizer = True\n",
            encoding="utf-8",
        )
        self.marker = self.root / "launcher.marker"
        downstream = scripts / "train_company.sh"
        downstream.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'printf "%s\\n" "$KIMODO_NODE_RANK|$MASTER_ADDR|$KIMODO_RUN_DIR|'
            '$KIMODO_LOCAL_MANIFEST_READ_PATH|$KIMODO_FEATURE_CACHE_INDEX_MODE|'
            '$KIMODO_SKIP_MANIFEST_PATH_STAT" '
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
        checkpoint = self.storage_root / self.parent_checkpoint
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
            ["/bin/sh", str(self.launcher)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            timeout=5,
        )
        content = self.marker.read_text(encoding="utf-8")
        self.assertIn("1|master.test|", content)
        self.assertIn("runtime.resume_mode=fork", content)
        self.assertIn(self.expected_resume_step, content)
        self.assertIn(self.expected_run_dir_token, content)
        fields = content.splitlines()[0].split("|")
        self.assertTrue(Path(fields[3]).is_file())
        self.assertEqual(fields[4], "deterministic")
        self.assertEqual(fields[5], "1")
        self.assertTrue(Path(fields[2]).is_dir())
        status = Path(f"{fields[2]}.launch-status/node-1.log")
        self.assertTrue(status.is_file())
        self.assertIn("downstream exec", status.read_text(encoding="utf-8"))

    def test_explicit_kimodo_rank_is_accepted(self) -> None:
        environment = self.environment()
        environment["KIMODO_NODE_RANK"] = "0"
        subprocess.run(
            ["/bin/sh", str(self.launcher)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            timeout=5,
        )
        self.assertTrue(self.marker.read_text(encoding="utf-8").startswith("0|master.test|"))

    def test_missing_rank_fails_before_downstream_launcher(self) -> None:
        result = subprocess.run(
            ["/bin/sh", str(self.launcher)],
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
        child = self.storage_root / self.partial_child
        child.mkdir(parents=True)
        partial = child / "train.jsonl"
        partial.write_text("partial\n", encoding="utf-8")
        environment = self.environment()
        environment["JOB_COMPLETION_INDEX"] = "0"
        result = subprocess.run(
            ["/bin/sh", str(self.launcher)],
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


class CompanyForkLauncherTests(_ForkLauncherTestBase):
    launcher = LAUNCHER_600K


class CompanyFork695kLauncherTests(_ForkLauncherTestBase):
    launcher = LAUNCHER_695K
    parent_checkpoint = (
        "runs/v2-1m-hostnet-kf-smooth-lr1e5/checkpoints/step-000695000.pt"
    )
    expected_resume_step = "step-000695000.pt"
    expected_run_dir_token = "v2-1m-hostnet-cap800-from695k"
    partial_child = "runs/v2-1m-hostnet-cap800-from695k"


class CompanyFork695kK7LauncherTests(_ForkLauncherTestBase):
    launcher = LAUNCHER_695K_K7
    parent_checkpoint = (
        "runs/v2-1m-hostnet-kf-smooth-lr1e5/checkpoints/step-000695000.pt"
    )
    expected_resume_step = "step-000695000.pt"
    expected_run_dir_token = "v2-1m-hostnet-k7-from695k"
    partial_child = "runs/v2-1m-hostnet-k7-from695k"


class CompanyFork690kK7LauncherTests(_ForkLauncherTestBase):
    launcher = LAUNCHER_690K_K7
    parent_checkpoint = (
        "runs/v2-1m-hostnet-kf-smooth-lr1e5/checkpoints/step-000690000.pt"
    )
    expected_resume_step = "step-000690000.pt"
    expected_run_dir_token = "v2-1m-hostnet-k7-from690k"
    partial_child = "runs/v2-1m-hostnet-k7-from690k"


class CompanyFork696kK7ReseedLauncherTests(_ForkLauncherTestBase):
    launcher = LAUNCHER_696K_K7_RESEED
    parent_checkpoint = (
        "runs/v2-1m-hostnet-k7-from690k/checkpoints/step-000696000.pt"
    )
    expected_resume_step = "step-000696000.pt"
    expected_run_dir_token = "v2-1m-hostnet-k7-reseed696k"
    partial_child = "runs/v2-1m-hostnet-k7-reseed696k"


class CompanyFork650kWd03LauncherTests(_ForkLauncherTestBase):
    launcher = LAUNCHER_650K_WD03
    parent_checkpoint = (
        "runs/v2-1m-hostnet-kf-smooth-lr1e5/checkpoints/step-000650000.pt"
    )
    expected_resume_step = "step-000650000.pt"
    expected_run_dir_token = "v2-1m-hostnet-wd03-from650k"
    partial_child = "runs/v2-1m-hostnet-wd03-from650k"


class CompanyFork780kLr3e6LauncherTests(_ForkLauncherTestBase):
    launcher = LAUNCHER_780K_LR3E6
    parent_checkpoint = (
        "runs/v2-1m-hostnet-wd03-from650k/checkpoints/step-000780000.pt"
    )
    expected_resume_step = "step-000780000.pt"
    expected_run_dir_token = "v2-1m-hostnet-wd03-from780k-lr3e6"
    partial_child = "runs/v2-1m-hostnet-wd03-from780k-lr3e6"


class CompanyFork750kLastWdLauncherTests(_ForkLauncherTestBase):
    launcher = LAUNCHER_750K_LASTWD
    parent_checkpoint = (
        "preserved-pre-collapse/v2-1m-hostnet-wd03-from650k/step-000750000.pt"
    )
    expected_resume_step = "step-000750000.pt"
    expected_run_dir_token = "v2-1m-hostnet-lastwd1-from750k"
    partial_child = "runs/v2-1m-hostnet-lastwd1-from750k"


if __name__ == "__main__":
    unittest.main()
