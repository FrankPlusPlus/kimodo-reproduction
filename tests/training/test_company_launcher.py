from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "scripts/train_company.sh"


def _environment(tmp_path: Path, *, pods: int, gpus_per_pod: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "KIMODO_PYTHON": "/bin/true",
            "KIMODO_PATHS_CONFIG": str(
                PROJECT_ROOT / "configs/training/kimodo_soma_seed_v2_1m_16h200.yaml"
            ),
            "KIMODO_RUN_DIR": str(tmp_path / "run"),
            "KIMODO_AUTO_RESUME": "0",
            "KIMODO_REQUIRE_RDMA": "0",
            "KIMODO_NNODES": str(pods),
            "KIMODO_NPROC_PER_NODE": str(gpus_per_pod),
            "KIMODO_NODE_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
        }
    )
    return environment


def test_company_launcher_accepts_multiple_sixteen_rank_pod_topologies(tmp_path):
    for pods, gpus_per_pod in ((2, 8), (4, 4), (16, 1)):
        subprocess.run(
            [str(LAUNCHER)],
            cwd=PROJECT_ROOT,
            env=_environment(tmp_path / f"{pods}x{gpus_per_pod}", pods=pods, gpus_per_pod=gpus_per_pod),
            check=True,
            timeout=10,
        )


def test_company_launcher_defaults_to_the_extracted_portable_v2_bundle(tmp_path):
    storage_root = tmp_path / "storage"
    data_root = storage_root / "benchmark-v2-soma30-v2.2"
    data_root.mkdir(parents=True)
    (data_root / "repro.paths.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    environment = _environment(tmp_path, pods=2, gpus_per_pod=8)
    environment.pop("KIMODO_PATHS_CONFIG")
    environment["KIMODO_STORAGE_ROOT"] = str(storage_root)
    subprocess.run(
        [str(LAUNCHER)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        timeout=10,
    )


def test_company_launcher_rejects_topology_that_misses_expected_world_size(tmp_path):
    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=PROJECT_ROOT,
        env=_environment(tmp_path, pods=2, gpus_per_pod=4),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "expected 16 total ranks" in result.stderr


def test_company_launcher_accepts_non_sixteen_when_env_contract_matches(tmp_path):
    environment = _environment(tmp_path, pods=2, gpus_per_pod=3)
    environment["KIMODO_EXPECTED_WORLD_SIZE"] = "6"
    environment["KIMODO_BATCH_SIZE"] = "8"
    environment["KIMODO_MAX_STEPS"] = "5"
    subprocess.run(
        [str(LAUNCHER)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        timeout=10,
    )


def test_company_launcher_accepts_explicit_v1_or_v2_config_with_separate_run_dir(tmp_path):
    environment = _environment(tmp_path, pods=2, gpus_per_pod=8)
    environment["KIMODO_TRAINING_CONFIG"] = str(
        PROJECT_ROOT / "configs/training/kimodo_soma_seed_reproduction.yaml"
    )
    subprocess.run(
        [str(LAUNCHER)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        timeout=10,
    )


def test_company_launcher_requires_separate_run_dir_for_nondefault_config(tmp_path):
    environment = _environment(tmp_path, pods=2, gpus_per_pod=8)
    environment.pop("KIMODO_RUN_DIR")
    environment["KIMODO_TRAINING_CONFIG"] = str(
        PROJECT_ROOT / "configs/training/kimodo_soma_seed_reproduction.yaml"
    )
    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "KIMODO_RUN_DIR is required" in result.stderr


def test_company_launcher_auto_resumes_latest_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "step-000010000.pt"
    checkpoint.touch()
    (checkpoint_dir / "latest.txt").write_text(checkpoint.name + "\n", encoding="utf-8")
    environment = _environment(tmp_path, pods=2, gpus_per_pod=8)
    environment["KIMODO_AUTO_RESUME"] = "1"

    subprocess.run(
        [str(LAUNCHER)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        timeout=10,
    )


def test_company_launcher_treats_the_final_checkpoint_as_complete(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "step-001000000.pt"
    checkpoint.touch()
    (checkpoint_dir / "latest.txt").write_text(checkpoint.name + "\n", encoding="utf-8")
    final_bundle = run_dir / "exports/step-001000000"
    (final_bundle / "stats").mkdir(parents=True)
    (final_bundle / "model.pt").touch()
    (final_bundle / "config.yaml").touch()
    environment = _environment(tmp_path, pods=4, gpus_per_pod=4)
    environment["KIMODO_AUTO_RESUME"] = "1"

    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "already complete" in result.stderr
