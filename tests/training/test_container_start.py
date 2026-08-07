from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts/container_start.sh"
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"


def test_dockerfile_uses_hardware_neutral_dispatcher():
    source = DOCKERFILE.read_text(encoding="utf-8")
    assert 'CMD ["/workspace/scripts/container_start.sh"]' in source
    assert 'CMD ["/workspace/scripts/train_company_16h200.sh"]' not in source
    assert "KIMODO_PYTHON=python /workspace/scripts/smoke_train.sh" in source


def test_dockerfile_uses_generic_storage_root_and_can_extract_v2_archive():
    source = DOCKERFILE.read_text(encoding="utf-8")
    namespace = "/mnt/kimodo"
    assert f"KIMODO_STORAGE_ROOT={namespace}" in source
    assert "KIMODO_DATA_ROOT=" not in source
    assert "KIMODO_RUN_ROOT=" not in source
    assert "/home/share/" not in source
    assert "      zstd \\\n" in source


def test_image_keeps_a_writable_git_worktree_for_pod_updates():
    source = DOCKERFILE.read_text(encoding="utf-8")
    ignored = {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert ".git" not in ignored
    assert ".github" not in ignored
    assert "COPY . /workspace" in source
    assert "git config --system --add safe.directory /workspace" in source
    assert "chmod -R a+rwX /workspace" in source


def test_container_dispatcher_help_lists_reviewed_modes():
    environment = dict(os.environ)
    environment["KIMODO_CONTAINER_MODE"] = "help"
    result = subprocess.run(
        [str(RUNNER)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    for mode in ("idle", "train-company", "prepare", "preflight", "eval-watch"):
        assert mode in result.stdout


def test_container_dispatcher_rejects_unknown_mode():
    environment = dict(os.environ)
    environment["KIMODO_CONTAINER_MODE"] = "surprise-training"
    result = subprocess.run(
        [str(RUNNER)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert "unknown KIMODO_CONTAINER_MODE" in result.stderr


def test_container_preflight_defaults_to_the_extracted_portable_v2_bundle(tmp_path):
    storage_root = tmp_path / "storage"
    data_root = storage_root / "benchmark-v2-soma30-v2.2"
    data_root.mkdir(parents=True)
    (data_root / "repro.paths.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    environment = dict(os.environ)
    environment.update(
        {
            "KIMODO_CONTAINER_MODE": "preflight",
            "KIMODO_PYTHON": "/bin/true",
            "KIMODO_STORAGE_ROOT": str(storage_root),
        }
    )
    subprocess.run(
        [str(RUNNER)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        timeout=5,
    )


def test_container_dispatcher_default_is_a_long_running_idle_process():
    environment = dict(os.environ)
    environment.pop("KIMODO_CONTAINER_MODE", None)
    process = subprocess.Popen(
        [str(RUNNER)],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        process.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        pass
    else:
        raise AssertionError("idle container dispatcher exited unexpectedly")
    finally:
        process.terminate()
        process.wait(timeout=5)
