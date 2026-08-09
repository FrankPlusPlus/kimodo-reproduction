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
    assert 'CMD ["/workspace/scripts/train_company.sh"]' not in source
    assert 'CMD ["/workspace/scripts/train_company_16h200.sh"]' not in source
    assert "KIMODO_CODE_ROOT=/workspace" in source
    assert "KIMODO_PYTHON=python /workspace/scripts/smoke_train.sh" in source


def test_dockerfile_defaults_to_company_share_pvc_paths():
    source = DOCKERFILE.read_text(encoding="utf-8")
    assert (
        "KIMODO_CODE_ROOT=/home/share/yzt/kimodo-reproduction" in source
    )
    assert (
        "KIMODO_STORAGE_ROOT=/home/share/yezitao-kimodo-reproduction" in source
    )
    assert "KIMODO_DATA_ROOT=" not in source
    assert "KIMODO_RUN_ROOT=" not in source
    assert "zstd openssh-server" in source


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
    runner = (PROJECT_ROOT / "scripts/container_start.sh").read_text(encoding="utf-8")
    assert 'DEFAULT_CODE_ROOT="/home/share/yzt/kimodo-reproduction"' in runner
    assert "PYTHONPATH=" in runner


def test_image_provides_public_key_only_ssh_for_remote_pod_development():
    source = DOCKERFILE.read_text(encoding="utf-8")
    entrypoint = (PROJECT_ROOT / "kimodo/scripts/docker-entrypoint.sh").read_text(
        encoding="utf-8"
    )
    assert "openssh-server" in source
    assert "PasswordAuthentication no" in source
    assert "PermitRootLogin prohibit-password" in source
    assert "ARG NB_USER=jovyan" in source
    assert 'useradd -M -u "${NB_UID}"' in source
    assert "usermod -p '*'" in source
    assert "KIMODO_SSH_PUBLIC_KEY" in entrypoint
    assert "collect_ssh_authorized_keys" in entrypoint
    assert "ensure_notebook_user" in entrypoint
    assert 'usermod -p \'*\'' in entrypoint or "usermod -p '*'" in entrypoint
    assert 'NB_USER="${NB_USER:-jovyan}"' in entrypoint
    assert "/home/${NB_USER}/.ssh/authorized_keys" in entrypoint
    assert "/root/.ssh/authorized_keys" in entrypoint
    assert "/etc/ssh/kimodo_authorized_keys" in entrypoint
    assert "world-writable sticky" in entrypoint
    assert "no authorized_keys found" in entrypoint
    assert "/usr/sbin/sshd" in entrypoint
    assert "continuing without SSH" in entrypoint
    assert "/tmp/kimodo-sshd" in entrypoint
    assert "Jupyter/Kubeflow default login" in source


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


def test_load_kimodo_env_fills_only_unset_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "WANDB_API_KEY=from-file\n"
        "PRODUCT_GRAPH_LLM_API_KEY=mimo-from-file\n"
        "ALREADY_SET=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KIMODO_CODE_ROOT", str(tmp_path))
    monkeypatch.setenv("KIMODO_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ALREADY_SET", "from-process")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("PRODUCT_GRAPH_LLM_API_KEY", raising=False)
    script = PROJECT_ROOT / "scripts/load_kimodo_env.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source '{script}' && kimodo_load_env_files && "
            'printf "%s\\n%s\\n%s\\n" "$WANDB_API_KEY" "$PRODUCT_GRAPH_LLM_API_KEY" "$ALREADY_SET"',
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        env=dict(os.environ),
    )
    assert result.stdout.strip().splitlines() == [
        "from-file",
        "mimo-from-file",
        "from-process",
    ]


def test_container_dispatcher_prefers_pvc_code_root(tmp_path):
    code_root = tmp_path / "kimodo-reproduction"
    (code_root / "kimodo").mkdir(parents=True)
    scripts = code_root / "scripts"
    scripts.mkdir()
    marker = tmp_path / "ran-from-pvc"
    launcher = scripts / "train_company.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        f"printf 'ok' > '{marker}'\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        {
            "KIMODO_CONTAINER_MODE": "train-company",
            "KIMODO_CODE_ROOT": str(code_root),
        }
    )
    subprocess.run(
        [str(RUNNER)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        timeout=5,
    )
    assert marker.read_text(encoding="utf-8") == "ok"


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
