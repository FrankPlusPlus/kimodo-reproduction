from __future__ import annotations

import json
import os
import socket

import pytest

import kimodo.training.run_lock as run_lock_module
from kimodo.training.run_lock import (
    LOCK_NAME,
    ExclusiveRunLock,
    clear_run_lock_with_token,
    clear_stale_run_lock,
    inspect_run_lock,
)


def test_run_lock_rejects_live_owner_and_releases_by_token(tmp_path) -> None:
    lock = ExclusiveRunLock(tmp_path)
    lock.acquire()
    assert inspect_run_lock(tmp_path)["status"] == "live"
    with pytest.raises(FileExistsError, match="status=live"):
        ExclusiveRunLock(tmp_path).acquire()
    with pytest.raises(RuntimeError, match="actively held"):
        clear_run_lock_with_token(tmp_path, lock.token)
    lock.release()
    assert inspect_run_lock(tmp_path)["status"] == "absent"


def test_run_lock_recovers_provably_dead_local_owner(tmp_path) -> None:
    path = tmp_path / LOCK_NAME
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "token": "dead-owner",
                "pid": 2**30,
                "hostname": socket.gethostname(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert inspect_run_lock(tmp_path)["status"] == "stale"
    lock = ExclusiveRunLock(tmp_path)
    lock.acquire()
    assert json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid()
    lock.release()


def test_run_lock_never_guesses_about_another_host(tmp_path) -> None:
    path = tmp_path / LOCK_NAME
    path.write_text(
        json.dumps({"token": "remote", "pid": 123, "hostname": "another-node"}) + "\n",
        encoding="utf-8",
    )
    inspection = inspect_run_lock(tmp_path)
    assert inspection["status"] == "unknown"
    assert inspection["reclaimable"] is False
    with pytest.raises(RuntimeError, match="refusing to clear unknown"):
        clear_stale_run_lock(tmp_path)
    with pytest.raises(RuntimeError, match="expected token does not match"):
        clear_run_lock_with_token(tmp_path, "wrong")
    assert clear_run_lock_with_token(tmp_path, "remote")["status"] == "cleared"


def test_stale_clear_never_deletes_a_replaced_live_lock(tmp_path, monkeypatch) -> None:
    path = tmp_path / LOCK_NAME
    path.write_text(
        json.dumps({"token": "stale", "pid": 2**30, "hostname": socket.gethostname()}),
        encoding="utf-8",
    )
    real_inspect = run_lock_module.inspect_run_lock

    def inspect_then_replace(output_dir):
        inspection = real_inspect(output_dir)
        path.unlink()
        path.write_text(
            json.dumps({"token": "replacement", "pid": os.getpid(), "hostname": socket.gethostname()}),
            encoding="utf-8",
        )
        return inspection

    monkeypatch.setattr(run_lock_module, "inspect_run_lock", inspect_then_replace)
    with pytest.raises(RuntimeError, match="changed while"):
        clear_stale_run_lock(tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["token"] == "replacement"
