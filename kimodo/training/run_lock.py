"""Inspectable single-writer locks for training output directories."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import time
import uuid
from pathlib import Path

LOCK_NAME = ".kimodo-active-run.lock"


def _boot_id() -> str | None:
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _process_start_ticks(pid: int) -> int | None:
    try:
        # Field 22 is the process start time. Split after the final ')' because
        # Linux permits spaces and parentheses in the comm field.
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_path(output_dir: str | Path) -> Path:
    return Path(output_dir).expanduser().resolve() / LOCK_NAME


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while chunk := os.read(descriptor, 64 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _claim_existing_lock(path: Path) -> int:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise RuntimeError(f"training lock is actively held: {path}") from error
    return descriptor


def _unlink_claimed_path(path: Path, descriptor: int) -> None:
    opened = os.fstat(descriptor)
    current = path.stat()
    if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
        raise RuntimeError("training lock was replaced while it was being inspected")
    path.unlink()


def inspect_run_lock(output_dir: str | Path) -> dict[str, object]:
    """Classify a lock without guessing about processes on another host."""

    path = _lock_path(output_dir)
    if not path.exists():
        return {"status": "absent", "path": str(path), "reclaimable": False}
    try:
        raw = path.read_bytes()
        record = json.loads(raw)
    except (OSError, ValueError) as error:
        return {
            "status": "unknown",
            "path": str(path),
            "reclaimable": False,
            "reason": f"unreadable lock: {type(error).__name__}",
        }
    if not isinstance(record, dict):
        return {
            "status": "unknown",
            "path": str(path),
            "reclaimable": False,
            "reason": "lock record is not an object",
            "record": record,
            "content_sha256": hashlib.sha256(raw).hexdigest(),
        }

    hostname = socket.gethostname()
    if record.get("hostname") != hostname:
        return {
            "status": "unknown",
            "path": str(path),
            "reclaimable": False,
            "reason": "owner is on another host; liveness cannot be proven locally",
            "record": record,
            "content_sha256": hashlib.sha256(raw).hexdigest(),
        }
    try:
        pid = int(record["pid"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "unknown",
            "path": str(path),
            "reclaimable": False,
            "reason": "local lock has no valid pid",
            "record": record,
            "content_sha256": hashlib.sha256(raw).hexdigest(),
        }

    current_boot = _boot_id()
    recorded_boot = record.get("boot_id")
    if current_boot is not None and recorded_boot is not None and recorded_boot != current_boot:
        status, reason = "stale", "owner belongs to an earlier boot of this host"
    else:
        pid_alive = _pid_is_alive(pid)
        observed_start = _process_start_ticks(pid)
        recorded_start = record.get("process_start_ticks")
        if not pid_alive:
            status, reason = "stale", "owner pid no longer exists on this host"
        elif recorded_start is not None and observed_start is not None:
            try:
                same_process = int(recorded_start) == observed_start
            except (TypeError, ValueError):
                return {
                    "status": "unknown",
                    "path": str(path),
                    "reclaimable": False,
                    "reason": "local lock has invalid process_start_ticks",
                    "record": record,
                    "content_sha256": hashlib.sha256(raw).hexdigest(),
                }
            if same_process:
                status, reason = "live", "owner process identity is live on this host"
            else:
                status, reason = "stale", "owner pid was reused by a different process"
        else:
            status, reason = "live", "owner process identity is live on this host"
    return {
        "status": status,
        "path": str(path),
        "reclaimable": status == "stale",
        "reason": reason,
        "record": record,
        "content_sha256": hashlib.sha256(raw).hexdigest(),
    }


def clear_stale_run_lock(output_dir: str | Path) -> dict[str, object]:
    """Remove only a lock whose local owner is provably stale."""

    inspection = inspect_run_lock(output_dir)
    if inspection["status"] == "absent":
        return inspection
    if not inspection["reclaimable"]:
        raise RuntimeError(
            f"refusing to clear {inspection['status']} training lock: "
            f"{inspection.get('reason')}"
        )
    path = Path(str(inspection["path"]))
    try:
        descriptor = _claim_existing_lock(path)
    except FileNotFoundError as error:
        raise RuntimeError("training lock disappeared while it was being inspected") from error
    try:
        raw = _read_descriptor(descriptor)
        if hashlib.sha256(raw).hexdigest() != inspection.get("content_sha256"):
            raise RuntimeError("training lock changed while it was being inspected")
        _unlink_claimed_path(path, descriptor)
    finally:
        os.close(descriptor)
    return {**inspection, "status": "cleared", "reclaimable": False}


def clear_run_lock_with_token(output_dir: str | Path, expected_token: str) -> dict[str, object]:
    """Explicit operator recovery for a remote/unknown lock after external liveness checks."""

    path = _lock_path(output_dir)
    try:
        descriptor = _claim_existing_lock(path)
    except FileNotFoundError:
        return {"status": "absent", "path": str(path)}
    try:
        raw = _read_descriptor(descriptor)
        record = json.loads(raw)
        if not isinstance(record, dict) or record.get("token") != expected_token:
            raise RuntimeError("refusing to clear training lock: expected token does not match")
        _unlink_claimed_path(path, descriptor)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot safely clear unreadable training lock: {path}") from error
    finally:
        os.close(descriptor)
    return {"status": "cleared", "path": str(path), "record": record}


class ExclusiveRunLock:
    """POSIX single-writer lock with conservative local stale-owner recovery.

    Correct cross-host behavior still depends on the shared filesystem honoring
    ``O_EXCL`` creation and ``flock`` semantics; that is an operational property,
    not something this process can prove about an arbitrary NFS deployment.
    """

    def __init__(self, output_dir: Path) -> None:
        self.path = _lock_path(output_dir)
        self.token = uuid.uuid4().hex
        self.held = False
        self.descriptor: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        record = {
            "schema_version": 2,
            "token": self.token,
            "pid": pid,
            "hostname": socket.gethostname(),
            "boot_id": _boot_id(),
            "process_start_ticks": _process_start_ticks(pid),
            "started_at_unix": time.time(),
        }
        for attempt in range(2):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
            except FileExistsError as error:
                inspection = inspect_run_lock(self.path.parent)
                if attempt == 0 and inspection.get("reclaimable") is True:
                    clear_stale_run_lock(self.path.parent)
                    continue
                raise FileExistsError(
                    f"training output_dir is already locked: {self.path}; "
                    f"status={inspection['status']}; reason={inspection.get('reason')}; "
                    f"owner={inspection.get('record', '<unreadable>')}"
                ) from error
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                payload = memoryview(
                    (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
                )
                while payload:
                    written = os.write(descriptor, payload)
                    if written <= 0:
                        raise OSError("short write while publishing training lock")
                    payload = payload[written:]
                os.fsync(descriptor)
            except BaseException:
                try:
                    _unlink_claimed_path(self.path, descriptor)
                except (FileNotFoundError, OSError, RuntimeError):
                    pass
                os.close(descriptor)
                raise
            self.descriptor = descriptor
            self.held = True
            return
        raise AssertionError("unreachable lock acquisition state")

    def release(self) -> None:
        if not self.held:
            return
        descriptor = self.descriptor
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                descriptor is not None
                and isinstance(current, dict)
                and current.get("token") == self.token
            ):
                _unlink_claimed_path(self.path, descriptor)
        except (FileNotFoundError, OSError, ValueError):
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)
            self.descriptor = None
        self.held = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("output_dir")
    stale_parser = subparsers.add_parser("clear-stale")
    stale_parser.add_argument("output_dir")
    clear_parser = subparsers.add_parser("clear")
    clear_parser.add_argument("output_dir")
    clear_parser.add_argument("--expected-token", required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_run_lock(args.output_dir)
    elif args.command == "clear-stale":
        result = clear_stale_run_lock(args.output_dir)
    else:
        result = clear_run_lock_with_token(args.output_dir, args.expected_token)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
