"""Plan, fetch, and verify pinned Hugging Face resources."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from filelock import FileLock, Timeout

from kimodo.common.file_permissions import publish_file

from .config import PathBinding, ResourceCatalog, ResourceFile, ResourcePaths, ResourceSpec


class ResourceVerificationError(RuntimeError):
    """Raised when an existing or downloaded asset violates the catalog."""


_LLM2VEC_FUNCTIONAL_EXTRAS = frozenset({"llm2vec_config.json"})


def _unexpected_functional_files(spec: ResourceSpec, root: Path) -> list[str]:
    if not spec.name.startswith("llm2vec_"):
        return []
    cataloged = {item.path for item in spec.files}
    return sorted(
        name
        for name in _LLM2VEC_FUNCTIONAL_EXTRAS
        if name not in cataloged and (root / name).is_file()
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        publish_file(temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _safe_target(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ResourceVerificationError(f"resource file escapes target root: {relative}") from error
    return target


def _check_file(root: Path, expected: ResourceFile, *, full_hash: bool) -> dict:
    path = _safe_target(root, expected.path)
    if not path.is_file():
        return {"path": expected.path, "status": "missing"}
    size = path.stat().st_size
    if expected.size is not None and size != expected.size:
        return {
            "path": expected.path,
            "status": "size_mismatch",
            "expected_size": expected.size,
            "actual_size": size,
        }
    if full_hash:
        actual = sha256_file(path)
        if actual != expected.sha256:
            return {
                "path": expected.path,
                "status": "sha256_mismatch",
                "expected_sha256": expected.sha256,
                "actual_sha256": actual,
                "actual_size": size,
            }
        return {"path": expected.path, "status": "verified", "size": size}
    return {"path": expected.path, "status": "present_size_ok", "size": size}


def _default_download(**kwargs) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required for fetch; run scripts/resources/setup_env.sh"
        ) from error
    try:
        return str(hf_hub_download(**kwargs))
    except Exception as error:
        # Do not echo request headers or authentication-bearing exception
        # details. The pinned public identity is sufficient for diagnostics.
        raise RuntimeError(
            "Hugging Face download failed for "
            f"{kwargs['repo_id']}/{kwargs['filename']}@{kwargs['revision']}"
        ) from error


class ResourceManager:
    def __init__(
        self,
        catalog: ResourceCatalog,
        paths: ResourcePaths,
        *,
        downloader: Callable[..., str] | None = None,
    ) -> None:
        self.catalog = catalog
        self.paths = paths
        self.downloader = downloader or _default_download

    def _selected(self, groups: list[str] | tuple[str, ...]) -> tuple[ResourceSpec, ...]:
        return self.catalog.select(groups)

    def _inspect(self, spec: ResourceSpec, binding: PathBinding, *, full_hash: bool) -> dict:
        root = binding.target
        files = [_check_file(root, item, full_hash=full_hash) for item in spec.files]
        ok_status = "verified" if full_hash else "present_size_ok"
        unexpected_functional_files = _unexpected_functional_files(spec, root)
        ok = all(item["status"] == ok_status for item in files) and not unexpected_functional_files
        return {
            "name": spec.name,
            "repo_id": spec.repo_id,
            "repo_type": spec.repo_type,
            "revision": spec.revision,
            "mode": binding.mode,
            "path": str(root),
            "ok": ok,
            "verification": "sha256" if full_hash else "presence_and_size",
            "files": files,
            "purpose": spec.purpose,
            "post_fetch": spec.post_fetch,
            "unexpected_functional_files": unexpected_functional_files,
        }

    def plan(self, groups: list[str] | tuple[str, ...]) -> dict:
        resources = []
        for spec in self._selected(groups):
            binding = self.paths.binding(spec.name)
            record = self._inspect(spec, binding, full_hash=False)
            if record["ok"]:
                action = "reuse_candidate" if binding.mode == "existing" else "already_present"
            elif binding.mode == "existing":
                action = "verify_existing_or_fix_paths"
            else:
                action = "download_missing_or_invalid"
            record["action"] = action
            record["network_required"] = binding.mode == "managed" and not record["ok"]
            record["catalog_bytes"] = spec.expected_bytes
            record["estimated_download_bytes"] = sum(
                int(expected.size or 0)
                for expected, current in zip(spec.files, record["files"], strict=True)
                if current["status"] != "present_size_ok"
            )
            resources.append(record)
        return {
            "status": "planned",
            "groups": list(groups),
            "catalog": str(self.catalog.path),
            "paths": str(self.paths.path),
            "resources": resources,
            "note": "plan is a fast presence/size check; verify performs full SHA-256",
        }

    def verify(self, groups: list[str] | tuple[str, ...], *, raise_on_error: bool = True) -> dict:
        resources = [
            self._inspect(spec, self.paths.binding(spec.name), full_hash=True)
            for spec in self._selected(groups)
        ]
        ok = all(record["ok"] for record in resources)
        result = {
            "status": "verified" if ok else "invalid",
            "groups": list(groups),
            "ok": ok,
            "resources": resources,
        }
        if not ok and raise_on_error:
            failures = [record["name"] for record in resources if not record["ok"]]
            raise ResourceVerificationError(
                "resource verification failed for: " + ", ".join(failures)
            )
        return result

    def _fetch_managed(self, spec: ResourceSpec, binding: PathBinding, local_files_only: bool) -> dict:
        if binding.destination is None:
            raise ResourceVerificationError(f"managed resource has no destination: {spec.name}")
        root = binding.destination
        root.mkdir(parents=True, exist_ok=True)
        state_dir = root / ".cache" / "kimodo"
        state_dir.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(state_dir / "fetch.lock"))
        try:
            lock.acquire(timeout=0)
        except Timeout as error:
            raise ResourceVerificationError(
                f"another fetch owns the managed resource lock: {root}"
            ) from error
        try:
            downloaded: list[str] = []
            reused: list[str] = []
            verified_files: list[dict] = []
            for expected in spec.files:
                current = _check_file(root, expected, full_hash=True)
                if current["status"] == "verified":
                    reused.append(expected.path)
                    verified_files.append(current)
                    continue
                target = _safe_target(root, expected.path)
                target.parent.mkdir(parents=True, exist_ok=True)
                self.downloader(
                    repo_id=spec.repo_id,
                    filename=expected.path,
                    repo_type=spec.repo_type,
                    revision=spec.revision,
                    local_dir=str(root),
                    force_download=target.exists(),
                    local_files_only=local_files_only,
                )
                checked = _check_file(root, expected, full_hash=True)
                if checked["status"] != "verified":
                    raise ResourceVerificationError(
                        f"downloaded file failed verification: {spec.name}/{expected.path} "
                        f"({checked['status']})"
                    )
                downloaded.append(expected.path)
                verified_files.append(checked)
            unexpected_functional_files = _unexpected_functional_files(spec, root)
            if unexpected_functional_files:
                raise ResourceVerificationError(
                    f"resource {spec.name} has unpinned functional files: "
                    + ", ".join(unexpected_functional_files)
                )
            receipt = {
                "schema_version": 1,
                "resource": spec.name,
                "repo_id": spec.repo_id,
                "repo_type": spec.repo_type,
                "revision": spec.revision,
                "files": [asdict(item) for item in spec.files],
                "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "post_fetch": spec.post_fetch,
            }
            _atomic_json(state_dir / "resource-receipt.json", receipt)
        finally:
            lock.release()
        return {
            "name": spec.name,
            "repo_id": spec.repo_id,
            "repo_type": spec.repo_type,
            "revision": spec.revision,
            "mode": "managed",
            "path": str(root),
            "ok": True,
            "verification": "sha256",
            "files": verified_files,
            "downloaded": downloaded,
            "reused": reused,
            "purpose": spec.purpose,
            "post_fetch": spec.post_fetch,
            "unexpected_functional_files": [],
        }

    def fetch(
        self,
        groups: list[str] | tuple[str, ...],
        *,
        local_files_only: bool = False,
    ) -> dict:
        resources = []
        for spec in self._selected(groups):
            binding = self.paths.binding(spec.name)
            if binding.mode == "existing":
                record = self._inspect(spec, binding, full_hash=True)
                if not record["ok"]:
                    raise ResourceVerificationError(
                        f"existing_path for {spec.name} failed SHA-256 verification; "
                        "fetch never modifies existing_path"
                    )
                record["downloaded"] = []
                record["reused"] = [item.path for item in spec.files]
                resources.append(record)
            else:
                resources.append(self._fetch_managed(spec, binding, local_files_only))
        verified = {
            "status": "verified",
            "groups": list(groups),
            "ok": True,
            "resources": resources,
        }
        return {
            "status": "fetched_and_verified",
            "groups": list(groups),
            "resources": resources,
            "verification": verified,
            "note": "fetch only acquires pinned files; run the separate preprocessing pipeline next",
        }
