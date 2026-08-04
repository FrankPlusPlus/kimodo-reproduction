"""Strict YAML contracts for remote resources and machine-local paths."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


class ResourceConfigError(ValueError):
    """Raised when a resource catalog or paths file is ambiguous."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ResourceConfigError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResourceConfigError(f"{label} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ResourceConfigError(f"{label} keys must be strings")
    return value


def _only_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ResourceConfigError(f"unknown {label} keys: {unknown}")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResourceConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    return _mapping(payload, str(path))


@dataclass(frozen=True)
class ResourceFile:
    path: str
    sha256: str
    size: int | None = None


@dataclass(frozen=True)
class ResourceSpec:
    name: str
    repo_id: str
    repo_type: str
    revision: str
    files: tuple[ResourceFile, ...]
    expected_bytes: int | None = None
    opt_in: bool = False
    purpose: str = ""
    post_fetch: str = "none"


@dataclass(frozen=True)
class ResourceCatalog:
    path: Path
    groups: dict[str, tuple[str, ...]]
    resources: dict[str, ResourceSpec]

    def select(self, groups: list[str] | tuple[str, ...]) -> tuple[ResourceSpec, ...]:
        if not groups:
            raise ResourceConfigError("at least one resource group is required")
        names: list[str] = []
        for group in groups:
            if group not in self.groups:
                raise ResourceConfigError(f"unknown resource group: {group!r}")
            for name in self.groups[group]:
                if name not in names:
                    names.append(name)
        return tuple(self.resources[name] for name in names)


@dataclass(frozen=True)
class PathBinding:
    destination: Path | None
    existing_path: Path | None

    @property
    def target(self) -> Path:
        if self.existing_path is not None:
            return self.existing_path
        if self.destination is not None:
            return self.destination
        raise ResourceConfigError("resource path has neither destination nor existing_path")

    @property
    def mode(self) -> str:
        return "existing" if self.existing_path is not None else "managed"


@dataclass(frozen=True)
class ResourcePaths:
    path: Path
    resources: dict[str, PathBinding]
    pipeline: PipelinePaths | None = None

    def binding(self, name: str) -> PathBinding:
        try:
            return self.resources[name]
        except KeyError as error:
            raise ResourceConfigError(
                f"paths file has no entry for selected resource {name!r}"
            ) from error


@dataclass(frozen=True)
class PipelinePaths:
    """Machine-local outputs and operational knobs for public preprocessing."""

    dataset_root: Path
    prepared_root: Path
    run_root: Path
    repro_paths_yaml: Path
    text_device: str = "cuda:0"
    motion_workers: int = 8
    threads_per_worker: int = 2
    stats_workers: int = 16


def _parse_file(path: str, raw: Any, label: str) -> ResourceFile:
    posix = PurePosixPath(path)
    if (
        "\\" in path
        or posix.is_absolute()
        or not posix.parts
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ResourceConfigError(f"{label} path must be a safe relative POSIX path: {path!r}")
    record = _mapping(raw, label)
    _only_keys(record, {"sha256", "size"}, label)
    digest = _nonempty_string(record.get("sha256"), f"{label}.sha256").lower()
    if not _SHA256.fullmatch(digest):
        raise ResourceConfigError(f"{label}.sha256 must be 64 lowercase hex characters")
    size = record.get("size")
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
        raise ResourceConfigError(f"{label}.size must be a non-negative integer")
    return ResourceFile(path=posix.as_posix(), sha256=digest, size=size)


def load_catalog(path: str | Path) -> ResourceCatalog:
    source = Path(path).expanduser().resolve()
    raw = _load_yaml(source)
    _only_keys(raw, {"schema_version", "groups", "resources"}, "catalog")
    if raw.get("schema_version") != 1:
        raise ResourceConfigError("catalog.schema_version must be 1")
    resources_raw = _mapping(raw.get("resources"), "catalog.resources")
    resources: dict[str, ResourceSpec] = {}
    for name, value in resources_raw.items():
        record = _mapping(value, f"catalog.resources.{name}")
        _only_keys(
            record,
            {
                "repo_id",
                "repo_type",
                "revision",
                "files",
                "expected_bytes",
                "opt_in",
                "purpose",
                "post_fetch",
            },
            f"catalog.resources.{name}",
        )
        repo_id = _nonempty_string(record.get("repo_id"), f"{name}.repo_id")
        if repo_id.count("/") != 1 or any(character.isspace() for character in repo_id):
            raise ResourceConfigError(f"{name}.repo_id must have the form organization/name")
        repo_type = _nonempty_string(record.get("repo_type"), f"{name}.repo_type")
        if repo_type not in {"model", "dataset"}:
            raise ResourceConfigError(f"{name}.repo_type must be model or dataset")
        revision = _nonempty_string(record.get("revision"), f"{name}.revision").lower()
        if not _REVISION.fullmatch(revision):
            raise ResourceConfigError(f"{name}.revision must be a pinned 40-character commit")
        files_raw = _mapping(record.get("files"), f"{name}.files")
        if not files_raw:
            raise ResourceConfigError(f"{name}.files must not be empty")
        files = tuple(
            _parse_file(filename, file_record, f"{name}.files.{filename}")
            for filename, file_record in sorted(files_raw.items())
        )
        expected_bytes = record.get("expected_bytes")
        if expected_bytes is not None and (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise ResourceConfigError(f"{name}.expected_bytes must be a non-negative integer")
        if expected_bytes is not None and all(item.size is not None for item in files):
            file_bytes = sum(int(item.size) for item in files if item.size is not None)
            if expected_bytes != file_bytes:
                raise ResourceConfigError(
                    f"{name}.expected_bytes={expected_bytes} does not equal file sizes={file_bytes}"
                )
        opt_in = record.get("opt_in", False)
        if not isinstance(opt_in, bool):
            raise ResourceConfigError(f"{name}.opt_in must be boolean")
        purpose = record.get("purpose", "")
        post_fetch = record.get("post_fetch", "none")
        if not isinstance(purpose, str) or not isinstance(post_fetch, str):
            raise ResourceConfigError(f"{name}.purpose/post_fetch must be strings")
        resources[name] = ResourceSpec(
            name=name,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            files=files,
            expected_bytes=expected_bytes,
            opt_in=opt_in,
            purpose=purpose,
            post_fetch=post_fetch,
        )

    groups_raw = _mapping(raw.get("groups"), "catalog.groups")
    groups: dict[str, tuple[str, ...]] = {}
    for group, value in groups_raw.items():
        if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
            raise ResourceConfigError(f"catalog.groups.{group} must be a non-empty string list")
        missing = sorted(set(value) - resources.keys())
        if missing:
            raise ResourceConfigError(f"catalog.groups.{group} references unknown resources: {missing}")
        if len(value) != len(set(value)):
            raise ResourceConfigError(f"catalog.groups.{group} contains duplicates")
        groups[group] = tuple(value)
    if "train-minimal" not in groups:
        raise ResourceConfigError("catalog must define the train-minimal group")
    opted_in = [name for name in groups["train-minimal"] if resources[name].opt_in]
    if opted_in:
        raise ResourceConfigError(f"train-minimal must not include opt-in resources: {opted_in}")
    return ResourceCatalog(path=source, groups=groups, resources=resources)


def _resolve_local_path(value: Any, base: Path, label: str) -> Path | None:
    if value is None:
        return None
    text = _nonempty_string(value, label)
    expanded = os.path.expandvars(os.path.expanduser(text))
    if "$" in expanded:
        raise ResourceConfigError(f"{label} contains an unresolved environment variable")
    path = Path(expanded)
    return (path if path.is_absolute() else base / path).resolve()


def load_paths(path: str | Path, catalog: ResourceCatalog) -> ResourcePaths:
    source = Path(path).expanduser().resolve()
    raw = _load_yaml(source)
    _only_keys(raw, {"schema_version", "resources", "pipeline"}, "paths")
    if raw.get("schema_version") != 1:
        raise ResourceConfigError("paths.schema_version must be 1")
    resources_raw = _mapping(raw.get("resources"), "paths.resources")
    unknown = sorted(set(resources_raw) - catalog.resources.keys())
    if unknown:
        raise ResourceConfigError(f"paths file references unknown resources: {unknown}")
    bindings: dict[str, PathBinding] = {}
    for name, value in resources_raw.items():
        record = _mapping(value, f"paths.resources.{name}")
        _only_keys(record, {"destination", "existing_path"}, f"paths.resources.{name}")
        destination = _resolve_local_path(
            record.get("destination"), source.parent, f"paths.resources.{name}.destination"
        )
        existing = _resolve_local_path(
            record.get("existing_path"), source.parent, f"paths.resources.{name}.existing_path"
        )
        if destination is None and existing is None:
            raise ResourceConfigError(
                f"paths.resources.{name} needs destination or existing_path"
            )
        bindings[name] = PathBinding(destination=destination, existing_path=existing)
    pipeline = None
    pipeline_raw = raw.get("pipeline")
    if pipeline_raw is not None:
        record = _mapping(pipeline_raw, "paths.pipeline")
        _only_keys(
            record,
            {
                "dataset_root",
                "prepared_root",
                "run_root",
                "repro_paths_yaml",
                "text_device",
                "motion_workers",
                "threads_per_worker",
                "stats_workers",
            },
            "paths.pipeline",
        )
        required = ("dataset_root", "prepared_root", "run_root", "repro_paths_yaml")
        resolved = {
            name: _resolve_local_path(record.get(name), source.parent, f"paths.pipeline.{name}")
            for name in required
        }
        missing = [name for name, value in resolved.items() if value is None]
        if missing:
            raise ResourceConfigError(f"paths.pipeline is missing required paths: {missing}")
        text_device = record.get("text_device", "cuda:0")
        if not isinstance(text_device, str) or not text_device.strip():
            raise ResourceConfigError("paths.pipeline.text_device must be a non-empty string")

        def positive_int(name: str, default: int) -> int:
            value = record.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ResourceConfigError(f"paths.pipeline.{name} must be a positive integer")
            return value

        pipeline = PipelinePaths(
            dataset_root=resolved["dataset_root"],  # type: ignore[arg-type]
            prepared_root=resolved["prepared_root"],  # type: ignore[arg-type]
            run_root=resolved["run_root"],  # type: ignore[arg-type]
            repro_paths_yaml=resolved["repro_paths_yaml"],  # type: ignore[arg-type]
            text_device=text_device.strip(),
            motion_workers=positive_int("motion_workers", 8),
            threads_per_worker=positive_int("threads_per_worker", 2),
            stats_workers=positive_int("stats_workers", 16),
        )
    return ResourcePaths(path=source, resources=bindings, pipeline=pipeline)
