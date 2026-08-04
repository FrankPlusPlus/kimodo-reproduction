"""Idempotent public BONES-SEED preprocessing from pinned resources to training assets."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from filelock import FileLock, Timeout

from kimodo.training.file_permissions import publish_file

from .config import PipelinePaths, ResourceCatalog, ResourcePaths


class PipelineError(RuntimeError):
    """Raised when a derived stage is partial, stale, or cannot be reproduced."""


EXPECTED_OFFICIAL_SPLIT_ENTRIES = 128351
EXPECTED_EFFECTIVE_SPLIT_ENTRIES = 128315
EXPECTED_MISSING_SPLIT_SHA256 = "dae2c4e03bdc2d5c1383e06f9dedb1d62d2c5e3dcc60937e012dfec1cab20d19"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if existing == payload:
            return
        raise PipelineError(
            f"refusing to overwrite edited/generated paths YAML: {path}; "
            "choose a new pipeline.repro_paths_yaml or inspect and remove the old generated file"
        )
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
            yaml.safe_dump(payload, output, sort_keys=False)
            output.flush()
            os.fsync(output.fileno())
        publish_file(temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def _safe_extract(archive: Path, destination: Path) -> str:
    """Extract once through same-filesystem staging and reject links/path escapes."""

    expected = destination / "soma_uniform" / "bvh"
    if expected.is_dir():
        return "reuse"
    if destination.exists():
        raise PipelineError(
            f"dataset_root exists but has no soma_uniform/bvh; inspect before retrying: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.extract.", dir=destination.parent)
    )
    published = False
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            root = staging.resolve()
            extracted_bytes = 0
            for member in bundle.getmembers():
                member_path = Path(member.name)
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or member.issym()
                    or member.islnk()
                    or not (member.isfile() or member.isdir())
                    or not (root / member_path).resolve().is_relative_to(root)
                ):
                    raise PipelineError(f"unsafe archive member: {member.name!r}")
                if member.isfile():
                    extracted_bytes += int(member.size)
            free_bytes = shutil.disk_usage(destination.parent).free
            required_bytes = extracted_bytes + 1024**3
            if free_bytes < required_bytes:
                raise PipelineError(
                    "insufficient free space for BONES extraction: "
                    f"need at least {required_bytes} bytes, have {free_bytes}"
                )
            # Members were validated above; omit Python 3.12's ``filter``
            # argument so the supported Python 3.10/3.11 runtimes also work.
            bundle.extractall(staging)
        if not (staging / "soma_uniform" / "bvh").is_dir():
            raise PipelineError("BONES archive did not contain soma_uniform/bvh")
        os.replace(staging, destination)
        published = True
    finally:
        if not published and staging.exists():
            # Only remove our uniquely named, unpublished staging tree.
            shutil.rmtree(staging)
    return "extract"


def _pair_state(
    primary: Path,
    metadata: Path,
    *,
    label: str,
    schema_version: int | None = None,
    path_mode: str | None = None,
) -> str:
    present = (primary.exists(), metadata.exists())
    if any(present) and not all(present):
        raise PipelineError(f"orphaned {label} output requires review: {primary}, {metadata}")
    if not all(present):
        return "build"
    if schema_version is not None or path_mode is not None:
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if schema_version is not None and record.get("schema_version") != schema_version:
            raise PipelineError(
                f"{label} uses legacy schema {record.get('schema_version')!r}; "
                f"a portable rebuild requires schema {schema_version}: {primary}"
            )
        if path_mode is not None and record.get("path_mode") != path_mode:
            raise PipelineError(
                f"{label} is not portable (path_mode={record.get('path_mode')!r}); "
                f"rebuild it with path_mode={path_mode}: {primary}"
            )
    return "reuse"


def _run(argv: list[str]) -> None:
    print("+ " + " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def _require_pipeline(paths: ResourcePaths) -> PipelinePaths:
    if paths.pipeline is None:
        raise PipelineError("paths YAML has no pipeline section")
    return paths.pipeline


def _resource_root(paths: ResourcePaths, name: str) -> Path:
    return paths.binding(name).target


def _flowmatching_identity(catalog: ResourceCatalog) -> dict[str, str]:
    lock_path = catalog.path.parent / "dependencies.lock.yaml"
    if not lock_path.is_file():
        raise PipelineError(f"flowmatching dependency lock is missing: {lock_path}")
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    expected = str((lock.get("flowmatching") or {}).get("revision", ""))
    if len(expected) != 40 or any(character not in "0123456789abcdef" for character in expected):
        raise PipelineError(f"invalid flowmatching revision in {lock_path}")
    try:
        package = importlib.import_module("kimodo_flow")
    except ImportError as error:
        raise PipelineError(
            "canonical conversion requires the locked kimodo-flowmatching checkout; "
            "rerun setup_env.sh --flowmatching-repo /path/to/checkout"
        ) from error
    package_path = Path(package.__file__).resolve()
    repository = next((parent for parent in package_path.parents if (parent / ".git").exists()), None)
    if repository is None:
        raise PipelineError(
            f"installed kimodo_flow is not traceable to a Git checkout: {package_path}"
        )
    actual = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise PipelineError(
            f"flowmatching revision mismatch: expected={expected}, actual={actual}"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty and os.environ.get("KIMODO_ALLOW_DIRTY_FLOWMATCHING") != "1":
        raise PipelineError("flowmatching checkout is dirty; converter producer identity is not pinned")
    return {
        "remote": str((lock.get("flowmatching") or {}).get("remote", "")),
        "revision": actual,
        "lock_sha256": _sha256(lock_path),
        "dirty_override": str(bool(dirty)).lower(),
    }


def _validate_manifest_pair(manifest: Path, *, source: Path | None = None) -> None:
    sidecar = manifest.with_suffix(manifest.suffix + ".metadata.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    output = metadata.get("output", {})
    if output.get("sha256") != _sha256(manifest):
        raise PipelineError(f"manifest content differs from sidecar: {manifest}")
    if source is not None and metadata.get("source_manifest_sha256") != _sha256(source):
        raise PipelineError(f"derived manifest has a different source: {manifest}")


def _validate_conversion_inventory(
    inventory: Path,
    *,
    dataset_root: Path,
    motion_root: Path,
    expected_coverage: tuple[int, int, str] | None = None,
) -> None:
    metadata_path = inventory.with_suffix(inventory.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_rows = metadata.get("effective_entries")
    rows = 0
    with inventory.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            rows += 1
            for field, root, digest_field in (
                ("source", dataset_root, "source_sha256"),
                ("cached", motion_root, "cached_sha256"),
            ):
                value = record.get(field)
                if not isinstance(value, str) or not value:
                    raise PipelineError(f"conversion inventory line {line_number} lacks {field}")
                candidate = (root / value).resolve()
                if not candidate.is_relative_to(root.resolve()):
                    raise PipelineError(
                        f"conversion inventory line {line_number} escapes {field} root"
                    )
                if not candidate.is_file():
                    raise PipelineError(f"conversion output is missing: {candidate}")
                if record.get(digest_field) != _sha256(candidate):
                    raise PipelineError(f"conversion output hash mismatch: {candidate}")
    if not isinstance(expected_rows, int) or expected_rows != rows:
        raise PipelineError(
            f"conversion inventory row count mismatch: metadata={expected_rows!r}, actual={rows}"
        )
    if expected_coverage is not None:
        observed = (
            metadata.get("official_split_entries"),
            metadata.get("effective_entries"),
            metadata.get("missing_from_metadata_sha256"),
        )
        if observed != expected_coverage:
            raise PipelineError(
                f"pinned BONES/benchmark coverage changed: expected={expected_coverage}, "
                f"observed={observed}"
            )


def _validate_stats_bundle(root: Path) -> dict[str, str]:
    metadata_path = root / "stats.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 3:
        raise PipelineError("stats require integrity/minimum-span schema_version=3")
    records = metadata.get("files")
    expected = {
        f"{group}/{filename}": dimension
        for group, dimension in (("global_root", 5), ("local_root", 4), ("body", 364))
        for filename in ("mean.npy", "std.npy")
    }
    if not isinstance(records, dict) or set(records) != set(expected):
        raise PipelineError("stats metadata does not bind exactly the six expected arrays")
    verified: dict[str, str] = {}
    for relative, dimension in expected.items():
        path = root / relative
        record = records[relative]
        if not isinstance(record, dict) or not path.is_file():
            raise PipelineError(f"stats array is missing or unbound: {path}")
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError) as error:
            raise PipelineError(f"stats array is unreadable: {path}") from error
        if (
            array.dtype.name != "float32"
            or array.shape != (dimension,)
            or not np.isfinite(array).all()
            or record.get("dtype") != "float32"
            or record.get("shape") != [dimension]
            or record.get("size") != path.stat().st_size
            or record.get("sha256") != _sha256(path)
        ):
            raise PipelineError(f"stats array failed integrity/shape validation: {path}")
        verified[relative] = str(record["sha256"])
    return verified


def plan_pipeline(paths: ResourcePaths) -> dict[str, Any]:
    pipeline = _require_pipeline(paths)
    prepared = pipeline.prepared_root
    conversion = prepared / "conversion" / "soma30-30fps.inventory.jsonl"
    raw = prepared / "train.raw.jsonl"
    cached = prepared / "train.cached.jsonl"
    stats = prepared / "stats" / "repro-soma30-30fps"
    inventory = prepared / "train.cached.references.jsonl"
    return {
        "dataset_extract": (
            "reuse"
            if (pipeline.dataset_root / "soma_uniform" / "bvh").is_dir()
            else "extract"
        ),
        "canonical_motion": _pair_state(
            conversion,
            conversion.with_suffix(conversion.suffix + ".metadata.json"),
            label="conversion inventory",
        ),
        "raw_manifest": _pair_state(
            raw,
            raw.with_suffix(raw.suffix + ".metadata.json"),
            label="raw manifest",
            schema_version=2,
            path_mode="relative",
        ),
        "text_cache": _pair_state(
            cached,
            cached.with_suffix(cached.suffix + ".metadata.json"),
            label="cached manifest",
            schema_version=5,
            path_mode="relative",
        ),
        "stats": "reuse" if stats.is_dir() and _validate_stats_bundle(stats) else "build",
        "reference_inventory": _pair_state(
            inventory,
            inventory.with_suffix(inventory.suffix + ".metadata.json"),
            label="reference inventory",
            schema_version=2,
        ),
        "repro_paths_yaml": str(pipeline.repro_paths_yaml),
    }


def prepare_pipeline(
    catalog: ResourceCatalog, paths: ResourcePaths, *, dry_run: bool = False
) -> dict[str, Any]:
    """Build/reuse every public repro artifact and emit the training paths YAML."""

    pipeline = _require_pipeline(paths)
    plan = plan_pipeline(paths)
    if dry_run:
        return {"status": "planned", "stages": plan}

    prepared = pipeline.prepared_root
    prepared.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(prepared.parent / f".{prepared.name}.prepare.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout as error:
        raise PipelineError(f"another process is preparing {prepared}") from error
    try:
        bones = _resource_root(paths, "bones_seed")
        benchmark = _resource_root(paths, "kimodo_benchmark")
        archive = bones / "soma_uniform.tar.gz"
        metadata = bones / "metadata" / "seed_metadata_v004.csv"
        temporal = bones / "metadata" / "seed_metadata_v002_temporal_labels.jsonl"
        split = benchmark / "splits" / "train_split_paths.txt"
        for required in (archive, metadata, temporal, split):
            if not required.is_file():
                raise FileNotFoundError(required)
        flowmatching_identity = _flowmatching_identity(catalog)
        _safe_extract(archive, pipeline.dataset_root)

        prepared.mkdir(parents=True, exist_ok=True)
        motions = prepared / "motions" / "soma30-30fps"
        conversion = prepared / "conversion" / "soma30-30fps.inventory.jsonl"
        conversion_meta = conversion.with_suffix(conversion.suffix + ".metadata.json")
        if plan["canonical_motion"] == "build":
            try:
                __import__("kimodo_flow")
            except ImportError as error:
                raise PipelineError(
                    "canonical conversion requires kimodo-flowmatching in this environment; "
                    "clone it anywhere and rerun setup_env.sh --flowmatching-repo /path/to/clone"
                ) from error
            _run(
                [
                    sys.executable,
                    "-m",
                    "kimodo_flow.data.bones",
                    "--dataset-root",
                    str(pipeline.dataset_root),
                    "--metadata",
                    str(metadata),
                    "--split-file",
                    str(split),
                    "--output-root",
                    str(motions),
                    "--inventory",
                    str(conversion),
                    "--workers",
                    str(pipeline.motion_workers),
                    "--threads-per-worker",
                    str(pipeline.threads_per_worker),
                    "--expected-split-entries",
                    str(EXPECTED_OFFICIAL_SPLIT_ENTRIES),
                    "--expected-effective-entries",
                    str(EXPECTED_EFFECTIVE_SPLIT_ENTRIES),
                    "--expected-missing-sha256",
                    EXPECTED_MISSING_SPLIT_SHA256,
                ]
            )
        else:
            record = json.loads(conversion_meta.read_text(encoding="utf-8"))
            if (
                record.get("inventory_sha256") != _sha256(conversion)
                or record.get("metadata_sha256") != _sha256(metadata)
                or record.get("split_sha256") != _sha256(split)
            ):
                raise PipelineError("conversion inventory provenance is stale")
        _validate_conversion_inventory(
            conversion,
            dataset_root=pipeline.dataset_root,
            motion_root=motions,
            expected_coverage=(
                EXPECTED_OFFICIAL_SPLIT_ENTRIES,
                EXPECTED_EFFECTIVE_SPLIT_ENTRIES,
                EXPECTED_MISSING_SPLIT_SHA256,
            ),
        )

        raw = prepared / "train.raw.jsonl"
        if plan["raw_manifest"] == "build":
            _run(
                [
                    sys.executable,
                    "-m",
                    "kimodo.training.manifest_cli",
                    "--metadata",
                    str(metadata),
                    "--temporal-labels",
                    str(temporal),
                    "--split-file",
                    str(split),
                    "--dataset-root",
                    str(pipeline.dataset_root),
                    "--motion-cache-root",
                    str(motions),
                    "--motion-cache-fps",
                    "30",
                    "--motion-inventory",
                    str(conversion),
                    "--path-mode",
                    "relative",
                    "--output",
                    str(raw),
                ]
            )
        else:
            _validate_manifest_pair(raw)

        cached = prepared / "train.cached.jsonl"
        cache_dir = prepared / "text-cache"
        if plan["text_cache"] == "build":
            foundation = _resource_root(paths, "llm2vec_foundation")
            mntp = _resource_root(paths, "llm2vec_mntp_adapter")
            supervised = _resource_root(paths, "llm2vec_supervised_adapter")
            _run(
                [
                    sys.executable,
                    "-m",
                    "kimodo.training.text_cache_cli",
                    "--manifest",
                    str(raw),
                    "--output-manifest",
                    str(cached),
                    "--cache-dir",
                    str(cache_dir),
                    "--provider",
                    "local",
                    "--device",
                    pipeline.text_device,
                    "--model-lock",
                    str(catalog.path),
                    "--foundation-model",
                    str(foundation),
                    "--foundation-repo-id",
                    catalog.resources["llm2vec_foundation"].repo_id,
                    "--foundation-revision",
                    catalog.resources["llm2vec_foundation"].revision,
                    "--mntp-model",
                    str(mntp),
                    "--mntp-repo-id",
                    catalog.resources["llm2vec_mntp_adapter"].repo_id,
                    "--mntp-revision",
                    catalog.resources["llm2vec_mntp_adapter"].revision,
                    "--supervised-model",
                    str(supervised),
                    "--supervised-repo-id",
                    catalog.resources["llm2vec_supervised_adapter"].repo_id,
                    "--supervised-revision",
                    catalog.resources["llm2vec_supervised_adapter"].revision,
                ]
            )
        else:
            _validate_manifest_pair(cached, source=raw)

        stats = prepared / "stats" / "repro-soma30-30fps"
        stats_metadata = stats / "stats.metadata.json"
        if stats.exists() and not stats_metadata.is_file():
            raise PipelineError(f"incomplete stats directory requires review: {stats}")
        if not stats.exists():
            _run(
                [
                    sys.executable,
                    "-m",
                    "kimodo.training.stats_cli",
                    "--manifest",
                    str(cached),
                    "--output",
                    str(stats),
                    "--split",
                    "train",
                    "--skeleton-joints",
                    "30",
                    "--fps",
                    "30",
                    "--min-frames",
                    "2",
                    "--num-workers",
                    str(pipeline.stats_workers),
                ]
            )
        else:
            record = json.loads(stats_metadata.read_text(encoding="utf-8"))
            if record.get("manifest_sha256") != _sha256(cached):
                raise PipelineError("stats were fitted from a different cached manifest")

        inventory = prepared / "train.cached.references.jsonl"
        inventory_meta = inventory.with_suffix(inventory.suffix + ".metadata.json")
        if plan["reference_inventory"] == "build":
            _run(
                [
                    sys.executable,
                    "-m",
                    "kimodo.training.reference_inventory_cli",
                    "build",
                    "--manifest",
                    str(cached),
                    "--output",
                    str(inventory),
                ]
            )
            _run(
                [
                    sys.executable,
                    "-m",
                    "kimodo.training.reference_inventory_cli",
                    "verify",
                    "--manifest",
                    str(cached),
                    "--inventory",
                    str(inventory),
                ]
            )
        else:
            from kimodo.training.reference_inventory import verify_reference_inventory_full

            # A train-ready receipt is a full content claim. Re-hash derived
            # motion/text assets on reuse instead of trusting only sidecar identity.
            verify_reference_inventory_full(cached, inventory)

        stats_files = _validate_stats_bundle(stats)

        paths_payload = {
            "schema_version": 1,
            "data": {
                "manifest": str(cached),
                "reference_inventory": str(inventory),
            },
            "model": {
                "stats_path": str(stats),
                "checkpoint_dir": None,
                "checkpoint_weights": None,
            },
            "runtime": {
                "output_dir": str(pipeline.run_root / "repro-soma30"),
                "resume": None,
            },
        }
        _atomic_yaml(pipeline.repro_paths_yaml, paths_payload)
        # Schema-5 loading scans every row and validates each motion length and
        # embedding identity.  The preflight then collates one representative
        # CPU batch without allocating the 283M denoiser.
        public_config = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "training"
            / "kimodo_soma_seed_public.yaml"
        )
        _run(
            [
                sys.executable,
                "-m",
                "kimodo.training.cli",
                "--config",
                str(public_config),
                "--paths",
                str(pipeline.repro_paths_yaml),
                "--preflight",
            ]
        )
        receipt = {
            "schema_version": 1,
            "status": "repro_train_ready",
            "catalog_sha256": _sha256(catalog.path),
            "paths_sha256": _sha256(paths.path),
            "flowmatching_producer": flowmatching_identity,
            "data_preflight": "full_manifest_contract_passed",
            "outputs": {
                "cached_manifest_sha256": _sha256(cached),
                "inventory_sha256": _sha256(inventory),
                "inventory_metadata_sha256": _sha256(inventory_meta),
                "stats_files": stats_files,
                "paths_yaml": str(pipeline.repro_paths_yaml),
            },
        }
        _atomic_json(prepared / "resource-state.json", receipt)
        return {"status": "repro_train_ready", "stages": plan, **receipt["outputs"]}
    finally:
        lock.release()
