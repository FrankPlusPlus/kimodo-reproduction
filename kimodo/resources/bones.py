"""Convert released BONES-SEED SOMA Uniform BVH files into Kimodo training NPZs.

This converter belongs to ``kimodo-reproduction`` and uses only this project's
motion loader and skeleton definitions.  It deliberately has no Flow Matching
package or checkout dependency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from kimodo.exports.motion_io import load_motion_file
from kimodo.skeleton import SOMASkeleton30

CONVERSION_REVISION = "bones-seed-soma-uniform-kimodo-reproduction-120to30-v1"

# This is the numerical conversion closure, not a list of every source file in
# the package.  Keep it explicit so a cache identity changes whenever BVH
# parsing, time resampling, T-pose conversion, SOMA77->30 mapping, or a helper
# executed by those paths changes.
_PRODUCER_SOURCE_FILES = (
    "assets.py",
    "exports/bvh.py",
    "exports/motion_io.py",
    "geometry.py",
    "motion_rep/feature_utils.py",
    "motion_rep/feet.py",
    "motion_rep/smooth_root.py",
    "resources/bones.py",
    "skeleton/__init__.py",
    "skeleton/base.py",
    "skeleton/bvh.py",
    "skeleton/definitions.py",
    "skeleton/kinematics.py",
    "skeleton/registry.py",
    "skeleton/transforms.py",
    "tools.py",
)
_PRODUCER_SKELETON_ASSETS = (
    "assets/skeletons/somaskel30/joints.p",
    "assets/skeletons/somaskel77/bvh_joints.p",
    "assets/skeletons/somaskel77/joints.p",
    "assets/skeletons/somaskel77/standard_t_pose_global_offsets_rots.p",
)
_PRODUCER_DEPENDENCIES = ("einops", "numpy", "scipy", "torch")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def motion_converter_identity() -> dict[str, object]:
    """Return the complete numerical producer identity for canonical motion NPZs.

    ``conversion_revision`` names the intended algorithm.  The fingerprint also
    binds the exact implementation files, skeleton tensors and numerical package
    versions, so revision discipline alone cannot silently misattribute an old
    cache to a changed producer.

    The result is cached because this function is called once per motion in each
    conversion worker.  Callers must treat the returned mapping as read-only.
    """

    package_root = Path(__file__).resolve().parents[1]

    def hashes(paths: tuple[str, ...]) -> dict[str, str]:
        result: dict[str, str] = {}
        for relative in paths:
            path = package_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"motion converter producer input is missing: {path}")
            result[relative] = _sha256(path)
        return result

    identity: dict[str, object] = {
        "schema_version": 1,
        "module": "kimodo.resources.bones",
        "conversion_revision": CONVERSION_REVISION,
        "source_files": hashes(_PRODUCER_SOURCE_FILES),
        "skeleton_assets": hashes(_PRODUCER_SKELETON_ASSETS),
        "dependency_versions": {
            name: {
                "distribution": importlib.metadata.version(name),
                "runtime": str(importlib.import_module(name).__version__),
            }
            for name in _PRODUCER_DEPENDENCIES
        },
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    identity["producer_fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    return identity


def _split_keys(path: Path) -> set[str]:
    return {
        line.strip().replace("\\", "/").removesuffix(".bvh").removesuffix(".csv")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _metadata_key(value: str) -> str:
    source = Path(value.replace("\\", "/"))
    try:
        marker = source.parts.index("bvh")
    except ValueError as error:
        raise ValueError(f"metadata motion path has no bvh component: {value!r}") from error
    tail = list(source.parts[marker + 1 :])
    if not tail:
        raise ValueError(f"metadata motion path has no file after bvh: {value!r}")
    tail[-1] = Path(tail[-1]).stem
    return "/".join(tail)


def _validate_arrays(rotations: np.ndarray, roots: np.ndarray) -> None:
    if rotations.ndim != 4 or rotations.shape[1:] != (30, 3, 3):
        raise ValueError(f"local_rot_mats must be [T,30,3,3], got {rotations.shape}")
    if rotations.shape[0] < 2:
        raise ValueError("converted motion must contain at least two frames")
    if roots.shape != (rotations.shape[0], 3):
        raise ValueError(f"root_positions must be [T,3], got {roots.shape}")
    if not np.isfinite(rotations).all() or not np.isfinite(roots).all():
        raise ValueError("converted motion contains non-finite values")
    identity = np.eye(3, dtype=rotations.dtype)
    orthogonality = float(
        np.max(np.abs(np.swapaxes(rotations, -1, -2) @ rotations - identity))
    )
    determinant_error = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
    if orthogonality > 5e-3 or determinant_error > 5e-3:
        raise ValueError(
            "local_rot_mats are not proper SO(3) rotations "
            f"(orthogonality={orthogonality:.3g}, determinant={determinant_error:.3g})"
        )


def _read_cached(path: Path) -> tuple[np.ndarray, np.ndarray, float, dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        missing = {"local_rot_mats", "root_positions", "fps", "source_provenance_json"} - set(
            payload.files
        )
        if missing:
            raise ValueError(f"cached motion is missing fields {sorted(missing)}: {path}")
        rotations = np.asarray(payload["local_rot_mats"])
        roots = np.asarray(payload["root_positions"])
        fps = float(payload["fps"].item())
        provenance = json.loads(str(payload["source_provenance_json"].item()))
    _validate_arrays(rotations, roots)
    return rotations, roots, fps, provenance


def convert_soma_uniform_bvh(
    source: str | Path,
    destination: str | Path,
    *,
    source_fps: float = 120.0,
    target_fps: float = 30.0,
) -> dict[str, object]:
    """Convert one released SOMA Uniform BVH into a canonical Kimodo SOMA30 NPZ."""

    if abs(source_fps - 120.0) > 1e-9 or abs(target_fps - 30.0) > 1e-9:
        raise ValueError("BONES-SEED conversion is fixed to released 120 Hz -> 30 Hz data")
    source_path = Path(source).expanduser().resolve()
    output_path = Path(destination).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_digest = _sha256(source_path)
    producer = motion_converter_identity()
    provenance: dict[str, object] = {
        "schema_version": 1,
        "converter": "kimodo.resources.bones.convert_soma_uniform_bvh",
        "conversion_revision": CONVERSION_REVISION,
        "motion_converter_producer": producer,
        "producer_fingerprint_sha256": producer["producer_fingerprint_sha256"],
        "source_sha256": source_digest,
        "source_fps": source_fps,
        "target_fps": target_fps,
    }
    if output_path.exists():
        rotations, _roots, cached_fps, cached_provenance = _read_cached(output_path)
        if abs(cached_fps - target_fps) > 1e-3:
            raise ValueError(f"cached motion fps is stale: {output_path}")
        if cached_provenance != provenance:
            raise ValueError(f"cached motion provenance is stale: {output_path}")
        return {
            "source": str(source_path),
            "source_sha256": source_digest,
            "cached": str(output_path),
            "cached_sha256": _sha256(output_path),
            "frames": int(rotations.shape[0]),
            "fps": cached_fps,
            "producer_fingerprint_sha256": producer["producer_fingerprint_sha256"],
            "status": "reused",
        }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Resampled motion from .* Hz to .* Hz .*", category=UserWarning
        )
        motion, source_joints = load_motion_file(
            str(source_path), source_fps=source_fps, target_fps=target_fps
        )
    local_rotations = motion["local_rot_mats"].float().cpu()
    if source_joints == 77:
        local_rotations = SOMASkeleton30().from_SOMASkeleton77(local_rotations)
    elif source_joints != 30:
        raise ValueError(f"expected SOMA30 or SOMA77 BVH, got {source_joints} joints")
    rotations = local_rotations.numpy().astype(np.float32, copy=False)
    roots = motion["root_positions"].float().cpu().numpy().astype(np.float32, copy=False)
    _validate_arrays(rotations, roots)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".npz", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        np.savez(
            temporary,
            local_rot_mats=rotations,
            root_positions=roots,
            fps=np.asarray(target_fps, dtype=np.float32),
            source_provenance_json=np.asarray(
                json.dumps(provenance, sort_keys=True, separators=(",", ":"))
            ),
        )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "source": str(source_path),
        "source_sha256": source_digest,
        "cached": str(output_path),
        "cached_sha256": _sha256(output_path),
        "frames": int(rotations.shape[0]),
        "fps": target_fps,
        "producer_fingerprint_sha256": producer["producer_fingerprint_sha256"],
        "status": "converted",
    }


def _convert_task(task: tuple[str, str, float, float]) -> dict[str, object]:
    return convert_soma_uniform_bvh(
        task[0], task[1], source_fps=task[2], target_fps=task[3]
    )


def _initialize_worker(threads: int) -> None:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def prepare_bones_seed(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    metadata = Path(args.metadata).expanduser().resolve()
    split_file = Path(args.split_file).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    inventory = Path(args.inventory).expanduser().resolve()
    if inventory.exists():
        raise FileExistsError(f"refusing to overwrite inventory: {inventory}")
    if abs(args.source_fps - 120.0) > 1e-9 or abs(args.target_fps - 30.0) > 1e-9:
        raise ValueError("BONES-SEED conversion is fixed to source_fps=120 and target_fps=30")
    split_keys = _split_keys(split_file)
    selected: dict[str, Path] = {}
    with metadata.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            value = str(row.get("move_soma_uniform_path") or "").strip()
            if value:
                key = _metadata_key(value)
                if key in split_keys:
                    selected[key] = dataset_root / value
    missing_from_metadata = sorted(split_keys - selected.keys())
    missing_digest = hashlib.sha256(
        json.dumps(missing_from_metadata, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected = (
        args.expected_split_entries,
        args.expected_effective_entries,
        args.expected_missing_sha256,
    )
    if any(value is not None for value in expected):
        if any(value is None for value in expected):
            raise ValueError("all expected split/effective/missing values must be supplied together")
        observed = (len(split_keys), len(selected), missing_digest)
        if observed != expected:
            raise ValueError(
                f"BONES/benchmark pinned coverage changed: expected={expected}, observed={observed}"
            )
    missing_files = sorted(key for key, path in selected.items() if not path.is_file())
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} selected BVH files are absent; first={missing_files[0]}"
        )

    tasks = [
        (
            str(source),
            str(output_root / Path(key).with_suffix(".npz")),
            args.source_fps,
            args.target_fps,
        )
        for key, source in sorted(selected.items())
    ]
    inventory.parent.mkdir(parents=True, exist_ok=True)
    temporary = inventory.with_name(inventory.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary inventory requires review: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as output, ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialize_worker,
            initargs=(args.threads_per_worker,),
        ) as pool:
            for index, result in enumerate(pool.map(_convert_task, tasks, chunksize=1), start=1):
                result["source"] = str(Path(str(result["source"])).relative_to(dataset_root))
                result["cached"] = str(Path(str(result["cached"])).relative_to(output_root))
                output.write(json.dumps(result, sort_keys=True) + "\n")
                if index % 100 == 0:
                    output.flush()
                    print(f"prepared {index}/{len(tasks)} motions", flush=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, inventory)
    finally:
        temporary.unlink(missing_ok=True)

    producer = motion_converter_identity()
    report = {
        "schema_version": 1,
        "conversion_revision": CONVERSION_REVISION,
        "converter_module": "kimodo.resources.bones",
        # Retained for readers of the first local-converter metadata schema.
        "converter_source_sha256": producer["source_files"]["resources/bones.py"],
        "producer_fingerprint_sha256": producer["producer_fingerprint_sha256"],
        "motion_converter_producer": producer,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "metadata": str(metadata),
        "metadata_sha256": _sha256(metadata),
        "split_file": str(split_file),
        "split_sha256": _sha256(split_file),
        "official_split_entries": len(split_keys),
        "effective_entries": len(selected),
        "missing_from_metadata": missing_from_metadata,
        "missing_from_metadata_sha256": missing_digest,
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "inventory": str(inventory),
        "inventory_sha256": _sha256(inventory),
    }
    report_path = inventory.with_suffix(inventory.suffix + ".metadata.json")
    _atomic_json(report_path, report)
    print(f"prepared {len(selected)} motions; report={report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--source-fps", type=float, default=120.0)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--expected-split-entries", type=int)
    parser.add_argument("--expected-effective-entries", type=int)
    parser.add_argument("--expected-missing-sha256")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.threads_per_worker < 1:
        raise SystemExit("--threads-per-worker must be positive")
    prepare_bones_seed(args)


if __name__ == "__main__":
    main()
