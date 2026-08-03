# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache deterministic float32 LLM2Vec embeddings and write a derived JSONL manifest."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

from kimodo.sanitize import sanitize_texts


def _cache_key(text: str, encoder_identity: str) -> str:
    payload = (encoder_identity + "\0" + text).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _embedding_is_valid(path: Path) -> bool:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return bool(
            array.dtype == np.float32
            and array.ndim == 2
            and array.shape[0] >= 1
            and array.shape[1] == 4096
            and np.isfinite(array).all()
        )
    except (OSError, ValueError, EOFError):
        return False


def _atomic_save_embedding(path: Path, array: np.ndarray) -> None:
    if not (
        array.dtype == np.float32
        and array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] == 4096
        and np.isfinite(array).all()
    ):
        raise ValueError(
            f"Encoder returned an invalid embedding: dtype={array.dtype}, shape={array.shape}"
        )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            np.save(output, array, allow_pickle=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@contextlib.contextmanager
def _exclusive_output_lock(path: Path):
    """Hold an NFS-safe exclusive lock for one manifest/sidecar output pair."""
    token = uuid.uuid4().hex
    record = {
        "token": token,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        try:
            owner = path.read_text(encoding="utf-8").strip()
        except OSError:
            owner = "<unreadable>"
        raise FileExistsError(
            f"Text-cache output is locked by another or interrupted task: {path}; "
            f"owner={owner}. Inspect the recorded process before removing a stale lock."
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(record, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        yield record
    finally:
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = None
        if isinstance(current, dict) and current.get("token") == token:
            path.unlink(missing_ok=True)


def _artifact_content_manifest(model_name_or_path: str) -> dict | None:
    """Fingerprint every local model file while excluding HF transfer metadata."""
    root = Path(model_name_or_path).expanduser()
    if not root.is_dir():
        return None
    root = root.resolve()
    files = {}
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or ".cache" in relative.parts:
            continue
        size = path.stat().st_size
        files[str(relative)] = {"sha256": _sha256_file(path), "size": size}
        total_bytes += size
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "total_bytes": total_bytes,
        "files": files,
    }


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dependency_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in (
        "torch",
        "transformers",
        "tokenizers",
        "peft",
        "safetensors",
        "numpy",
        "llm2vec",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _cache_provenance(args) -> dict:
    project_root = Path(__file__).resolve().parents[2]
    implementation_files = {
        "kimodo/sanitize.py": project_root / "kimodo" / "sanitize.py",
        "kimodo/training/text_cache_cli.py": Path(__file__).resolve(),
    }
    for path in sorted((project_root / "kimodo" / "model" / "llm2vec").rglob("*.py")):
        implementation_files[str(path.relative_to(project_root))] = path
    provenance = {
        "repo_git_commit": _git_commit(project_root),
        "implementation_file_sha256": {
            name: _sha256_file(path) for name, path in implementation_files.items()
        },
        "dependency_versions": _dependency_versions(),
        "sanitizer": "kimodo.sanitize.sanitize_texts",
    }
    model_lock = getattr(args, "model_lock", None)
    if model_lock:
        lock_path = Path(model_lock).expanduser().resolve()
        if not lock_path.is_file():
            raise FileNotFoundError(f"Model lock does not exist: {lock_path}")
        provenance["model_lock"] = {
            "path": str(lock_path),
            "sha256": _sha256_file(lock_path),
        }
    return provenance


def _bind_identity(identity: str, encoder_artifacts: dict | None, provenance: dict) -> str:
    content = {}
    for name, record in (encoder_artifacts or {}).items():
        if "content" in record:
            content[name] = record["content"]["sha256"]
    binding = {
        "artifact_content_sha256": content,
        "provenance": provenance,
    }
    payload = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{identity};provenance_sha256={hashlib.sha256(payload).hexdigest()}"


def _build_encoder(args):
    if args.provider == "api":
        from kimodo.model.text_encoder_api import TextEncoderAPI

        return TextEncoderAPI(args.api_url), f"api:{args.api_url}"
    from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder

    encoder = LLM2VecEncoder(
        foundation_model_name_or_path=args.foundation_model,
        base_model_name_or_path=args.mntp_model,
        peft_model_name_or_path=args.supervised_model,
        dtype="float32",
        llm_dim=4096,
        device=args.device,
        foundation_revision=args.foundation_revision,
        base_revision=args.mntp_revision,
        peft_revision=args.supervised_revision,
    )
    identity = (
        "llm2vec:"
        f"foundation={args.foundation_model}@{args.foundation_revision};"
        f"mntp={args.mntp_model}@{args.mntp_revision};"
        f"supervised={args.supervised_model}@{args.supervised_revision};"
        "pooling=mean;dtype=float32;internal_batch_size=1"
    )
    return encoder, identity


def _encoder_artifacts(args) -> dict[str, dict] | None:
    if args.provider != "local":
        return None
    artifacts = {
        "foundation": {
            "model_name_or_path": args.foundation_model,
            "revision": args.foundation_revision,
        },
        "mntp_adapter": {
            "model_name_or_path": args.mntp_model,
            "revision": args.mntp_revision,
        },
        "supervised_adapter": {
            "model_name_or_path": args.supervised_model,
            "revision": args.supervised_revision,
        },
    }
    for record in artifacts.values():
        content = _artifact_content_manifest(record["model_name_or_path"])
        if content is not None:
            record["content"] = content
    return artifacts


def _run_locked(args) -> None:
    source = Path(args.manifest).expanduser().resolve()
    destination = Path(args.output_manifest).expanduser().resolve()
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    existing_outputs = [path for path in (destination, metadata_path) if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite an output manifest/sidecar pair; both paths must be absent. "
            f"Inspect and remove any orphaned derived output before retrying: {existing_outputs}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encode_call_size = int(getattr(args, "encode_call_size", 64))
    if encode_call_size < 1:
        raise ValueError("encode_call_size must be positive")
    encoder_artifacts = _encoder_artifacts(args)
    cache_provenance = _cache_provenance(args)
    encoder, identity = _build_encoder(args)
    identity = _bind_identity(identity, encoder_artifacts, cache_provenance)
    metadata = {
        "schema_version": 3,
        "encoder": identity,
        "source_manifest": str(source),
        "source_manifest_sha256": _sha256_file(source),
        "dtype": "float32",
        "internal_batch_size": 1,
        "outer_encode_call_size": encode_call_size,
        "cache_provenance": cache_provenance,
    }
    if encoder_artifacts is not None:
        metadata["encoder_artifacts"] = encoder_artifacts
    source_metadata = source.with_suffix(source.suffix + ".metadata.json")
    if source_metadata.is_file():
        metadata["source_manifest_metadata"] = str(source_metadata)
        metadata["source_manifest_metadata_sha256"] = _sha256_file(source_metadata)

    temporary_path = None
    count = 0
    embeddings_created = 0
    validated_embeddings = set()

    def flush_records(records: list[dict], output) -> int:
        nonlocal embeddings_created
        prepared = []
        missing: dict[str, tuple[str, Path]] = {}
        for entry in records:
            sanitized = sanitize_texts([str(entry["text"])])[0]
            key = _cache_key(sanitized, identity)
            embedding_path = cache_dir / f"{key}.npy"
            if key not in validated_embeddings and key not in missing:
                if _embedding_is_valid(embedding_path):
                    validated_embeddings.add(key)
                else:
                    missing[key] = (sanitized, embedding_path)
            prepared.append((entry, key, embedding_path))

        if missing:
            keys = list(missing)
            texts = [missing[key][0] for key in keys]
            features, lengths = encoder(texts)
            if len(features) != len(keys) or len(lengths) != len(keys):
                raise ValueError("Encoder batch cardinality does not match requested texts")
            for index, key in enumerate(keys):
                length = int(lengths[index])
                array = features[index, :length].detach().float().cpu().numpy()
                _atomic_save_embedding(missing[key][1], array)
                validated_embeddings.add(key)
                embeddings_created += 1

        for entry, key, embedding_path in prepared:
            entry["text_embedding"] = str(embedding_path)
            entry["text_cache_key"] = key
            output.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        return len(prepared)

    try:
        started = time.perf_counter()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            with source.open("r", encoding="utf-8") as handle:
                pending_records = []
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if "text" not in entry:
                        raise ValueError(f"{source}:{line_number} has no text field")
                    pending_records.append(entry)
                    if len(pending_records) >= encode_call_size:
                        count += flush_records(pending_records, output)
                        pending_records.clear()
                        if count % (encode_call_size * 100) == 0:
                            elapsed = time.perf_counter() - started
                            print(
                                f"Text-cache progress: {count} rows, "
                                f"{len(validated_embeddings)} unique keys validated, "
                                f"{embeddings_created} embeddings created, "
                                f"{count / elapsed:.1f} rows/s",
                                flush=True,
                            )
                if pending_records:
                    count += flush_records(pending_records, output)
            output.flush()
            os.fsync(output.fileno())
        if destination.exists() or metadata_path.exists():
            raise FileExistsError(
                "Output manifest or sidecar appeared while caching; refusing to overwrite it"
            )
        metadata["output"] = {
            "path": str(destination),
            "sha256": _sha256_file(temporary_path),
            "entries": count,
        }
        # Publish provenance first.  Consumers can never observe a completed
        # manifest without its sidecar, even if the process dies between the
        # two atomic replacements.
        _atomic_write_json(metadata_path, metadata)
        if destination.exists():
            raise FileExistsError(f"Output manifest appeared while caching: {destination}")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"Wrote {count} entries to {destination}")


def run(args) -> None:
    destination = Path(args.output_manifest).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with _exclusive_output_lock(lock_path):
        _run_locked(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--provider", choices=("local", "api"), default="local")
    parser.add_argument("--api-url", default="http://127.0.0.1:9550/")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--encode-call-size",
        type=int,
        default=64,
        help=(
            "Number of manifest rows grouped per encoder call. LLM2Vec still uses internal "
            "batch_size=1, so this only removes process/Python overhead and preserves embeddings."
        ),
    )
    parser.add_argument(
        "--model-lock",
        help=(
            "Optional model lock whose SHA-256 is bound into cache identity; local model bytes "
            "are fingerprinted independently, but lock contents are not interpreted by this command"
        ),
    )
    parser.add_argument(
        "--foundation-model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Pinned foundation model repo or local snapshot path",
    )
    parser.add_argument(
        "--mntp-model",
        "--base-model",
        dest="mntp_model",
        default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        help="Pinned MNTP adapter repo or local snapshot path",
    )
    parser.add_argument(
        "--supervised-model",
        "--peft-model",
        dest="supervised_model",
        default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        help="Pinned supervised adapter repo or local snapshot path",
    )
    parser.add_argument(
        "--foundation-revision",
        required=True,
        help="Pinned foundation revision recorded in cache identity",
    )
    parser.add_argument(
        "--mntp-revision",
        "--base-revision",
        dest="mntp_revision",
        required=True,
        help="Pinned MNTP adapter revision recorded in cache identity",
    )
    parser.add_argument(
        "--supervised-revision",
        "--peft-revision",
        dest="supervised_revision",
        required=True,
        help="Pinned supervised adapter revision recorded in cache identity",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
