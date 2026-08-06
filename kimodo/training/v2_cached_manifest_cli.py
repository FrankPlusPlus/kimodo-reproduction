# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract and compose V2 cached manifests without re-encoding V1 text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from .file_permissions import publish_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".metadata.json")


def _atomic_publish_jsonl(destination: Path, write_rows) -> tuple[int, str, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or _metadata_path(destination).exists():
        raise FileExistsError(f"Refusing to overwrite output pair: {destination}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            count = write_rows(output)
            output.flush()
            os.fsync(output.fileno())
        digest = _sha256(temporary)
        publish_file(temporary)
        return count, digest, temporary
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _publish_sidecar_then_manifest(destination: Path, temporary: Path, metadata: dict) -> None:
    sidecar = _metadata_path(destination)
    sidecar_temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=sidecar.parent,
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            sidecar_temporary = Path(output.name)
            json.dump(metadata, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        publish_file(sidecar_temporary)
        os.replace(sidecar_temporary, sidecar)
        sidecar_temporary = None
        os.replace(temporary, destination)
    finally:
        if sidecar_temporary is not None:
            sidecar_temporary.unlink(missing_ok=True)
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def extract(args) -> dict:
    source = Path(args.v2_raw_manifest).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    source_metadata_path = _metadata_path(source)
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("output", {}).get("sha256") != _sha256(source):
        raise ValueError("V2 raw manifest hash disagrees with its sidecar")

    def write_rows(output) -> int:
        count = 0
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("sample_kind") in {"timeline_multi_qwen", "timeline_multi_llm"}:
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
        return count

    count, digest, temporary = _atomic_publish_jsonl(destination, write_rows)
    metadata = {
        "schema_version": 3,
        "builder": "kimodo.training.v2_cached_manifest_cli.extract",
        "source_v2_raw": {
            "path": source.name,
            "sha256": _sha256(source),
            "metadata_sha256": _sha256(source_metadata_path),
        },
        "paper_data_recipe": source_metadata.get("paper_data_recipe"),
        "paper_parity_gate": source_metadata.get("paper_parity_gate"),
        "v2_recipe": source_metadata.get("v2_recipe"),
        "leakage_gate": source_metadata.get("leakage_gate"),
        "output": {"path": destination.name, "sha256": digest, "entries": count},
    }
    _publish_sidecar_then_manifest(destination, temporary, metadata)
    return metadata


def _portable_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def compose(args) -> dict:
    v2_raw = Path(args.v2_raw_manifest).expanduser().resolve()
    base = Path(args.v1_cached_manifest).expanduser().resolve()
    llm_manifest = getattr(args, "llm_cached_manifest", None) or getattr(args, "qwen_cached_manifest", None)
    if not llm_manifest:
        raise ValueError("An LLM cached manifest is required")
    llm = Path(llm_manifest).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    destination_root = destination.parent
    raw_metadata_path = _metadata_path(v2_raw)
    base_metadata_path = _metadata_path(base)
    llm_metadata_path = _metadata_path(llm)
    raw_metadata = json.loads(raw_metadata_path.read_text(encoding="utf-8"))
    base_metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
    llm_metadata = json.loads(llm_metadata_path.read_text(encoding="utf-8"))
    for path, metadata in (
        (v2_raw, raw_metadata),
        (base, base_metadata),
        (llm, llm_metadata),
    ):
        if metadata.get("output", {}).get("sha256") != _sha256(path):
            raise ValueError(f"Manifest hash disagrees with sidecar: {path}")
    if base_metadata.get("source_manifest_sha256") != raw_metadata.get("sources", {}).get("v1_raw_manifest", {}).get(
        "sha256"
    ):
        raise ValueError("V1 cached manifest does not belong to the V1 raw source of V2")
    if llm_metadata.get("paper_parity_gate") != raw_metadata.get("paper_parity_gate"):
        raise ValueError("LLM cached manifest lost the V2 paper-parity gate")

    base_cache = destination_root / args.base_cache_dir
    llm_cache_dir = getattr(args, "llm_cache_dir", None) or getattr(args, "qwen_cache_dir", None)
    llm_kinds = set()
    with llm.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                llm_kinds.add(json.loads(line).get("sample_kind"))
    if not llm_kinds or not llm_kinds <= {"timeline_multi_qwen", "timeline_multi_llm"}:
        raise ValueError(f"Unexpected LLM cached sample kinds: {sorted(llm_kinds)}")
    llm_lane = "v2_qwen_multi" if llm_kinds == {"timeline_multi_qwen"} else "v2_llm_multi"
    if llm_cache_dir is None:
        llm_cache_dir = "text-cache-v2-qwen" if llm_lane == "v2_qwen_multi" else "text-cache-v2-llm"
    llm_cache = destination_root / llm_cache_dir
    seen_ids = set()
    validated_files = set()
    counts = Counter()

    def validate_file(path: Path) -> None:
        if path not in validated_files:
            if not path.is_file():
                raise FileNotFoundError(f"V2 cached reference is missing: {path}")
            validated_files.add(path)

    def rewrite_common(row: dict) -> None:
        motion = destination_root / Path(str(row["motion"]))
        validate_file(motion)
        row["motion"] = _portable_path(motion, destination_root)

    def rewrite_embedding(row: dict, cache_root: Path) -> None:
        embedding = cache_root / Path(str(row["text_embedding"])).name
        embedding_metadata = cache_root / Path(str(row["text_embedding_metadata"])).name
        validate_file(embedding)
        validate_file(embedding_metadata)
        row["text_embedding"] = _portable_path(embedding, destination_root)
        row["text_embedding_metadata"] = _portable_path(embedding_metadata, destination_root)

    def write_rows(output) -> int:
        for source, lane, cache_root in (
            (base, "v1_base", base_cache),
            (llm, llm_lane, llm_cache),
        ):
            with source.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    kind = row.get("sample_kind", "full")
                    if lane == "v1_base" and kind == "combined_events":
                        counts["removed_legacy_combined"] += 1
                        continue
                    if lane == "v1_base" and kind not in {"full", "event"}:
                        raise ValueError(f"Unexpected V1 row kind at {source}:{line_number}: {kind}")
                    if lane != "v1_base" and kind not in {"timeline_multi_qwen", "timeline_multi_llm"}:
                        raise ValueError(f"Unexpected LLM row kind at {source}:{line_number}: {kind}")
                    if lane == "v2_qwen_multi" and kind != "timeline_multi_qwen":
                        raise ValueError(f"Qwen compatibility lane contains {kind} at {source}:{line_number}")
                    if lane == "v2_llm_multi" and kind != "timeline_multi_llm":
                        raise ValueError(f"Generic LLM lane contains {kind} at {source}:{line_number}")
                    sample_id = str(row["id"])
                    if sample_id in seen_ids:
                        raise ValueError(f"Cached V2 repeats id: {sample_id}")
                    seen_ids.add(sample_id)
                    rewrite_common(row)
                    rewrite_embedding(row, cache_root)
                    row["mixture_source"] = lane
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    counts[f"output_kind/{kind}"] += 1
        return sum(value for key, value in counts.items() if key.startswith("output_kind/"))

    count, digest, temporary = _atomic_publish_jsonl(destination, write_rows)
    expected = raw_metadata.get("output", {}).get("entries")
    if count != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Cached V2 row count {count} does not match raw V2 {expected}")
    metadata = {
        "schema_version": 5,
        "builder": "kimodo.training.v2_cached_manifest_cli.compose",
        "path_mode": "relative",
        "encoder_identities": {
            "v1_base": base_metadata.get("encoder"),
            llm_lane: llm_metadata.get("encoder"),
        },
        "paper_data_recipe": raw_metadata.get("paper_data_recipe"),
        "paper_parity_gate": raw_metadata.get("paper_parity_gate"),
        "v2_recipe": raw_metadata.get("v2_recipe"),
        "leakage_gate": raw_metadata.get("leakage_gate"),
        "sources": {
            "v2_raw": {"path": v2_raw.name, "sha256": _sha256(v2_raw)},
            "v1_cached": {"path": str(base), "sha256": _sha256(base)},
            "llm_cached": {"path": llm.name, "sha256": _sha256(llm)},
        },
        "counts": dict(sorted(counts.items())),
        "output": {"path": destination.name, "sha256": digest, "entries": count},
    }
    _publish_sidecar_then_manifest(destination, temporary, metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--v2-raw-manifest", required=True)
    extract_parser.add_argument("--output", required=True)
    extract_parser.set_defaults(handler=extract)
    compose_parser = subparsers.add_parser("compose")
    compose_parser.add_argument("--v2-raw-manifest", required=True)
    compose_parser.add_argument("--v1-cached-manifest", required=True)
    cached_group = compose_parser.add_mutually_exclusive_group(required=True)
    cached_group.add_argument("--llm-cached-manifest")
    cached_group.add_argument("--qwen-cached-manifest")
    compose_parser.add_argument("--base-cache-dir", default="text-cache-v1")
    compose_parser.add_argument("--llm-cache-dir")
    compose_parser.add_argument("--qwen-cache-dir")
    compose_parser.add_argument("--output", required=True)
    compose_parser.set_defaults(handler=compose)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = args.handler(args)
    print(json.dumps(metadata["output"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
