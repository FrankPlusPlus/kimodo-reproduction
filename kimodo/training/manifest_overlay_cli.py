# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose a base cached manifest with one small zero-copy augmentation overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path

from .file_permissions import publish_file

SCHEMA_VERSION = 1
_SOURCE_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_PATH_FIELDS = ("motion", "text_embedding", "text_embedding_metadata")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".metadata.json")


def _selected_rows(path: Path, split: str):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split", "")) != split:
                continue
            missing = {"id", "motion", "text", "split", "text_embedding"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing cached fields: {sorted(missing)}")
            yield row


def _rewrite_row(
    row: dict,
    *,
    source_manifest: Path,
    output_parent: Path,
    source_name: str,
    copy_index: int,
) -> dict:
    rewritten = dict(row)
    original_id = str(row["id"])
    rewritten["id"] = f"mix:{source_name}:{copy_index:06d}:{original_id}"
    rewritten["mixture_source"] = source_name
    rewritten["mixture_copy"] = int(copy_index)
    for key in _PATH_FIELDS:
        value = row.get(key)
        if value is None:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = source_manifest.parent / path
        rewritten[key] = _portable(path, output_parent)
    return rewritten


def _source_record(path: Path, output_parent: Path, name: str) -> dict:
    record = {
        "name": name,
        "path": _portable(path, output_parent),
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }
    metadata = _sidecar(path)
    if metadata.is_file():
        record.update(
            metadata_path=_portable(metadata, output_parent),
            metadata_sha256=_sha256(metadata),
        )
    return record


def build_overlay_manifest(args) -> dict:
    base = Path(args.base_manifest).expanduser().resolve()
    overlay = Path(args.overlay_manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    metadata_path = _sidecar(output)
    for path in (base, overlay):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite overlay manifest pair: {output}, {metadata_path}"
        )
    if not _SOURCE_NAME.fullmatch(args.base_name) or not _SOURCE_NAME.fullmatch(
        args.overlay_name
    ):
        raise ValueError("source names must match [a-z][a-z0-9_-]{0,31}")
    if args.base_name == args.overlay_name:
        raise ValueError("base_name and overlay_name must differ")
    fraction = float(args.overlay_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("overlay_fraction must be in [0, 1)")

    base_count = sum(1 for _ in _selected_rows(base, args.split))
    if base_count == 0:
        raise ValueError(f"Base manifest has no split={args.split!r} rows")
    overlay_rows = list(_selected_rows(overlay, args.split))
    if fraction > 0.0 and not overlay_rows:
        raise ValueError(f"Overlay manifest has no split={args.split!r} rows")
    target_overlay_count = (
        0
        if fraction == 0.0
        else max(1, round(fraction * base_count / (1.0 - fraction)))
    )

    randomizer = random.Random(int(args.seed))
    selected_overlay: list[tuple[dict, int]] = []
    cycle = 0
    while len(selected_overlay) < target_overlay_count:
        order = list(range(len(overlay_rows)))
        randomizer.shuffle(order)
        for index in order:
            if len(selected_overlay) >= target_overlay_count:
                break
            selected_overlay.append((overlay_rows[index], cycle))
        cycle += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    seen_ids: set[str] = set()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in _selected_rows(base, args.split):
                rewritten = _rewrite_row(
                    row,
                    source_manifest=base,
                    output_parent=output.parent,
                    source_name=args.base_name,
                    copy_index=0,
                )
                if rewritten["id"] in seen_ids:
                    raise ValueError(f"Duplicate base manifest id: {row['id']}")
                seen_ids.add(rewritten["id"])
                handle.write(json.dumps(rewritten, ensure_ascii=False, sort_keys=True) + "\n")
            for sequence, (row, cycle_index) in enumerate(selected_overlay):
                rewritten = _rewrite_row(
                    row,
                    source_manifest=overlay,
                    output_parent=output.parent,
                    source_name=args.overlay_name,
                    copy_index=cycle_index,
                )
                # Multiple selections from one cycle cannot occur; include the
                # output sequence to make repeated cycles unambiguously unique.
                rewritten["id"] = f"{rewritten['id']}:sample-{sequence:09d}"
                if rewritten["id"] in seen_ids:
                    raise ValueError(f"Duplicate overlay manifest id: {row['id']}")
                seen_ids.add(rewritten["id"])
                handle.write(json.dumps(rewritten, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        publish_file(temporary)
        total = base_count + target_overlay_count
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "builder": "kimodo.training.manifest_overlay_cli",
            "split": args.split,
            "seed": int(args.seed),
            "target_overlay_fraction": fraction,
            "actual_overlay_fraction": target_overlay_count / total,
            "base_entries": base_count,
            "overlay_entries": target_overlay_count,
            "source_manifests": [
                _source_record(base, output.parent, args.base_name),
                _source_record(overlay, output.parent, args.overlay_name),
            ],
            "producer": {
                "path": "kimodo/training/manifest_overlay_cli.py",
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "output": {
                "path": output.name,
                "sha256": _sha256(temporary),
                "entries": total,
            },
        }
        from .text_cache_cli import _atomic_write_json

        _atomic_write_json(metadata_path, metadata)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--overlay-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay-fraction", type=float, required=True)
    parser.add_argument("--base-name", default="base")
    parser.add_argument("--overlay-name", default="dance")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main() -> None:
    metadata = build_overlay_manifest(build_parser().parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
