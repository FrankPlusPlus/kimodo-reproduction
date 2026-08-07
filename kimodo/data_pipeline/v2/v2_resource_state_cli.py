# SPDX-License-Identifier: Apache-2.0
"""Verify every content-addressed artifact named by a V2 resource-state receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_resource_state(root_value: str | Path) -> dict:
    root = Path(root_value).expanduser().resolve()
    state = root / "resource-state.json"
    record = json.loads(state.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("status") != "v2_train_ready":
        raise ValueError("resource-state.json does not mark a V2 train-ready bundle")
    outputs = record.get("outputs")
    paths = record.get("output_paths")
    if not isinstance(outputs, dict) or not isinstance(paths, dict) or set(outputs) != set(paths):
        raise ValueError("resource-state output hashes and paths must have identical keys")
    verified = {}
    for field in sorted(outputs):
        value = paths[field]
        relative = PurePosixPath(value) if isinstance(value, str) else None
        if (
            relative is None
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != value
        ):
            raise ValueError(f"unsafe resource-state output path for {field}: {value!r}")
        path = (root / Path(*relative.parts)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"resource-state output is missing: {field} -> {path}")
        digest = _sha256(path)
        if outputs[field] != digest:
            raise ValueError(f"resource-state hash mismatch: {field} -> {path}")
        verified[field] = {"path": value, "sha256": digest}
    return {
        "resource_state_sha256": _sha256(state),
        "verified_outputs": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_resource_state(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
