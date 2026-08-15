#!/usr/bin/env python3
"""Atomically stage one immutable startup file on node-local storage."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path


def _cache_key(source: Path) -> str:
    stat = source.stat()
    identity = f"{source.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def stage(source: Path, cache_root: Path) -> Path:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    size = source.stat().st_size
    cache_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache_root).free
    if free < size + 256 * 1024 * 1024:
        raise OSError(
            f"not enough free node-local space in {cache_root}: "
            f"need at least {size + 256 * 1024 * 1024}, have {free}"
        )
    target = cache_root / f"{_cache_key(source)}-{source.name}"
    if target.is_file() and target.stat().st_size == size:
        return target
    temporary = cache_root / f".{target.name}.tmp-{os.getpid()}"
    started = time.perf_counter()
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if temporary.stat().st_size != size:
            raise OSError(f"staged file size mismatch for {source}")
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    print(
        f"staged {source} -> {target} bytes={size} elapsed_s={time.perf_counter() - started:.1f}",
        file=sys.stderr,
        flush=True,
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--fallback-root", type=Path)
    args = parser.parse_args()
    try:
        target = stage(args.source, args.cache_root)
    except OSError as error:
        if args.fallback_root is None:
            raise
        print(
            f"primary startup cache unavailable ({error}); falling back to {args.fallback_root}",
            file=sys.stderr,
            flush=True,
        )
        target = stage(args.source, args.fallback_root)
    print(target, flush=True)


if __name__ == "__main__":
    main()
