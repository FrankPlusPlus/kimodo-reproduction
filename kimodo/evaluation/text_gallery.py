"""Stamp a frozen TMR text gallery onto a generated eval tree.

Live text encoding drifts across pods (precision, LLM2Vec weights, device).
Ground-truth R@3 / gen-text FID then stop being comparable. Copy the 750k
parent `text_embedding.npy` files instead of re-encoding prompts.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any


GALLERY_ENV = "KIMODO_EVAL_TEXT_GALLERY"
TEXT_NAME = "text_embedding.npy"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_text_embeddings(root: Path) -> dict[Path, Path]:
    root = root.expanduser().resolve()
    found: dict[Path, Path] = {}
    for path in root.rglob(TEXT_NAME):
        found[path.relative_to(root)] = path
    return found


def gallery_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for rel, path in sorted(discover_text_embeddings(root).items(), key=lambda item: str(item[0])):
        digest.update(str(rel).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def stamp_text_gallery(
    gallery: Path,
    dest: Path,
    *,
    required: bool = True,
) -> dict[str, Any]:
    """Overwrite ``dest/**/text_embedding.npy`` from ``gallery``.

    Every generated sample that has ``meta.json`` must have a matching gallery
    file when ``required`` is true. Extra gallery files are ignored.
    """
    gallery = gallery.expanduser().resolve()
    dest = dest.expanduser().resolve()
    if not gallery.is_dir():
        raise FileNotFoundError(f"TMR text gallery is missing: {gallery}")
    if not dest.is_dir():
        raise FileNotFoundError(f"generated eval tree is missing: {dest}")
    sources = discover_text_embeddings(gallery)
    if not sources:
        raise ValueError(f"TMR text gallery has no {TEXT_NAME} files: {gallery}")
    copied = 0
    missing: list[str] = []
    for meta in dest.rglob("meta.json"):
        rel = (meta.parent / TEXT_NAME).relative_to(dest)
        source = sources.get(rel)
        if source is None:
            missing.append(str(rel))
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    if required and missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"TMR text gallery missing {len(missing)} embeddings for {dest} "
            f"(e.g. {preview})"
        )
    return {
        "gallery": str(gallery),
        "dest": str(dest),
        "copied": copied,
        "missing": missing,
        "gallery_files": len(sources),
        "fingerprint": gallery_fingerprint(gallery),
    }
