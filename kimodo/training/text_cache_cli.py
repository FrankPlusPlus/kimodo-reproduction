# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache deterministic float32 LLM2Vec embeddings and write a derived JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
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


def _encoder_artifacts(args) -> dict[str, dict[str, str]] | None:
    if args.provider != "local":
        return None
    return {
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


def run(args) -> None:
    source = Path(args.manifest).expanduser().resolve()
    destination = Path(args.output_manifest).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite output manifest: {destination}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoder, identity = _build_encoder(args)
    metadata = {
        "schema_version": 2,
        "encoder": identity,
        "source_manifest": str(source),
        "source_manifest_sha256": _sha256_file(source),
        "dtype": "float32",
        "internal_batch_size": 1,
    }
    encoder_artifacts = _encoder_artifacts(args)
    if encoder_artifacts is not None:
        metadata["encoder_artifacts"] = encoder_artifacts
    source_metadata = source.with_suffix(source.suffix + ".metadata.json")
    if source_metadata.is_file():
        metadata["source_manifest_metadata"] = str(source_metadata)
        metadata["source_manifest_metadata_sha256"] = _sha256_file(source_metadata)

    temporary_path = None
    count = 0
    try:
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
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if "text" not in entry:
                        raise ValueError(f"{source}:{line_number} has no text field")
                    sanitized = sanitize_texts([str(entry["text"])])[0]
                    key = _cache_key(sanitized, identity)
                    embedding_path = cache_dir / f"{key}.npy"
                    if not embedding_path.exists():
                        features, lengths = encoder([sanitized])
                        length = int(lengths[0])
                        array = features[0, :length].detach().float().cpu().numpy()
                        np.save(embedding_path, array, allow_pickle=False)
                    entry["text_embedding"] = str(embedding_path)
                    entry["text_cache_key"] = key
                    output.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
            output.flush()
            os.fsync(output.fileno())
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite output manifest: {destination}")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {count} entries to {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--provider", choices=("local", "api"), default="local")
    parser.add_argument("--api-url", default="http://127.0.0.1:9550/")
    parser.add_argument("--device", default="auto")
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
