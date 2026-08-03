# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache deterministic float32 LLM2Vec embeddings and write a derived JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from kimodo.sanitize import sanitize_texts


def _cache_key(text: str, encoder_identity: str) -> str:
    payload = (encoder_identity + "\0" + text).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_encoder(args):
    if args.provider == "api":
        from kimodo.model.text_encoder_api import TextEncoderAPI

        return TextEncoderAPI(args.api_url), f"api:{args.api_url}"
    from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder

    encoder = LLM2VecEncoder(
        base_model_name_or_path=args.base_model,
        peft_model_name_or_path=args.peft_model,
        dtype="float32",
        llm_dim=4096,
        device=args.device,
        base_revision=args.base_revision,
        peft_revision=args.peft_revision,
    )
    identity = f"llm2vec:{args.base_model}@{args.base_revision}+{args.peft_model}@{args.peft_revision}:float32"
    return encoder, identity


def run(args) -> None:
    source = Path(args.manifest).expanduser().resolve()
    destination = Path(args.output_manifest).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite output manifest: {destination}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    encoder, identity = _build_encoder(args)
    output_lines = []
    metadata = {
        "schema_version": 1,
        "encoder": identity,
        "source_manifest": str(source),
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dtype": "float32",
        "internal_batch_size": 1,
    }
    source_metadata = source.with_suffix(source.suffix + ".metadata.json")
    if source_metadata.is_file():
        metadata["source_manifest_metadata"] = str(source_metadata)
        metadata["source_manifest_metadata_sha256"] = hashlib.sha256(
            source_metadata.read_bytes()
        ).hexdigest()
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
            output_lines.append(json.dumps(entry, ensure_ascii=False, sort_keys=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(output_lines)} entries to {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--provider", choices=("local", "api"), default="local")
    parser.add_argument("--api-url", default="http://127.0.0.1:9550/")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--base-model",
        default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    )
    parser.add_argument(
        "--peft-model",
        default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    )
    parser.add_argument("--base-revision", required=True, help="Pinned model revision recorded in cache identity")
    parser.add_argument("--peft-revision", required=True, help="Pinned PEFT revision recorded in cache identity")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
