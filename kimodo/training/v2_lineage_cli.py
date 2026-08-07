# SPDX-License-Identifier: Apache-2.0
"""Verify that every reusable V2 manifest stage descends from selected responses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


STAGES = ("raw", "llm_raw", "llm_cached", "cached")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(path: Path, label: str) -> tuple[dict, Path, str]:
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"{label} output pair is incomplete: {path}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    digest = _sha256(path)
    if payload.get("output", {}).get("sha256") != digest:
        raise ValueError(f"{label} hash disagrees with its sidecar")
    return payload, sidecar, digest


def validate_lineage(
    building_root: str | Path, responses_path: str | Path, through: str = "cached"
) -> dict:
    if through not in STAGES:
        raise ValueError(f"unknown V2 lineage stage: {through}")
    building = Path(building_root).expanduser().resolve()
    responses = Path(responses_path).expanduser().resolve()
    if not responses.is_relative_to(building):
        raise ValueError("selected responses must be inside the V2 building root")
    response_sha = _sha256(responses)
    response_metadata_path = responses.with_suffix(
        responses.suffix + ".metadata.json"
    )
    response_metadata = json.loads(response_metadata_path.read_text(encoding="utf-8"))
    response_metadata_sha = _sha256(response_metadata_path)

    raw = building / "train.raw.jsonl"
    raw_meta, raw_sidecar, raw_sha = _metadata(raw, "V2 raw manifest")
    response_sources = raw_meta.get("sources", {}).get("llm_responses", [])
    matched_sources = [
        source
        for source in response_sources
        if isinstance(source, dict) and source.get("sha256") == response_sha
    ] if isinstance(response_sources, list) else []
    if len(matched_sources) != 1:
        raise ValueError("V2 raw manifest does not descend from selected responses")
    response_source = matched_sources[0]
    expected_response_bindings = {
        "metadata_sha256": response_metadata_sha,
        "producer_identity_sha256": response_metadata.get(
            "producer_identity_sha256"
        ),
        "requests_sha256": response_metadata.get("requests", {}).get("sha256"),
    }
    if any(
        response_source.get(field) != value
        for field, value in expected_response_bindings.items()
    ):
        raise ValueError("V2 raw manifest uses stale selected-response metadata")
    record = {
        "selected_responses_sha256": response_sha,
        "selected_response_metadata_sha256": response_metadata_sha,
        "selected_response_producer_identity_sha256": response_metadata.get(
            "producer_identity_sha256"
        ),
        "selected_response_requests_sha256": response_metadata.get(
            "requests", {}
        ).get("sha256"),
        "v2_raw_sha256": raw_sha,
        "v2_raw_metadata_sha256": _sha256(raw_sidecar),
    }
    if through == "raw":
        return record

    llm_raw = building / "train.llm.raw.jsonl"
    llm_raw_meta, llm_raw_sidecar, llm_raw_sha = _metadata(
        llm_raw, "V2 LLM raw manifest"
    )
    llm_raw_source = llm_raw_meta.get("source_v2_raw", {})
    if (
        llm_raw_source.get("sha256") != raw_sha
        or llm_raw_source.get("metadata_sha256") != _sha256(raw_sidecar)
    ):
        raise ValueError("V2 LLM raw manifest has stale raw-manifest lineage")
    record.update(
        {
            "llm_raw_sha256": llm_raw_sha,
            "llm_raw_metadata_sha256": _sha256(llm_raw_sidecar),
        }
    )
    if through == "llm_raw":
        return record

    llm_cached = building / "train.llm.cached.jsonl"
    llm_cached_meta, llm_cached_sidecar, llm_cached_sha = _metadata(
        llm_cached, "V2 LLM cached manifest"
    )
    if (
        llm_cached_meta.get("source_manifest_sha256") != llm_raw_sha
        or llm_cached_meta.get("source_manifest_metadata_sha256")
        != _sha256(llm_raw_sidecar)
    ):
        raise ValueError("V2 LLM cached manifest has stale LLM-raw lineage")
    record.update(
        {
            "llm_cached_sha256": llm_cached_sha,
            "llm_cached_metadata_sha256": _sha256(llm_cached_sidecar),
        }
    )
    if through == "llm_cached":
        return record

    cached = building / "train.cached.jsonl"
    cached_meta, cached_sidecar, cached_sha = _metadata(cached, "V2 cached manifest")
    sources = cached_meta.get("sources", {})
    if (
        sources.get("v2_raw", {}).get("sha256") != raw_sha
        or sources.get("v2_raw", {}).get("metadata_sha256")
        != _sha256(raw_sidecar)
        or sources.get("llm_cached", {}).get("sha256") != llm_cached_sha
        or sources.get("llm_cached", {}).get("metadata_sha256")
        != _sha256(llm_cached_sidecar)
    ):
        raise ValueError("V2 cached manifest has stale selected-response lineage")
    record.update(
        {
            "cached_sha256": cached_sha,
            "cached_metadata_sha256": _sha256(cached_sidecar),
        }
    )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building-root", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--through", choices=STAGES, default="cached")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        json.dumps(
            validate_lineage(args.building_root, args.responses, args.through),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
