# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and atomically publish a benchmark-oriented V2 training bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kimodo.data_pipeline.reference_inventory import verify_reference_inventory_full
from kimodo.resources.pipeline import _atomic_json, _atomic_yaml

from .response_selection_cli import resolve as resolve_response_selection
from .v2_lineage_cli import validate_lineage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} must be inside the V2 building root: {resolved}")
    return resolved


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain one JSON object: {path}")
    return value


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def _validate_manifest(path: Path, expected_entries: int) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(f"V2 cached manifest is missing: {path}")
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    metadata = _load_json(sidecar, "V2 cached manifest sidecar")
    digest = _sha256(path)
    output = metadata.get("output")
    if not isinstance(output, dict) or output.get("sha256") != digest:
        raise ValueError("V2 cached manifest hash disagrees with its sidecar")
    entries = _line_count(path)
    if output.get("entries") != entries or entries != expected_entries:
        raise ValueError(
            f"V2 cached manifest entry count mismatch: sidecar={output.get('entries')!r}, "
            f"actual={entries}, expected={expected_entries}"
        )
    parity = metadata.get("paper_parity_gate")
    if not isinstance(parity, dict) or parity.get("eligible") is not False:
        raise ValueError("V2 cached manifest must retain the honest paper-parity gate")
    leakage = metadata.get("leakage_gate")
    if not isinstance(leakage, dict) or leakage.get("eligible") is not True:
        raise ValueError("V2 cached manifest lacks a passing train-only leakage gate")
    return metadata, digest


def _validate_quality(path: Path, responses: Path) -> dict:
    report = _load_json(path, "LLM quality report")
    gate = report.get("quality_gate")
    coverage = report.get("coverage")
    if not isinstance(gate, dict) or gate.get("eligible") is not True:
        raise ValueError("LLM quality gate is not eligible")
    if not isinstance(coverage, dict) or coverage.get("missing") != 0 or coverage.get("unexpected") != 0:
        raise ValueError("LLM quality report does not have complete request coverage")
    if coverage.get("requests") != coverage.get("responses"):
        raise ValueError("LLM quality report request/response counts differ")
    actual_sha = _sha256(responses)
    response_metadata_path = responses.with_suffix(responses.suffix + ".metadata.json")
    response_metadata = _load_json(response_metadata_path, "LLM response metadata")
    requests_sha256 = response_metadata.get("requests", {}).get("sha256")
    response_sources = report.get("sources", {}).get("responses", [])
    if not isinstance(response_sources, list) or not any(
        isinstance(source, dict)
        and source.get("sha256") == actual_sha
        and source.get("metadata_sha256") == _sha256(response_metadata_path)
        and source.get("producer_identity_sha256")
        == response_metadata.get("producer_identity_sha256")
        and source.get("requests_sha256") == requests_sha256
        for source in response_sources
    ):
        raise ValueError("LLM quality report is stale or belongs to different responses")
    if report.get("sources", {}).get("requests", {}).get("sha256") != requests_sha256:
        raise ValueError("LLM quality report belongs to different requests")
    sample = report.get("review_sample")
    if not isinstance(sample, dict):
        raise TypeError("LLM quality report has no bound review sample")
    sample_path = Path(str(sample.get("path", ""))).expanduser()
    if not sample_path.is_absolute():
        sample_path = path.parent / sample_path
    if not sample_path.is_file() or sample.get("sha256") != _sha256(sample_path):
        raise ValueError("LLM review sample is missing or disagrees with the quality report")
    return report


def _validate_expert_review(
    path: Path,
    verdicts: Path,
    *,
    responses_sha256: str,
    quality_sha256: str,
    review_sample_sha256: str,
) -> dict:
    report = _load_json(path, "independent expert review")
    if report.get("schema_version") != 1 or report.get("status") != "approved":
        raise ValueError("independent expert review has not approved V2")
    bindings = report.get("bindings")
    expected = {
        "responses_sha256": responses_sha256,
        "quality_report_sha256": quality_sha256,
        "review_sample_sha256": review_sample_sha256,
    }
    if not isinstance(bindings, dict) or any(bindings.get(key) != value for key, value in expected.items()):
        raise ValueError("independent expert review is stale or bound to different LLM artifacts")
    if bindings.get("verdicts_sha256") != _sha256(verdicts):
        raise ValueError("independent expert verdict ledger disagrees with its report")
    review = report.get("review")
    if not isinstance(review, dict) or review.get("reviewed_unique_requests", 0) < 1_200:
        raise ValueError("independent expert review covers fewer than 1,200 unique requests")
    if review.get("unresolved_critical_errors", 1) != 0:
        raise ValueError("independent expert review contains unresolved critical errors")
    error_rate = review.get("major_semantic_error_rate")
    if not isinstance(error_rate, (int, float)) or error_rate > 0.005:
        raise ValueError("independent expert review exceeds the semantic error limit")
    return report


def _validate_stats(root: Path, manifest_sha256: str) -> dict:
    metadata = _load_json(root / "stats.metadata.json", "V2 stats metadata")
    if metadata.get("schema_version") != 3 or metadata.get("manifest_sha256") != manifest_sha256:
        raise ValueError("V2 stats are not bound to the finalized cached manifest")
    expected = {
        f"{group}/{name}": dimension
        for group, dimension in (("global_root", 5), ("local_root", 4), ("body", 364))
        for name in ("mean.npy", "std.npy")
    }
    records = metadata.get("files")
    if not isinstance(records, dict) or set(records) != set(expected):
        raise ValueError("V2 stats metadata must bind exactly six arrays")
    for relative, dimension in expected.items():
        path = root / relative
        record = records[relative]
        if not path.is_file() or not isinstance(record, dict):
            raise FileNotFoundError(f"V2 stats array is missing: {path}")
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (dimension,) or not np.isfinite(array).all():
            raise ValueError(f"V2 stats array violates its numeric contract: {path}")
        if (
            record.get("sha256") != _sha256(path)
            or record.get("size") != path.stat().st_size
            or record.get("dtype") != "float32"
            or record.get("shape") != [dimension]
        ):
            raise ValueError(f"V2 stats array disagrees with its metadata: {path}")
    return metadata


def _validate_preflight(
    path: Path,
    expected_entries: int,
    *,
    manifest_sha256: str,
    inventory_sha256: str,
    inventory_metadata_sha256: str,
    stats_metadata_sha256: str,
) -> dict:
    report = _load_json(path, "data preflight report")
    if report.get("event") != "kimodo_full_data_preflight_passed":
        raise ValueError("data preflight did not report success")
    if report.get("manifest_entries_validated") != expected_entries:
        raise ValueError("data preflight did not validate every V2 manifest entry")
    if report.get("dataset_entries") != expected_entries or report.get("excluded_short_entries") != 0:
        raise ValueError("data preflight filtered one or more V2 training entries")
    motion_shape = report.get("motion_shape")
    text_shape = report.get("text_shape")
    if not (
        isinstance(motion_shape, list)
        and len(motion_shape) == 3
        and motion_shape[-2:] == [300, 369]
        and isinstance(text_shape, list)
        and len(text_shape) == 3
        and text_shape[-2:] == [1, 4096]
    ):
        raise ValueError("data preflight returned unexpected motion/text tensor shapes")
    observed = {
        (row.get("mixture_source"), row.get("sample_kind"), row.get("event_count"))
        for row in report.get("sampled_coverage", [])
        if isinstance(row, dict)
    }
    if expected_entries == 1_440_741:
        required = {
            ("v1_base", "full", None),
            ("v1_base", "event", None),
            *(("v2_llm_multi", "timeline_multi_llm", count) for count in range(2, 6)),
        }
        if not required <= observed:
            raise ValueError(
                f"data preflight missed required V2 strata: {sorted(required - observed, key=str)}"
            )
    bindings = report.get("bindings")
    expected_bindings = {
        "manifest_sha256": manifest_sha256,
        "inventory_sha256": inventory_sha256,
        "inventory_metadata_sha256": inventory_metadata_sha256,
        "stats_metadata_sha256": stats_metadata_sha256,
    }
    if not isinstance(bindings, dict) or any(
        bindings.get(key) != value for key, value in expected_bindings.items()
    ):
        raise ValueError("data preflight is stale or bound to different bundle artifacts")
    return report


def _safe_bundle_reference(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    serialized = Path(value)
    if serialized.is_absolute() or ".." in serialized.parts or serialized.as_posix() != value:
        raise ValueError(f"{label} is not a safe portable POSIX path: {value!r}")
    resolved = (root / serialized).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes the V2 bundle: {value!r}")
    return resolved


def _validate_cached_asset_contract(manifest: Path, expected_entries: int) -> dict:
    """Validate lane counts and every embedding's row/sidecar/numeric contract."""
    root = manifest.parent.resolve()
    lane_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    event_counts: Counter[int] = Counter()
    checked_embeddings: dict[Path, str] = {}
    entries = 0
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entries += 1
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"cached manifest row {line_number} is not an object")
            lane = str(row.get("mixture_source", ""))
            kind = str(row.get("sample_kind", ""))
            lane_counts[lane] += 1
            kind_counts[kind] += 1
            if row.get("event_count") is not None:
                event_counts[int(row["event_count"])] += 1
            _safe_bundle_reference(root, row.get("motion"), f"row {line_number} motion")
            embedding = _safe_bundle_reference(
                root, row.get("text_embedding"), f"row {line_number} text_embedding"
            )
            metadata_path = _safe_bundle_reference(
                root,
                row.get("text_embedding_metadata"),
                f"row {line_number} text_embedding_metadata",
            )
            row_sha = row.get("text_embedding_sha256")
            if embedding in checked_embeddings:
                if checked_embeddings[embedding] != row_sha:
                    raise ValueError(f"embedding rows disagree on SHA-256: {embedding}")
                continue
            if not embedding.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"cached embedding pair is missing: {embedding}")
            metadata = _load_json(metadata_path, "text embedding metadata")
            actual_sha = _sha256(embedding)
            array = np.load(embedding, allow_pickle=False, mmap_mode="r")
            if (
                row_sha != actual_sha
                or metadata.get("sha256") != actual_sha
                or metadata.get("dtype") != "float32"
                or metadata.get("shape") != [1, 4096]
                or metadata.get("size") != embedding.stat().st_size
                or array.dtype != np.float32
                or array.shape != (1, 4096)
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"cached embedding violates its numeric/hash contract: {embedding}")
            checked_embeddings[embedding] = actual_sha
    if entries != expected_entries:
        raise ValueError(f"cached asset scan saw {entries} entries, expected {expected_entries}")
    if expected_entries == 1_440_741:
        if lane_counts != Counter({"v1_base": 1_216_852, "v2_llm_multi": 223_889}):
            raise ValueError(f"unexpected final V2 mixture counts: {dict(lane_counts)}")
        if any(event_counts.get(count, 0) == 0 for count in range(2, 6)):
            raise ValueError("V2 LLM lane is missing one or more event-count strata")
    return {
        "entries": entries,
        "unique_embeddings": len(checked_embeddings),
        "mixture_counts": dict(sorted(lane_counts.items())),
        "sample_kind_counts": dict(sorted(kind_counts.items())),
        "event_counts": {str(key): value for key, value in sorted(event_counts.items())},
        "verification": "full_embedding_numeric_and_hash_contract",
    }


def _validate_inventory_paths(root: Path, inventory: Path) -> None:
    with inventory.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            _safe_bundle_reference(root, row.get("path"), f"inventory row {line_number} path")


def _validate_embedding_canary(path: Path) -> dict:
    report = _load_json(path, "V1/V2 embedding numerical canary")
    if report.get("schema_version") != 1 or report.get("passed") is not True:
        raise ValueError("V1/V2 embedding numerical canary did not pass")
    if report.get("sample_count", 0) < 16:
        raise ValueError("V1/V2 embedding numerical canary used fewer than 16 samples")
    observed = report.get("observed")
    if (
        not isinstance(observed, dict)
        or observed.get("max_abs_error", float("inf")) > 1e-4
        or observed.get("min_cosine_similarity", 0.0) < 0.999999
    ):
        raise ValueError("V1/V2 embedding numerical canary exceeds compatibility tolerances")
    return report


def _validate_response_selection(path: Path, responses: Path) -> dict:
    record = _load_json(path, "final response selection")
    requests = (path.parent / record.get("requests", {}).get("path", "")).resolve()
    selected = resolve_response_selection(
        SimpleNamespace(selection=str(path), requests=str(requests))
    )
    if selected != responses:
        raise ValueError("response selection resolves to different responses")
    metadata = responses.with_suffix(responses.suffix + ".metadata.json")
    receipts = responses.with_suffix(responses.suffix + ".api-receipts.jsonl")
    return {
        "selection_sha256": _sha256(path),
        "response_metadata_sha256": _sha256(metadata),
        "api_receipts_sha256": _sha256(receipts),
    }


def publish(args) -> dict:
    building = Path(args.building_root).expanduser().resolve()
    final = Path(args.final_root).expanduser().resolve()
    if not building.is_dir():
        raise FileNotFoundError(f"V2 building root is missing: {building}")
    if building.parent != final.parent or building == final:
        raise ValueError("V2 building and final roots must be distinct siblings")
    if final.exists():
        raise FileExistsError(f"Refusing to replace an existing V2 final bundle: {final}")

    manifest = _inside(building, Path(args.manifest), "manifest")
    if manifest != building / "train.cached.jsonl":
        raise ValueError("V2 publisher only accepts the lineage-checked train.cached.jsonl")
    inventory = _inside(building, Path(args.inventory), "inventory")
    stats = _inside(building, Path(args.stats), "stats")
    quality_path = _inside(building, Path(args.quality_report), "quality report")
    responses_path = _inside(building, Path(args.responses), "LLM responses")
    response_selection_path = _inside(
        building, Path(args.response_selection), "final response selection"
    )
    expert_review_path = _inside(
        building, Path(args.expert_review), "independent expert review"
    )
    embedding_canary_path = _inside(
        building, Path(args.embedding_canary), "embedding numerical canary"
    )
    expert_verdicts_path = _inside(
        building, Path(args.expert_verdicts), "independent expert verdict ledger"
    )
    preflight_path = _inside(building, Path(args.preflight_report), "preflight report")

    manifest_metadata, manifest_sha256 = _validate_manifest(manifest, args.expected_entries)
    response_selection = _validate_response_selection(
        response_selection_path, responses_path
    )
    selected_response_lineage = validate_lineage(building, responses_path, "cached")
    if selected_response_lineage.get("cached_sha256") != manifest_sha256:
        raise ValueError("selected-response lineage ends at a different cached manifest")
    quality = _validate_quality(quality_path, responses_path)
    quality_sha256 = _sha256(quality_path)
    review_sample_sha256 = quality["review_sample"]["sha256"]
    expert_review = _validate_expert_review(
        expert_review_path,
        expert_verdicts_path,
        responses_sha256=_sha256(responses_path),
        quality_sha256=quality_sha256,
        review_sample_sha256=review_sample_sha256,
    )
    embedding_canary = _validate_embedding_canary(embedding_canary_path)
    stats_metadata = _validate_stats(stats, manifest_sha256)
    stats_metadata_sha256 = _sha256(stats / "stats.metadata.json")
    cached_asset_verification = _validate_cached_asset_contract(
        manifest, args.expected_entries
    )
    _validate_inventory_paths(building, inventory)
    inventory_result = verify_reference_inventory_full(manifest, inventory)
    portable_inventory_result = {
        **inventory_result,
        "path": inventory.relative_to(building).as_posix(),
        "metadata_path": inventory.with_suffix(inventory.suffix + ".metadata.json")
        .relative_to(building)
        .as_posix(),
    }
    preflight = _validate_preflight(
        preflight_path,
        args.expected_entries,
        manifest_sha256=manifest_sha256,
        inventory_sha256=inventory_result["sha256"],
        inventory_metadata_sha256=inventory_result["metadata_sha256"],
        stats_metadata_sha256=stats_metadata_sha256,
    )

    relative_manifest = manifest.relative_to(building).as_posix()
    relative_inventory = inventory.relative_to(building).as_posix()
    relative_stats = stats.relative_to(building).as_posix()
    data_root = "${oc.env:KIMODO_DATA_ROOT}"
    run_root = "${oc.env:KIMODO_RUN_ROOT}"
    final_paths = {
        "schema_version": 1,
        "data": {
            "manifest": f"{data_root}/{relative_manifest}",
            "reference_inventory": f"{data_root}/{relative_inventory}",
        },
        "model": {
            "stats_path": f"{data_root}/{relative_stats}",
            "checkpoint_dir": None,
            "checkpoint_weights": None,
        },
        "runtime": {"output_dir": f"{run_root}/v2-1m-production", "resume": None},
    }
    paths_file = building / args.paths_name
    if paths_file.exists():
        raise FileExistsError(f"Refusing to overwrite V2 paths file: {paths_file}")
    _atomic_yaml(paths_file, final_paths)

    receipt = {
        "schema_version": 1,
        "status": "v2_train_ready",
        "recipe": "benchmark-v2-soma30-v2.2",
        "expected_entries": args.expected_entries,
        "outputs": {
            "v2_raw_manifest_sha256": _sha256(building / "train.raw.jsonl"),
            "v2_raw_manifest_metadata_sha256": _sha256(
                building / "train.raw.jsonl.metadata.json"
            ),
            "llm_raw_manifest_sha256": _sha256(building / "train.llm.raw.jsonl"),
            "llm_raw_manifest_metadata_sha256": _sha256(
                building / "train.llm.raw.jsonl.metadata.json"
            ),
            "llm_cached_manifest_sha256": _sha256(
                building / "train.llm.cached.jsonl"
            ),
            "llm_cached_manifest_metadata_sha256": _sha256(
                building / "train.llm.cached.jsonl.metadata.json"
            ),
            "cached_manifest_sha256": manifest_sha256,
            "cached_manifest_metadata_sha256": _sha256(
                manifest.with_suffix(manifest.suffix + ".metadata.json")
            ),
            "inventory_sha256": inventory_result["sha256"],
            "inventory_metadata_sha256": inventory_result["metadata_sha256"],
            "stats_metadata_sha256": stats_metadata_sha256,
            **{
                f"stats_{group}_{name.removesuffix('.npy')}_sha256": _sha256(
                    stats / group / name
                )
                for group in ("global_root", "local_root", "body")
                for name in ("mean.npy", "std.npy")
            },
            "paths_yaml_sha256": _sha256(paths_file),
            "llm_responses_sha256": _sha256(responses_path),
            "llm_response_selection_sha256": response_selection[
                "selection_sha256"
            ],
            "llm_response_metadata_sha256": response_selection[
                "response_metadata_sha256"
            ],
            "llm_api_receipts_sha256": response_selection[
                "api_receipts_sha256"
            ],
            "llm_quality_report_sha256": quality_sha256,
            "expert_review_sha256": _sha256(expert_review_path),
            "expert_verdicts_sha256": _sha256(expert_verdicts_path),
            "embedding_canary_sha256": _sha256(embedding_canary_path),
            "preflight_report_sha256": _sha256(preflight_path),
        },
        "output_paths": {
            "v2_raw_manifest_sha256": "train.raw.jsonl",
            "v2_raw_manifest_metadata_sha256": "train.raw.jsonl.metadata.json",
            "llm_raw_manifest_sha256": "train.llm.raw.jsonl",
            "llm_raw_manifest_metadata_sha256": "train.llm.raw.jsonl.metadata.json",
            "llm_cached_manifest_sha256": "train.llm.cached.jsonl",
            "llm_cached_manifest_metadata_sha256": "train.llm.cached.jsonl.metadata.json",
            "cached_manifest_sha256": manifest.relative_to(building).as_posix(),
            "cached_manifest_metadata_sha256": manifest.with_suffix(
                manifest.suffix + ".metadata.json"
            ).relative_to(building).as_posix(),
            "inventory_sha256": inventory.relative_to(building).as_posix(),
            "inventory_metadata_sha256": inventory.with_suffix(
                inventory.suffix + ".metadata.json"
            ).relative_to(building).as_posix(),
            "stats_metadata_sha256": (stats / "stats.metadata.json")
            .relative_to(building)
            .as_posix(),
            **{
                f"stats_{group}_{name.removesuffix('.npy')}_sha256": (
                    stats / group / name
                )
                .relative_to(building)
                .as_posix()
                for group in ("global_root", "local_root", "body")
                for name in ("mean.npy", "std.npy")
            },
            "paths_yaml_sha256": paths_file.relative_to(building).as_posix(),
            "llm_responses_sha256": responses_path.relative_to(building).as_posix(),
            "llm_response_selection_sha256": response_selection_path
            .relative_to(building)
            .as_posix(),
            "llm_response_metadata_sha256": responses_path.with_suffix(
                responses_path.suffix + ".metadata.json"
            ).relative_to(building).as_posix(),
            "llm_api_receipts_sha256": responses_path.with_suffix(
                responses_path.suffix + ".api-receipts.jsonl"
            ).relative_to(building).as_posix(),
            "llm_quality_report_sha256": quality_path.relative_to(building).as_posix(),
            "expert_review_sha256": expert_review_path.relative_to(building).as_posix(),
            "expert_verdicts_sha256": expert_verdicts_path.relative_to(building).as_posix(),
            "embedding_canary_sha256": embedding_canary_path.relative_to(building).as_posix(),
            "preflight_report_sha256": preflight_path.relative_to(building).as_posix(),
        },
        "quality_gate": quality["quality_gate"],
        "paper_parity_gate": manifest_metadata["paper_parity_gate"],
        "leakage_gate": manifest_metadata["leakage_gate"],
        "stats": {
            "processed_spans": stats_metadata.get("processed_spans"),
            "frame_counts": stats_metadata.get("frame_counts"),
        },
        "reference_verification": portable_inventory_result,
        "cached_asset_verification": cached_asset_verification,
        "selected_response_lineage": selected_response_lineage,
        "expert_review": {
            "status": expert_review["status"],
            "review": expert_review["review"],
        },
        "embedding_canary": embedding_canary,
        "data_preflight": preflight,
    }
    receipt_path = building / "resource-state.json"
    if receipt_path.exists():
        raise FileExistsError(f"Refusing to overwrite V2 resource state: {receipt_path}")
    _atomic_json(receipt_path, receipt)

    os.replace(building, final)
    parent_fd = os.open(final.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return {
        "status": "v2_train_ready",
        "bundle": str(final),
        "paths": str(final / args.paths_name),
        "entries": args.expected_entries,
        "manifest_sha256": manifest_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building-root", required=True)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--response-selection", required=True)
    parser.add_argument("--expert-review", required=True)
    parser.add_argument("--expert-verdicts", required=True)
    parser.add_argument("--embedding-canary", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--expected-entries", type=int, default=1_440_741)
    parser.add_argument("--paths-name", default="repro.paths.yaml")
    return parser


def main() -> None:
    result = publish(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
