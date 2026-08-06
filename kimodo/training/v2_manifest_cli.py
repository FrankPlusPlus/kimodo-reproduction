# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assemble the benchmark-oriented V2 raw manifest from an audited plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from .file_permissions import publish_file
from .qwen_augmentation_cli import JUDGE_PROMPT
from .timeline_multi_cli import (
    SYSTEM_PROMPT,
    _canonical_hash,
    _event_index,
    _motion_key,
    _sha256_file,
    _split_keys,
    validate_description,
)


def _load_responses(
    paths: list[str], *, expected_model: str, expected_revision: str
) -> tuple[dict[str, dict], list[dict]]:
    responses = {}
    sources = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        metadata_path = path.with_suffix(path.suffix + ".metadata.json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"LLM response metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        output_record = metadata.get("output", {})
        if output_record.get("sha256") != _sha256_file(path):
            raise ValueError(f"LLM response hash disagrees with metadata: {path}")
        if metadata.get("model") != expected_model or metadata.get("revision") != expected_revision:
            raise ValueError(f"LLM response uses an unexpected model identity: {path}")
        if metadata.get("prompt_sha256") != hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest():
            raise ValueError(f"LLM response uses an unexpected generation prompt: {path}")
        if metadata.get("judge_prompt_sha256") != hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest():
            raise ValueError(f"LLM response uses an unexpected semantic judge prompt: {path}")
        local_snapshot_sha256 = (metadata.get("local_model_snapshot") or {}).get("aggregate_sha256")
        producer_identity_sha256 = metadata.get("producer_identity_sha256")
        immutable_producer_identity = producer_identity_sha256 or local_snapshot_sha256
        sources.append(
            {
                "path": str(path),
                "sha256": output_record["sha256"],
                "metadata_path": str(metadata_path),
                "metadata_sha256": _sha256_file(metadata_path),
                "requests_sha256": metadata.get("requests", {}).get("sha256"),
                "generator": metadata.get("generator"),
                "provider": metadata.get("provider", "local_transformers"),
                "producer_identity_sha256": immutable_producer_identity,
                "local_model_snapshot_aggregate_sha256": local_snapshot_sha256,
                "model_weight_identity": metadata.get(
                    "model_weight_identity",
                    "local_content_addressed_snapshot" if local_snapshot_sha256 else None,
                ),
            }
        )
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                request_id = str(row["request_id"])
                if request_id in responses:
                    raise ValueError(f"Duplicate LLM response id: {request_id}")
                if row.get("error") or not row.get("description"):
                    raise ValueError(f"Invalid LLM response at {path}:{line_number}: {row.get('error')}")
                if row.get("model") != expected_model or row.get("revision") != expected_revision:
                    raise ValueError(f"Mixed LLM model identities at {path}:{line_number}")
                judge = row.get("semantic_judge")
                accepted_by_judge = isinstance(judge, dict) and judge.get("accepted") is True
                accepted_fallback = bool(
                    row.get("fallback") == "deterministic_source_preserving_template"
                    and row.get("deterministic_source_preservation") is True
                )
                if not accepted_by_judge and not accepted_fallback:
                    raise ValueError(f"LLM response lacks an accepted semantic judge: {path}:{line_number}")
                responses[request_id] = row
    return responses, sources


def build(args) -> dict:
    source = Path(args.source_manifest).expanduser().resolve()
    plan_path = Path(args.plan).expanduser().resolve()
    split = Path(args.train_split).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    sidecar = destination.with_suffix(destination.suffix + ".metadata.json")
    if destination.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite V2 manifest output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    train_keys = _split_keys(split)
    validated_motions = set()

    def validate_bundled_motion(value: str) -> None:
        if value in validated_motions:
            return
        relative = Path(value)
        if relative.is_absolute():
            raise ValueError("V2 bundle requires portable relative motion paths")
        source_motion = (source.parent / relative).resolve()
        bundled_motion = (destination.parent / relative).resolve()
        if not bundled_motion.is_relative_to(destination.parent.resolve()):
            raise ValueError(f"Motion path escapes the V2 bundle: {value!r}")
        if not source_motion.is_file() or not bundled_motion.is_file():
            raise FileNotFoundError(f"V2 motion asset is not present in both source and destination layouts: {value!r}")
        if not os.path.samefile(source_motion, bundled_motion):
            raise ValueError(
                f"V2 build requires audited V1 hardlinks for motion assets: {value!r}; "
                "a copied layout must first provide an independently verified content inventory"
            )
        validated_motions.add(value)

    responses, response_sources = _load_responses(
        args.responses,
        expected_model=args.expected_model,
        expected_revision=args.expected_revision,
    )
    plan_metadata_path = plan_path.with_suffix(plan_path.suffix + ".metadata.json")
    if not plan_metadata_path.is_file():
        raise FileNotFoundError(f"Timeline plan metadata is missing: {plan_metadata_path}")
    plan_metadata = json.loads(plan_metadata_path.read_text(encoding="utf-8"))
    if plan_metadata.get("outputs", {}).get("plan", {}).get("sha256") != _sha256_file(plan_path):
        raise ValueError("Timeline plan hash disagrees with its metadata")
    if plan_metadata.get("source_manifest", {}).get("sha256") != _sha256_file(source):
        raise ValueError("Timeline plan was prepared from a different V1 manifest")
    if plan_metadata.get("official_train_split", {}).get("sha256") != _sha256_file(split):
        raise ValueError("Timeline plan was prepared with a different train whitelist")
    if plan_metadata.get("prompt", {}).get("sha256") != hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest():
        raise ValueError("Timeline plan was prepared for a different LLM semantic prompt")
    expected_requests_sha = plan_metadata.get("outputs", {}).get("requests", {}).get("sha256")
    if any(source_record["requests_sha256"] != expected_requests_sha for source_record in response_sources):
        raise ValueError("LLM responses were generated from different request content")
    producer_hashes = {source_record["producer_identity_sha256"] for source_record in response_sources}
    if len(producer_hashes) != 1 or None in producer_hashes:
        raise ValueError("LLM response shards lack one consistent producer identity")
    qwen_producer = args.expected_model.lower().startswith("qwen/") or args.expected_model.lower().startswith("qwen")
    llm_sample_kind = "timeline_multi_qwen" if qwen_producer else "timeline_multi_llm"

    source_events = {}
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sample_kind") != "event":
                continue
            if row["id"] in source_events:
                raise ValueError(f"V1 source repeats event id {row['id']!r}")
            source_events[row["id"]] = row
    plans = []
    required_requests = set()
    ids = set()
    with plan_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["motion_key"] not in train_keys:
                raise ValueError(f"Plan row {line_number} is outside the official train split")
            if _motion_key(str(row["motion"])) != row["motion_key"]:
                raise ValueError(f"Plan row {line_number} motion path disagrees with motion_key")
            validate_bundled_motion(str(row["motion"]))
            try:
                events = [source_events[event_id] for event_id in row["source_event_ids"]]
            except KeyError as error:
                raise ValueError(f"Plan row {line_number} references an unknown V1 event") from error
            if any(event["motion"] != row["motion"] for event in events):
                raise ValueError(f"Plan row {line_number} mixes source motions")
            if [event["text"] for event in events] != row["source_texts"]:
                raise ValueError(f"Plan row {line_number} source texts disagree with V1")
            event_indices = [_event_index(event) for event in events]
            if event_indices != list(range(event_indices[0], event_indices[0] + len(events))):
                raise ValueError(f"Plan row {line_number} source events are not consecutive")
            ranges = [
                [
                    max(0, round(float(event["start_time"]) * float(event["source_fps"]))),
                    min(
                        int(event["frame_count"]),
                        round(float(event["end_time"]) * float(event["source_fps"])),
                    ),
                ]
                for event in events
            ]
            if ranges != row["source_time_ranges"]:
                raise ValueError(f"Plan row {line_number} source ranges disagree with V1")
            if row["start_frame"] != ranges[0][0] or row["end_frame"] != ranges[-1][1]:
                raise ValueError(f"Plan row {line_number} span boundary disagrees with V1")
            expected_request = _canonical_hash({"ordered_source_texts": row["source_texts"]})
            request_id = row.get("llm_request_id", row.get("qwen_request_id"))
            if request_id != expected_request:
                raise ValueError(f"Plan row {line_number} has a corrupted request identity")
            if row["id"] in ids:
                raise ValueError(f"Duplicate V2 plan id: {row['id']}")
            ids.add(row["id"])
            required_requests.add(expected_request)
            plans.append(row)
    missing = sorted(required_requests - responses.keys())
    unexpected = sorted(responses.keys() - required_requests)
    if missing or unexpected:
        raise ValueError(f"LLM response coverage mismatch: missing={len(missing)}, unexpected={len(unexpected)}")

    counts = Counter()
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
            with source.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = _motion_key(str(row["motion"]))
                    if row.get("split") != "train" or key not in train_keys:
                        raise ValueError(f"Source row {line_number} violates the train whitelist")
                    validate_bundled_motion(str(row["motion"]))
                    if row.get("sample_kind") == "combined_events":
                        counts["removed_legacy_combined"] += 1
                        continue
                    if row["id"] in ids:
                        raise ValueError(f"V2 id collides with source id: {row['id']}")
                    ids.add(row["id"])
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    counts[f"output_kind/{row.get('sample_kind', 'full')}"] += 1
            for plan in plans:
                request_id = plan.get("llm_request_id", plan.get("qwen_request_id"))
                response = responses[request_id]
                description = " ".join(str(response["description"]).split())
                validate_description(plan["source_texts"], description)
                used_fallback = response.get("fallback") == "deterministic_source_preserving_template"
                row = {
                    "id": plan["id"],
                    "motion": plan["motion"],
                    "text": description,
                    "split": "train",
                    "source_fps": plan["source_fps"],
                    "frame_count": plan["frame_count"],
                    "start_time": plan["start_time"],
                    "end_time": plan["end_time"],
                    "sample_kind": llm_sample_kind,
                    "augmentation_provenance": (
                        "deterministic_source_preserving_fallback_after_llm_rejection"
                        if used_fallback
                        else "llm_rewrite_of_adjacent_same_motion_train_events"
                    ),
                    "source_event_ids": plan["source_event_ids"],
                    "source_time_ranges": plan["source_time_ranges"],
                    "source_texts": plan["source_texts"],
                    "event_count": plan["event_count"],
                    "text_source": (
                        "deterministic_source_preserving_template"
                        if used_fallback
                        else "benchmark_oriented_llm_engineering_prompt"
                    ),
                    "source_text_id": request_id,
                    "text_generator_model": response["model"],
                    "text_generator_revision": response["revision"],
                    "text_generator_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                }
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                counts[f"output_kind/{llm_sample_kind}"] += 1
                counts[f"timeline_multi_events/{plan['event_count']}"] += 1
                if used_fallback:
                    counts["timeline_multi_deterministic_fallback"] += 1
            output.flush()
            os.fsync(output.fileno())
        metadata = {
            "schema_version": 3,
            "builder": "kimodo.training.v2_manifest_cli",
            "v2_recipe": {
                "objective": "public_benchmark_oriented_text_and_constraint_coverage",
                "legacy_combined_policy": "removed_and_replaced_by_natural_2_to_5_event_semantic_spans",
                "overview_single_policy": "preserve_all_v1_dataset_annotations",
                "statistics_policy": "recompute_for_v2_semantic_spans_before_train_ready_publication",
                "llm_prompt_status": "engineering_reconstruction_exact_official_prompt_not_disclosed",
                "transition_generation": "not_implemented_in_v2_text_first_release",
            },
            "paper_data_recipe": {
                "full_motion_clips": "preserved_from_v1",
                "single_action_subclips": "preserved_from_v1",
                "combined_action_subclips": "implemented_as_natural_llm_rewrites_of_2_to_5_adjacent_train_events",
                "language_model_paraphrases": (
                    f"timeline_multi_composition_with_{args.expected_model}_and_an_audited_engineering_prompt"
                ),
                "random_cross_motion_stitching": "not_generated",
                "diffusion_transition_clips": "not_generated",
                "official_mixture_distribution": "not_disclosed_benchmark_multi_distribution_used_as_proxy",
            },
            "paper_parity_gate": {
                "eligible": False,
                "status": "blocked_missing_official_transition_recipe_and_exact_language_recipe",
                "blockers": [
                    "official_qwen_prompt_and_sampling_recipe_not_disclosed",
                    *([] if qwen_producer else ["generation_model_differs_from_paper_qwen3_32b"]),
                    "random_cross_motion_stitching",
                    "diffusion_transition_clips",
                ],
            },
            "leakage_gate": {
                "eligible": True,
                "source_lineage": "v1_train_only",
                "official_train_split_sha256": _sha256_file(split),
                "validated_motion_keys": len({_motion_key(row["motion"]) for row in plans}),
                "out_of_train_rows": 0,
            },
            "sources": {
                "v1_raw_manifest": {"path": str(source), "sha256": _sha256_file(source)},
                "timeline_plan": {"path": str(plan_path), "sha256": _sha256_file(plan_path)},
                "official_train_split": {"path": str(split), "sha256": _sha256_file(split)},
                "llm_responses": response_sources,
            },
            "counts": dict(sorted(counts.items())),
            "output": {
                "path": destination.name,
                "sha256": _sha256_file(temporary),
                "entries": sum(value for key, value in counts.items() if key.startswith("output_kind/")),
            },
        }
        sidecar_temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=sidecar.parent,
                prefix=f".{sidecar.name}.",
                suffix=".tmp",
                delete=False,
            ) as metadata_output:
                sidecar_temporary = Path(metadata_output.name)
                json.dump(metadata, metadata_output, indent=2, sort_keys=True)
                metadata_output.write("\n")
                metadata_output.flush()
                os.fsync(metadata_output.fileno())
            publish_file(sidecar_temporary)
            os.replace(sidecar_temporary, sidecar)
            sidecar_temporary = None
        finally:
            if sidecar_temporary is not None:
                sidecar_temporary.unlink(missing_ok=True)
        publish_file(temporary)
        os.replace(temporary, destination)
        temporary = None
        return metadata
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--responses", nargs="+", required=True)
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--expected-revision", default="9216db5781bf21249d130ec9da846c4624c16137")
    return parser


def main() -> None:
    metadata = build(build_parser().parse_args())
    print(json.dumps(metadata["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
