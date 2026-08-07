# SPDX-License-Identifier: Apache-2.0
"""Remediate an independently rejected V2 caption set without hiding the rejection.

The command is deliberately data-contract heavy. ``finalize`` changes only captions
that an immutable adjudication policy confirms as true major errors. ``adjudicate``
then carries the original 1,200-row Qwen review forward, proves deterministic
fallbacks against the ordered sources, and records false-positive resolutions in a
new verdict ledger and report bound to the remediated artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from kimodo.resources.pipeline import _atomic_json

from .file_permissions import publish_file
from .qwen_augmentation_cli import _source_preserving_fallback
from .response_selection_cli import resolve as resolve_selection
from .timeline_multi_cli import validate_description

API_ID_FIELDS = (
    "api_generation_response_id",
    "api_judge_response_id",
    "api_repair_generation_response_id",
    "api_repair_judge_response_id",
)
GENERATOR = "kimodo.training.independent_review_remediation_cli"
PRODUCER_KIND = "composite_independent_review_remediation"
FALLBACK_METHOD = "deterministic_source_preserving_template"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected one JSON object: {path}")
    return value


def _load_rows(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    by_id: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = str(row["request_id"])
            if request_id in by_id:
                raise ValueError(f"duplicate request_id at {path}:{line_number}")
            rows.append(row)
            by_id[request_id] = row
    return rows, by_id


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as sink:
            temporary = Path(sink.name)
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        publish_file(temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _ids_sha256(values: set[str] | list[str]) -> str:
    return _canonical_sha256(sorted(values))


def _policy(path: Path) -> tuple[dict, dict[str, dict]]:
    policy = _load_json(path)
    if policy.get("schema_version") != 1 or not policy.get("policy_id"):
        raise ValueError("unsupported adjudication policy")
    if not isinstance(policy.get("reviewer_identity"), dict):
        raise ValueError("adjudication policy lacks reviewer identity")
    decisions: dict[str, dict] = {}
    for row in policy.get("decisions", []):
        request_id = str(row["request_id"])
        if request_id in decisions:
            raise ValueError(f"duplicate adjudication decision: {request_id}")
        if row.get("adjudication") not in {"true_major", "false_positive"}:
            raise ValueError(f"invalid adjudication for {request_id}")
        if not isinstance(row.get("issue_codes"), list) or not str(
            row.get("evidence", "")
        ).strip():
            raise ValueError(f"adjudication lacks evidence: {request_id}")
        decisions[request_id] = row
    return policy, decisions


def _validate_rejected_review(
    *,
    responses: Path,
    quality: Path,
    sample: Path,
    verdicts: Path,
    report: Path,
) -> tuple[dict, list[dict], dict[str, dict]]:
    value = _load_json(report)
    if value.get("schema_version") != 1 or value.get("status") != (
        "rejected_requires_targeted_repair"
    ):
        raise ValueError("source expert report is not a rejected review")
    expected = {
        "responses_sha256": _sha256(responses),
        "quality_report_sha256": _sha256(quality),
        "review_sample_sha256": _sha256(sample),
        "verdicts_sha256": _sha256(verdicts),
    }
    bindings = value.get("bindings", {})
    if any(bindings.get(key) != digest for key, digest in expected.items()):
        raise ValueError("rejected expert report has stale or broken bindings")
    verdict_rows, verdict_by_id = _load_rows(verdicts)
    if len(verdict_rows) != 1_200:
        raise ValueError("rejected expert verdict ledger must contain exactly 1,200 rows")
    major_ids = {
        request_id
        for request_id, row in verdict_by_id.items()
        if row.get("verdict") == "major"
    }
    reported = set(value.get("review", {}).get("major_request_ids", []))
    if major_ids != reported or value.get("review", {}).get(
        "major_semantic_errors"
    ) != len(major_ids):
        raise ValueError("rejected report major set disagrees with its verdict ledger")
    return value, verdict_rows, verdict_by_id


def finalize(args) -> dict:
    requests = Path(args.requests).expanduser().resolve()
    source = Path(args.source_responses).expanduser().resolve()
    source_selection = Path(args.source_selection).expanduser().resolve()
    source_quality = Path(args.source_quality_report).expanduser().resolve()
    source_sample = Path(args.source_review_sample).expanduser().resolve()
    source_verdicts = Path(args.source_expert_verdicts).expanduser().resolve()
    source_report = Path(args.source_expert_report).expanduser().resolve()
    policy_path = Path(args.policy).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output_metadata = output.with_suffix(output.suffix + ".metadata.json")
    output_receipts = output.with_suffix(output.suffix + ".api-receipts.jsonl")
    ledger = Path(args.ledger).expanduser().resolve()
    ledger_metadata = ledger.with_suffix(ledger.suffix + ".metadata.json")
    destinations = (output, output_metadata, output_receipts, ledger, ledger_metadata)
    if any(path.exists() for path in destinations):
        raise FileExistsError("refusing to overwrite independent-review remediation output")

    selected = resolve_selection(
        SimpleNamespace(selection=str(source_selection), requests=str(requests))
    )
    if selected != source:
        raise ValueError("source selection resolves to a different response file")
    source_metadata_path = source.with_suffix(source.suffix + ".metadata.json")
    source_metadata = _load_json(source_metadata_path)
    rejected, _, verdict_by_id = _validate_rejected_review(
        responses=source,
        quality=source_quality,
        sample=source_sample,
        verdicts=source_verdicts,
        report=source_report,
    )
    request_rows, request_by_id = _load_rows(requests)
    source_rows, source_by_id = _load_rows(source)
    if set(request_by_id) != set(source_by_id):
        raise ValueError("request and source response coverage differs")
    policy, decisions = _policy(policy_path)
    major_ids = set(rejected["review"]["major_request_ids"])
    true_major_ids = {
        request_id
        for request_id, row in decisions.items()
        if row["adjudication"] == "true_major"
    }
    false_positive_ids = set(decisions) - true_major_ids
    if set(decisions) != major_ids or true_major_ids & false_positive_ids:
        raise ValueError("adjudication must partition the exact rejected major set")
    if len(major_ids) != 24 or len(true_major_ids) != 9 or len(false_positive_ids) != 15:
        raise ValueError("unexpected V2 adjudication cardinality")

    policy_sha = _sha256(policy_path)
    source_report_sha = _sha256(source_report)
    ledger_rows: list[dict] = []
    for request_id in sorted(decisions):
        decision = decisions[request_id]
        request = request_by_id[request_id]
        candidate = source_by_id[request_id]
        original_verdict = verdict_by_id[request_id]
        if original_verdict.get("verdict") != "major":
            raise ValueError(f"adjudication targets a non-major verdict: {request_id}")
        ledger_rows.append(
            {
                "schema_version": 1,
                "request_id": request_id,
                "original_verdict": "major",
                "original_verdict_sha256": _canonical_sha256(original_verdict),
                "original_expert_report_sha256": source_report_sha,
                "source_texts_sha256": _canonical_sha256(request["source_texts"]),
                "candidate_description_sha256": hashlib.sha256(
                    str(candidate["description"]).encode("utf-8")
                ).hexdigest(),
                "adjudication": decision["adjudication"],
                "resolution": (
                    FALLBACK_METHOD
                    if decision["adjudication"] == "true_major"
                    else "retain_original_response"
                ),
                "issue_codes": decision["issue_codes"],
                "evidence": decision["evidence"],
                "reviewer_identity": policy["reviewer_identity"],
                "reviewed_at": policy.get("reviewed_at"),
                "policy_id": policy["policy_id"],
                "policy_sha256": policy_sha,
            }
        )
    _atomic_jsonl(ledger, ledger_rows)
    ledger_sha = _sha256(ledger)
    ledger_record = {
        "schema_version": 1,
        "entries": 24,
        "true_major_count": 9,
        "false_positive_count": 15,
        "sorted_request_ids_sha256": _ids_sha256(set(decisions)),
        "true_major_ids_sha256": _ids_sha256(true_major_ids),
        "false_positive_ids_sha256": _ids_sha256(false_positive_ids),
        "ledger_sha256": ledger_sha,
        "policy": {"path": str(policy_path), "sha256": policy_sha},
    }
    _atomic_json(ledger_metadata, ledger_record)

    finalized: list[dict] = []
    changed_description_ids: set[str] = set()
    for source_row in source_rows:
        request_id = str(source_row["request_id"])
        if request_id not in true_major_ids:
            finalized.append(source_row)
            continue
        row = dict(source_row)
        prior_description = str(row["description"])
        fallback = _source_preserving_fallback(request_by_id[request_id]["source_texts"])
        validate_description(request_by_id[request_id]["source_texts"], fallback)
        if fallback != prior_description:
            changed_description_ids.add(request_id)
        prior_api_ids = {field: row.get(field) for field in API_ID_FIELDS}
        for field in API_ID_FIELDS:
            row[field] = None
        row.update(
            {
                "description": fallback,
                "fallback": FALLBACK_METHOD,
                "fallback_reason": "independent multi-expert review confirmed a major semantic omission",
                "deterministic_source_preservation": True,
                "semantic_judge": None,
                "repair_attempted": True,
                "repair_succeeded": False,
                "independent_review_remediation": {
                    "method": FALLBACK_METHOD,
                    "prior_description_sha256": hashlib.sha256(
                        prior_description.encode("utf-8")
                    ).hexdigest(),
                    "prior_api_response_ids": prior_api_ids,
                    "original_verdict_sha256": _canonical_sha256(
                        verdict_by_id[request_id]
                    ),
                    "adjudication_ledger_sha256": ledger_sha,
                    "adjudication_request_id": request_id,
                },
            }
        )
        finalized.append(row)
    if changed_description_ids != true_major_ids:
        raise ValueError("every confirmed true major must change to a new fallback description")
    for source_row, final_row in zip(source_rows, finalized, strict=True):
        request_id = str(source_row["request_id"])
        if request_id not in true_major_ids and final_row != source_row:
            raise AssertionError(f"non-remediated response changed: {request_id}")

    _atomic_jsonl(output, finalized)
    source_receipts = source.with_suffix(source.suffix + ".api-receipts.jsonl")
    receipt_record = source_metadata.get("api_receipts", {})
    if receipt_record.get("sha256") != _sha256(source_receipts):
        raise ValueError("source API receipt ledger is missing or corrupted")
    output_receipts.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = output_receipts.with_name(output_receipts.name + ".tmp")
    if temporary_receipt.exists():
        raise FileExistsError(f"stale receipt staging file: {temporary_receipt}")
    shutil.copyfile(source_receipts, temporary_receipt)
    publish_file(temporary_receipt)
    os.replace(temporary_receipt, output_receipts)

    remediation = {
        "source_response_sha256": _sha256(source),
        "source_response_metadata_sha256": _sha256(source_metadata_path),
        "source_producer_identity_sha256": source_metadata.get(
            "producer_identity_sha256"
        ),
        "source_selection_sha256": _sha256(source_selection),
        "source_quality_report_sha256": _sha256(source_quality),
        "source_review_sample_sha256": _sha256(source_sample),
        "rejected_expert_report_sha256": source_report_sha,
        "rejected_expert_verdicts_sha256": _sha256(source_verdicts),
        "adjudication_ledger": {
            "path": os.path.relpath(ledger, output.parent),
            "sha256": ledger_sha,
            "metadata_path": os.path.relpath(ledger_metadata, output.parent),
            "metadata_sha256": _sha256(ledger_metadata),
            "entries": 24,
        },
        "original_major_count": 24,
        "true_major_fallback_rows": 9,
        "false_positive_retained_rows": 15,
        "resolved_original_major_rows": 24,
        "remaining_unresolved_original_major_rows": 0,
        "true_major_ids_sha256": _ids_sha256(true_major_ids),
        "false_positive_ids_sha256": _ids_sha256(false_positive_ids),
        "changed_description_ids_sha256": _ids_sha256(changed_description_ids),
        "retained_response_ids_sha256": _ids_sha256(false_positive_ids),
        "fallback_method": FALLBACK_METHOD,
        "policy_sha256": policy_sha,
        "requires_bound_post_remediation_expert_review": True,
    }
    producer_identity = {
        "kind": PRODUCER_KIND,
        "version": 1,
        "source_producer_identity_sha256": source_metadata.get(
            "producer_identity_sha256"
        ),
        "remediation": remediation,
    }
    metadata = dict(source_metadata)
    metadata.update(
        {
            "schema_version": 5,
            "generator": GENERATOR,
            "producer_identity": producer_identity,
            "producer_identity_sha256": _canonical_sha256(producer_identity),
            "output": {
                "path": output.name,
                "sha256": _sha256(output),
                "entries": len(finalized),
            },
            "api_receipts": {
                **receipt_record,
                "path": output_receipts.name,
                "sha256": _sha256(output_receipts),
            },
            "quality": {
                **source_metadata.get("quality", {}),
                "invalid_published": sum(
                    bool(row.get("error")) or not bool(row.get("description"))
                    for row in finalized
                ),
                "semantic_fallbacks": sum(
                    row.get("fallback") == FALLBACK_METHOD for row in finalized
                ),
            },
            "independent_review_remediation": remediation,
            "requests": {
                "path": os.path.relpath(requests, output.parent),
                "sha256": _sha256(requests),
            },
        }
    )
    _atomic_json(output_metadata, metadata)
    return {
        "responses": str(output),
        "entries": len(finalized),
        "true_major_fallback_rows": 9,
        "false_positive_retained_rows": 15,
        "responses_sha256": _sha256(output),
        "ledger_sha256": ledger_sha,
    }


def adjudicate(args) -> dict:
    responses = Path(args.responses).expanduser().resolve()
    quality = Path(args.quality_report).expanduser().resolve()
    sample = Path(args.review_sample).expanduser().resolve()
    source_sample = Path(args.source_review_sample).expanduser().resolve()
    source_verdicts = Path(args.source_expert_verdicts).expanduser().resolve()
    source_report = Path(args.source_expert_report).expanduser().resolve()
    supplemental_verdicts = Path(args.supplemental_expert_verdicts).expanduser().resolve()
    supplemental_report = Path(args.supplemental_expert_report).expanduser().resolve()
    ledger = Path(args.ledger).expanduser().resolve()
    output_verdicts = Path(args.verdicts).expanduser().resolve()
    output_report = Path(args.output).expanduser().resolve()
    if output_verdicts.exists() or output_report.exists():
        raise FileExistsError("refusing to overwrite adjudicated expert outputs")

    response_metadata = _load_json(
        responses.with_suffix(responses.suffix + ".metadata.json")
    )
    remediation = response_metadata.get("independent_review_remediation", {})
    if remediation.get("adjudication_ledger", {}).get("sha256") != _sha256(ledger):
        raise ValueError("response metadata is not bound to the adjudication ledger")
    quality_report = _load_json(quality)
    if quality_report.get("quality_gate", {}).get("eligible") is not True:
        raise ValueError("post-remediation deterministic quality gate is not eligible")
    if quality_report.get("review_sample", {}).get("sha256") != _sha256(sample):
        raise ValueError("post-remediation sample disagrees with quality report")
    if not any(
        isinstance(row, dict) and row.get("sha256") == _sha256(responses)
        for row in quality_report.get("sources", {}).get("responses", [])
    ):
        raise ValueError("post-remediation quality report is stale")

    source_quality = Path(args.source_quality_report).expanduser().resolve()
    _validate_rejected_review(
        responses=Path(args.source_responses).expanduser().resolve(),
        quality=source_quality,
        sample=source_sample,
        verdicts=source_verdicts,
        report=source_report,
    )
    supplemental = _load_json(supplemental_report)
    supplemental_bindings = supplemental.get("bindings", {})
    supplemental_expected = {
        "responses_sha256": _sha256(responses),
        "quality_report_sha256": _sha256(quality),
        "review_sample_sha256": _sha256(sample),
        "verdicts_sha256": _sha256(supplemental_verdicts),
    }
    if any(
        supplemental_bindings.get(key) != value
        for key, value in supplemental_expected.items()
    ):
        raise ValueError("supplemental Qwen review has stale or broken bindings")
    if supplemental.get("reviewer") != _load_json(source_report).get("reviewer"):
        raise ValueError("supplemental review used a different Qwen reviewer contract")

    new_rows, new_by_id = _load_rows(sample)
    old_rows, old_by_id = _load_rows(source_sample)
    _, old_verdict_by_id = _load_rows(source_verdicts)
    _, supplemental_by_id = _load_rows(supplemental_verdicts)
    ledger_rows, ledger_by_id = _load_rows(ledger)
    added_ids = set(new_by_id) - set(old_by_id)
    removed_ids = set(old_by_id) - set(new_by_id)
    if len(new_rows) != 1_200 or len(added_ids) != len(removed_ids) or len(added_ids) > 8:
        raise ValueError("post-remediation sample drift exceeds the bounded supplemental review")
    if len(ledger_rows) != 24:
        raise ValueError("adjudication ledger must cover exactly 24 original majors")

    true_major = {
        request_id
        for request_id, row in ledger_by_id.items()
        if row.get("adjudication") == "true_major"
    }
    false_positive = set(ledger_by_id) - true_major
    verdict_rows: list[dict] = []
    for source_row in new_rows:
        request_id = str(source_row["request_id"])
        original = old_verdict_by_id.get(request_id)
        if request_id in true_major:
            if original is None:
                raise ValueError(f"remediated major lacks its original verdict: {request_id}")
            expected = _source_preserving_fallback(source_row["source_texts"])
            if source_row.get("description") != expected:
                raise ValueError(
                    f"remediated major is not exact source-preserving fallback: {request_id}"
                )
            verdict = {
                "verdict": "pass",
                "categories": [],
                "reason": "Exact deterministic source equivalence preserves every ordered source segment.",
                "review_method": "deterministic_source_equivalence",
            }
        elif request_id in false_positive:
            if original is None:
                raise ValueError(f"false positive lacks its original verdict: {request_id}")
            if source_row != old_by_id[request_id]:
                raise ValueError(f"false-positive response changed in review sample: {request_id}")
            verdict = {
                "verdict": "pass",
                "categories": [],
                "reason": ledger_by_id[request_id]["evidence"],
                "review_method": "independent_multi_expert_adjudication",
            }
        else:
            if request_id in added_ids:
                supplemental_verdict = supplemental_by_id.get(request_id)
                if supplemental_verdict is None or supplemental_verdict.get(
                    "verdict"
                ) not in {"pass", "minor"}:
                    raise ValueError(f"supplemental Qwen review did not clear: {request_id}")
                original = supplemental_verdict
                verdict = {
                    "verdict": original["verdict"],
                    "categories": original.get("categories", []),
                    "reason": original["reason"],
                    "review_method": "supplemental_qwen_review",
                }
                verdict_rows.append(
                    {
                        "request_id": request_id,
                        "event_count": source_row["event_count"],
                        "risk_flags": source_row.get("risk_flags", []),
                        "plan_reuse_count": source_row.get("plan_reuse_count", 1),
                        "format_retry_count": original.get("format_retry_count", 0),
                        "original_qwen_verdict": original["verdict"],
                        "original_qwen_verdict_sha256": _canonical_sha256(original),
                        **verdict,
                    }
                )
                continue
            if source_row != old_by_id[request_id]:
                raise ValueError(f"unadjudicated review row changed: {request_id}")
            if original is None:
                raise ValueError(f"unchanged sample lacks its original verdict: {request_id}")
            if original.get("verdict") == "major":
                raise ValueError(f"unresolved original major: {request_id}")
            verdict = {
                "verdict": original["verdict"],
                "categories": original.get("categories", []),
                "reason": original["reason"],
                "review_method": "carried_forward_unchanged_qwen_review",
            }
        verdict_rows.append(
            {
                "request_id": request_id,
                "event_count": source_row["event_count"],
                "risk_flags": source_row.get("risk_flags", []),
                "plan_reuse_count": source_row.get("plan_reuse_count", 1),
                "format_retry_count": original.get("format_retry_count", 0),
                "original_qwen_verdict": original["verdict"],
                "original_qwen_verdict_sha256": _canonical_sha256(original),
                **verdict,
            }
        )
    _atomic_jsonl(output_verdicts, verdict_rows)
    counts = Counter(row["verdict"] for row in verdict_rows)
    category_counts = Counter(
        category for row in verdict_rows for category in row["categories"]
    )
    report = {
        "schema_version": 1,
        "status": "approved",
        "reviewer": {
            "kind": "composite_qwen_and_independent_multi_expert_adjudication",
            "source_rejected_report_sha256": _sha256(source_report),
            "source_qwen_verdicts_sha256": _sha256(source_verdicts),
            "supplemental_qwen_report_sha256": _sha256(supplemental_report),
            "supplemental_qwen_verdicts_sha256": _sha256(supplemental_verdicts),
            "adjudication_ledger_sha256": _sha256(ledger),
            "supplemental_qwen_request_ids_sha256": _ids_sha256(added_ids),
            "policy": "unchanged rows retain Qwen verdict; true majors require exact source equivalence; false positives require ledger evidence",
        },
        "bindings": {
            "responses_sha256": _sha256(responses),
            "quality_report_sha256": _sha256(quality),
            "review_sample_sha256": _sha256(sample),
            "verdicts_sha256": _sha256(output_verdicts),
        },
        "review": {
            "reviewed_unique_requests": len(verdict_rows),
            "verdict_counts": dict(sorted(counts.items())),
            "category_counts": dict(sorted(category_counts.items())),
            "major_semantic_errors": 0,
            "major_semantic_error_rate": 0.0,
            "critical_major_errors": 0,
            "unresolved_critical_errors": 0,
            "format_retry_requests": sum(
                bool(row.get("format_retry_count")) for row in verdict_rows
            ),
            "major_request_ids": [],
        },
        "remediation": {
            "original_major_count": 24,
            "true_major_fallback_rows": len(true_major),
            "false_positive_retained_rows": len(false_positive),
            "resolved_original_major_rows": len(ledger_rows),
            "remaining_unresolved_original_major_rows": 0,
            "true_major_ids_sha256": _ids_sha256(true_major),
            "false_positive_ids_sha256": _ids_sha256(false_positive),
            "adjudication_ledger_sha256": _sha256(ledger),
        },
        "deterministic_quality_gate": quality_report["quality_gate"],
    }
    _atomic_json(output_report, report)
    return report["review"]


def prepare_supplement(args) -> dict:
    sample = Path(args.review_sample).expanduser().resolve()
    source_sample = Path(args.source_review_sample).expanduser().resolve()
    source_verdicts = Path(args.source_expert_verdicts).expanduser().resolve()
    output_partial = Path(args.output_partial).expanduser().resolve()
    if output_partial.exists():
        raise FileExistsError(f"refusing to overwrite supplemental partial: {output_partial}")
    new_rows, new_by_id = _load_rows(sample)
    _, old_by_id = _load_rows(source_sample)
    old_verdict_rows, old_verdict_by_id = _load_rows(source_verdicts)
    if len(new_rows) != 1_200 or set(old_by_id) != set(old_verdict_by_id):
        raise ValueError("source review artifacts do not have exact 1,200-row coverage")
    overlap = set(new_by_id) & set(old_by_id)
    missing = set(new_by_id) - overlap
    if not missing or len(missing) > 8:
        raise ValueError("supplemental review must contain between one and eight new IDs")
    carried = [
        row for row in old_verdict_rows if str(row["request_id"]) in overlap
    ]
    _atomic_jsonl(output_partial, carried)
    return {
        "carried_forward_rows": len(carried),
        "supplemental_rows": len(missing),
        "supplemental_request_ids": sorted(missing),
        "partial": str(output_partial),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("finalize")
    create.add_argument("--requests", required=True)
    create.add_argument("--source-responses", required=True)
    create.add_argument("--source-selection", required=True)
    create.add_argument("--source-quality-report", required=True)
    create.add_argument("--source-review-sample", required=True)
    create.add_argument("--source-expert-verdicts", required=True)
    create.add_argument("--source-expert-report", required=True)
    create.add_argument("--policy", required=True)
    create.add_argument("--ledger", required=True)
    create.add_argument("--output", required=True)
    review = commands.add_parser("adjudicate")
    review.add_argument("--responses", required=True)
    review.add_argument("--quality-report", required=True)
    review.add_argument("--review-sample", required=True)
    review.add_argument("--source-responses", required=True)
    review.add_argument("--source-quality-report", required=True)
    review.add_argument("--source-review-sample", required=True)
    review.add_argument("--source-expert-verdicts", required=True)
    review.add_argument("--source-expert-report", required=True)
    review.add_argument("--supplemental-expert-verdicts", required=True)
    review.add_argument("--supplemental-expert-report", required=True)
    review.add_argument("--ledger", required=True)
    review.add_argument("--verdicts", required=True)
    review.add_argument("--output", required=True)
    supplement = commands.add_parser("prepare-supplement")
    supplement.add_argument("--review-sample", required=True)
    supplement.add_argument("--source-review-sample", required=True)
    supplement.add_argument("--source-expert-verdicts", required=True)
    supplement.add_argument("--output-partial", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "finalize":
        result = finalize(args)
    elif args.command == "adjudicate":
        result = adjudicate(args)
    else:
        result = prepare_supplement(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
