# SPDX-License-Identifier: Apache-2.0
"""Atomically select and later revalidate the immutable V2 LLM response set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .file_permissions import publish_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def _request_ids(path: Path) -> set[str]:
    result = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            request_id = str(json.loads(line)["request_id"])
            if request_id in result:
                raise ValueError(f"duplicate request_id at {path}:{line_number}")
            result.add(request_id)
    return result


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_independent_review_remediation(
    responses: Path, metadata: dict
) -> dict:
    remediation = metadata.get("independent_review_remediation")
    if not isinstance(remediation, dict):
        raise ValueError("remediated response metadata lacks its review contract")
    expected_counts = {
        "original_major_count": 24,
        "true_major_fallback_rows": 9,
        "false_positive_retained_rows": 15,
        "resolved_original_major_rows": 24,
        "remaining_unresolved_original_major_rows": 0,
    }
    if any(remediation.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("independent-review remediation counts are invalid")
    ledger_record = remediation.get("adjudication_ledger")
    if not isinstance(ledger_record, dict) or ledger_record.get("entries") != 24:
        raise ValueError("independent-review remediation ledger record is invalid")
    ledger = (responses.parent / str(ledger_record.get("path", ""))).resolve()
    ledger_metadata = (
        responses.parent / str(ledger_record.get("metadata_path", ""))
    ).resolve()
    if (
        not ledger.is_file()
        or not ledger_metadata.is_file()
        or ledger_record.get("sha256") != _sha256(ledger)
        or ledger_record.get("metadata_sha256") != _sha256(ledger_metadata)
    ):
        raise ValueError("independent-review remediation ledger is missing or stale")
    ledger_meta = json.loads(ledger_metadata.read_text(encoding="utf-8"))
    if (
        ledger_meta.get("ledger_sha256") != _sha256(ledger)
        or ledger_meta.get("entries") != 24
        or ledger_meta.get("true_major_count") != 9
        or ledger_meta.get("false_positive_count") != 15
    ):
        raise ValueError("independent-review remediation ledger metadata is invalid")
    rows = []
    with ledger.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    ids = [str(row["request_id"]) for row in rows]
    true_major = {
        str(row["request_id"])
        for row in rows
        if row.get("adjudication") == "true_major"
    }
    false_positive = {
        str(row["request_id"])
        for row in rows
        if row.get("adjudication") == "false_positive"
    }
    if (
        len(ids) != 24
        or len(set(ids)) != 24
        or len(true_major) != 9
        or len(false_positive) != 15
        or true_major & false_positive
        or true_major | false_positive != set(ids)
    ):
        raise ValueError("adjudication ledger does not partition 24 unique requests")
    checks = {
        "true_major_ids_sha256": _canonical_sha256(sorted(true_major)),
        "false_positive_ids_sha256": _canonical_sha256(sorted(false_positive)),
    }
    if any(remediation.get(key) != value for key, value in checks.items()):
        raise ValueError("response remediation ID hashes disagree with its ledger")
    if metadata.get("producer_identity", {}).get("kind") != (
        "composite_independent_review_remediation"
    ):
        raise ValueError("unexpected independent-review producer identity")
    return {
        **expected_counts,
        "adjudication_ledger_sha256": _sha256(ledger),
        **checks,
    }


def _validate_response(requests: Path, responses: Path) -> tuple[dict, dict]:
    metadata_path = responses.with_suffix(responses.suffix + ".metadata.json")
    if not requests.is_file() or not responses.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("requests, responses, and response metadata must all exist")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    schema_version = metadata.get("schema_version")
    generator = metadata.get("generator")
    valid_generator = (
        schema_version == 4
        and generator == "kimodo.training.semantic_response_finalize_cli"
    ) or (
        schema_version == 5
        and generator == "kimodo.training.independent_review_remediation_cli"
    )
    if not valid_generator:
        raise ValueError("selected response metadata is not a finalized V2 response")
    output = metadata.get("output", {})
    response_sha = _sha256(responses)
    metadata_sha = _sha256(metadata_path)
    request_sha = _sha256(requests)
    if output.get("sha256") != response_sha:
        raise ValueError("response content disagrees with its metadata")
    if output.get("entries") != _rows(responses) or _rows(requests) != _rows(responses):
        raise ValueError("selected response coverage is incomplete")
    if _request_ids(requests) != _request_ids(responses):
        raise ValueError("selected response request IDs do not exactly cover requests")
    if metadata.get("requests", {}).get("sha256") != request_sha:
        raise ValueError("selected responses were produced from different requests")
    finalization = metadata.get("semantic_finalization", {})
    if finalization.get("remaining_expert_required_count_facts") != 0:
        raise ValueError("semantic finalization is incomplete")
    if finalization.get("requests_sha256") != request_sha:
        raise ValueError("semantic finalization is bound to different requests")
    if metadata.get("quality", {}).get("invalid_published") != 0:
        raise ValueError("selected responses contain an invalid published row")
    if schema_version == 5:
        _validate_independent_review_remediation(responses, metadata)
    producer_identity = metadata.get("producer_identity")
    if not isinstance(producer_identity, dict):
        raise ValueError("selected responses lack a producer identity")
    canonical = json.dumps(
        producer_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if metadata.get("producer_identity_sha256") != hashlib.sha256(canonical).hexdigest():
        raise ValueError("selected producer identity hash is invalid")
    receipts = responses.with_suffix(responses.suffix + ".api-receipts.jsonl")
    receipt_record = metadata.get("api_receipts", {})
    if not receipts.is_file() or receipt_record.get("sha256") != _sha256(receipts):
        raise ValueError("selected API receipt ledger is missing or corrupted")
    return metadata, {
        "responses_sha256": response_sha,
        "responses_metadata_sha256": metadata_sha,
        "requests_sha256": request_sha,
    }


def select(args) -> dict:
    requests = Path(args.requests).expanduser().resolve()
    responses = Path(args.responses).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite response selection: {destination}")
    metadata, bindings = _validate_response(requests, responses)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "status": "selected_final_v2_response",
        "responses": {
            "path": os.path.relpath(responses, destination.parent),
            "sha256": bindings["responses_sha256"],
            "entries": metadata["output"]["entries"],
        },
        "metadata": {
            "path": os.path.relpath(
                responses.with_suffix(responses.suffix + ".metadata.json"),
                destination.parent,
            ),
            "sha256": bindings["responses_metadata_sha256"],
        },
        "requests": {
            "path": os.path.relpath(requests, destination.parent),
            "sha256": bindings["requests_sha256"],
        },
        "producer_identity_sha256": metadata.get("producer_identity_sha256"),
        "semantic_finalization": metadata["semantic_finalization"],
    }
    if metadata.get("schema_version") == 5:
        record["independent_review_remediation"] = (
            _validate_independent_review_remediation(responses, metadata)
        )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as sink:
            temporary = Path(sink.name)
            json.dump(record, sink, indent=2, sort_keys=True)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        publish_file(temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return record


def resolve(args) -> Path:
    selection = Path(args.selection).expanduser().resolve()
    requests = Path(args.requests).expanduser().resolve()
    record = json.loads(selection.read_text(encoding="utf-8"))
    if record.get("status") != "selected_final_v2_response":
        raise ValueError("response selection is not finalized")
    responses = (selection.parent / record["responses"]["path"]).resolve()
    metadata, bindings = _validate_response(requests, responses)
    metadata_path = responses.with_suffix(responses.suffix + ".metadata.json")
    if record["responses"].get("sha256") != bindings["responses_sha256"]:
        raise ValueError("selected response hash changed")
    if record["metadata"].get("sha256") != bindings["responses_metadata_sha256"]:
        raise ValueError("selected response metadata hash changed")
    if (selection.parent / record["metadata"]["path"]).resolve() != metadata_path:
        raise ValueError("selected metadata path is inconsistent")
    if record["requests"].get("sha256") != bindings["requests_sha256"]:
        raise ValueError("selected request hash changed")
    if record.get("producer_identity_sha256") != metadata.get(
        "producer_identity_sha256"
    ):
        raise ValueError("selected producer identity changed")
    if metadata.get("schema_version") == 5:
        expected_remediation = _validate_independent_review_remediation(
            responses, metadata
        )
        if record.get("independent_review_remediation") != expected_remediation:
            raise ValueError("selected independent-review remediation changed")
    return responses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("select")
    create.add_argument("--requests", required=True)
    create.add_argument("--responses", required=True)
    create.add_argument("--output", required=True)
    verify = subparsers.add_parser("resolve")
    verify.add_argument("--selection", required=True)
    verify.add_argument("--requests", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "select":
        print(json.dumps(select(args), indent=2, sort_keys=True))
    else:
        print(resolve(args))


if __name__ == "__main__":
    main()
