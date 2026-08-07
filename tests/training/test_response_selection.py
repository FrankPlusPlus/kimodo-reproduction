from __future__ import annotations

import argparse
import hashlib
import json

import pytest

from kimodo.training.response_selection_cli import resolve, select


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    requests = tmp_path / "requests.jsonl"
    responses = tmp_path / "responses.jsonl"
    requests.write_text('{"request_id":"one"}\n', encoding="utf-8")
    responses.write_text(
        '{"request_id":"one","description":"motion","model":"m","revision":"r"}\n',
        encoding="utf-8",
    )
    receipts = responses.with_suffix(".jsonl.api-receipts.jsonl")
    receipts.write_text('{"response_id":"one"}\n', encoding="utf-8")
    producer_identity = {"kind": "test"}
    producer_identity_sha256 = hashlib.sha256(
        json.dumps(
            producer_identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema_version": 4,
        "generator": "kimodo.training.semantic_response_finalize_cli",
        "output": {"sha256": _sha(responses), "entries": 1},
        "requests": {"sha256": _sha(requests)},
        "producer_identity": producer_identity,
        "producer_identity_sha256": producer_identity_sha256,
        "api_receipts": {"sha256": _sha(receipts)},
        "quality": {"invalid_published": 0},
        "semantic_finalization": {
            "remaining_expert_required_count_facts": 0,
            "requests_sha256": _sha(requests),
        },
    }
    responses.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return requests, responses


def test_selection_resolves_only_hash_bound_final_response(tmp_path):
    requests, responses = _fixture(tmp_path)
    selection = tmp_path / "selected.json"
    select(
        argparse.Namespace(
            requests=str(requests), responses=str(responses), output=str(selection)
        )
    )
    assert resolve(
        argparse.Namespace(selection=str(selection), requests=str(requests))
    ) == responses.resolve()


def test_selection_detects_response_mutation(tmp_path):
    requests, responses = _fixture(tmp_path)
    selection = tmp_path / "selected.json"
    select(
        argparse.Namespace(
            requests=str(requests), responses=str(responses), output=str(selection)
        )
    )
    responses.write_text('{}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        resolve(argparse.Namespace(selection=str(selection), requests=str(requests)))


def test_selection_rejects_unfinished_semantic_finalization(tmp_path):
    requests, responses = _fixture(tmp_path)
    metadata_path = responses.with_suffix(".jsonl.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["semantic_finalization"]["remaining_expert_required_count_facts"] = 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        select(
            argparse.Namespace(
                requests=str(requests),
                responses=str(responses),
                output=str(tmp_path / "selected.json"),
            )
        )
