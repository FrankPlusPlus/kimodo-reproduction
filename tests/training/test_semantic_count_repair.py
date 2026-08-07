from __future__ import annotations

import hashlib
import json
from argparse import Namespace

from kimodo.training import semantic_count_repair_cli as count_repair
from kimodo.training.llm_quality_cli import _count_signatures


class _FakeClient:
    def __init__(self, *, receipts, **kwargs):
        assert kwargs["api_key"] == "test-secret"
        self.receipts = receipts

    def post(self, payload, *, request_kind, request_ids):
        if request_kind == "semantic_count_remediation_generation":
            items = [
                {
                    "request_id": request_id,
                    "description": "The person takes three steps forward and then stops in place.",
                }
                for request_id in request_ids
            ]
        else:
            items = [
                {"request_id": request_id, "accepted": True, "reason": "count preserved"}
                for request_id in request_ids
            ]
        self.receipts.write(
            json.dumps(
                {
                    "request_kind": request_kind,
                    "logical_items": len(request_ids),
                    "response_model": payload["model"],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            )
            + "\n"
        )
        self.receipts.flush()
        return {"id": request_kind, "choices": [{"message": {"content": json.dumps({"items": items})}}]}


def test_semantic_count_repair_changes_only_detected_targets(tmp_path, monkeypatch):
    requests = tmp_path / "requests.jsonl"
    request_rows = [
        {"request_id": "target", "source_texts": ["takes three steps forward", "stops"]},
        {"request_id": "untouched", "source_texts": ["walks forward", "stops"]},
    ]
    requests.write_text(
        "".join(json.dumps(row) + "\n" for row in request_rows), encoding="utf-8"
    )
    responses = tmp_path / "responses.jsonl"
    response_rows = [
        {
            "request_id": "target",
            "description": "The person walks forward several steps and then stops in place.",
        },
        {
            "request_id": "untouched",
            "description": "The person walks forward naturally and then stops in place.",
        },
    ]
    responses.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in response_rows),
        encoding="utf-8",
    )
    receipts = responses.with_suffix(".jsonl.api-receipts.jsonl")
    receipts.write_text("", encoding="utf-8")
    responses.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps(
            {
                "producer_identity_sha256": "fixture",
                "requests": {
                    "sha256": hashlib.sha256(requests.read_bytes()).hexdigest()
                },
                "api_receipts": {
                    "path": str(receipts),
                    "sha256": hashlib.sha256(receipts.read_bytes()).hexdigest(),
                },
                "output": {
                    "sha256": hashlib.sha256(responses.read_bytes()).hexdigest(),
                    "entries": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_API_KEY", "test-secret")
    monkeypatch.setattr(count_repair, "_ApiClient", _FakeClient)
    targets = tmp_path / "targets.jsonl"
    targets.write_text(
        json.dumps(
            {
                "request_id": "target",
                "verdict": "major",
                "required_count_facts": ["3:steps"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "repaired.jsonl"
    result = count_repair.repair(
        Namespace(
            requests=str(requests),
            responses=str(responses),
            output=str(output),
            targets=str(targets),
            api_key_env="TEST_API_KEY",
            base_url="https://example.invalid/v1",
            model="mimo-v2.5-pro",
            judge_model="mimo-v2.5-pro",
            revision="provider-managed",
            batch_size=8,
            concurrency=1,
            temperature=0.25,
            requests_per_minute=90,
            timeout=10,
            max_retries=0,
            max_completion_tokens=512,
            judge_max_completion_tokens=512,
        )
    )
    repaired = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["targeted_requests"] == 1
    assert result["remaining_expert_required_count_facts"] == 0
    assert repaired[0]["description"].startswith("The person takes three steps")
    assert repaired[1] == response_rows[1]
    assert "test-secret" not in b"".join(
        path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    ).decode()


def test_required_count_facts_are_bound_to_their_action_unit():
    assert (4, "gestures") in _count_signatures("arm gestures four times")
    assert count_repair._missing_required_facts(
        ["2:steps"], "The person makes two arm movements."
    ) == ["2:steps"]
    assert count_repair._missing_required_facts(
        ["4:gestures"], "The person takes four steps."
    ) == ["4:gestures"]
    assert count_repair._missing_required_facts(
        ["2:steps"], "The person takes two steady steps."
    ) == []
