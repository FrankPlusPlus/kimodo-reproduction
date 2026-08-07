from __future__ import annotations

import hashlib
import json
from argparse import Namespace

from kimodo.training import llm_api_augmentation_cli as llm_api
from kimodo.training.llm_quality_cli import (
    _semantic_risk_flags,
    audit,
    missing_explicit_count_groups,
)
from kimodo.training.qwen_augmentation_cli import _source_preserving_fallback
from kimodo.training.timeline_multi_cli import (
    SYSTEM_PROMPT,
    description_word_limit,
    validate_description,
)
from kimodo.training.v2_manifest_cli import _load_responses


def _requests(tmp_path):
    path = tmp_path / "requests.jsonl"
    rows = []
    actions = ["walks left", "waves right", "steps forward", "turns backward", "stops still"]
    for count in range(2, 6):
        source_texts = actions[:count]
        request_id = hashlib.sha256(json.dumps(source_texts, separators=(",", ":")).encode()).hexdigest()
        rows.append({"request_id": request_id, "source_texts": source_texts})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    path.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps(
            {
                "prompt": {"sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()},
                "output": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    return path, rows


class _FakeApiClient:
    calls = 0

    def __init__(self, *, receipts, **kwargs):
        assert kwargs["api_key"] == "unit-test-secret"
        self.receipts = receipts

    def post(self, payload, *, request_kind, request_ids):
        type(self).calls += 1
        inputs = json.loads(payload["messages"][1]["content"])["items"]
        if request_kind == "generation":
            items = []
            for item in inputs:
                description = "The person first, " + ". Then, ".join(item["source_texts"]) + "."
                items.append({"request_id": item["request_id"], "description": description})
        else:
            items = [{"request_id": item["request_id"], "accepted": True, "reason": "complete"} for item in inputs]
        response_id = f"fake-{type(self).calls}"
        response = {
            "id": response_id,
            "created": 1,
            "model": payload["model"],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            "choices": [{"message": {"content": json.dumps({"items": items})}}],
        }
        self.receipts.write(
            json.dumps(
                {
                    "request_kind": request_kind,
                    "logical_items": len(request_ids),
                    "batch_sha256": "fixture",
                    "response_id": response_id,
                    "created": 1,
                    "response_model": payload["model"],
                    "usage": response["usage"],
                }
            )
            + "\n"
        )
        self.receipts.flush()
        return response


def test_remote_llm_generation_is_resumable_auditable_and_secret_free(tmp_path, monkeypatch):
    requests, request_rows = _requests(tmp_path)
    monkeypatch.setenv("TEST_LLM_KEY", "unit-test-secret")
    monkeypatch.setattr(llm_api, "_ApiClient", _FakeApiClient)
    output = tmp_path / "responses.jsonl"
    metadata = llm_api.generate(
        Namespace(
            requests=str(requests),
            output=str(output),
            base_url="https://example.invalid/v1",
            api_key_env="TEST_LLM_KEY",
            model="mimo-v2.5-pro",
            judge_model="mimo-v2.5-pro",
            revision="provider-managed",
            batch_size=4,
            concurrency=2,
            requests_per_minute=100,
            timeout=10,
            max_retries=0,
            max_completion_tokens=1024,
            judge_max_completion_tokens=512,
            temperature=0.2,
            max_requests=None,
            shard_index=0,
            shard_count=1,
        )
    )
    assert metadata["output"]["entries"] == len(request_rows)
    assert metadata["quality"] == {
        "semantic_fallbacks": 0,
        "repairs_attempted": 0,
        "repairs_succeeded": 0,
        "invalid_published": 0,
    }
    assert metadata["api_receipts"]["api_calls"] == 2
    assert not output.with_suffix(".jsonl.partial").exists()
    combined = b"".join(path.read_bytes() for path in tmp_path.iterdir() if path.is_file())
    assert b"unit-test-secret" not in combined
    loaded, sources = _load_responses(
        [str(output)], expected_model="mimo-v2.5-pro", expected_revision="provider-managed"
    )
    assert len(loaded) == 4
    assert sources[0]["provider"] == "openai_compatible"
    assert sources[0]["producer_identity_sha256"] == metadata["producer_identity_sha256"]

    report_path = tmp_path / "quality.json"
    sample_path = tmp_path / "review.jsonl"
    report = audit(
        Namespace(
            requests=str(requests),
            responses=[str(output)],
            report=str(report_path),
            review_sample=str(sample_path),
            allow_partial=False,
            sample_per_event_count=1,
            max_risk_samples=10,
            sample_seed=7,
            low_lexical_recall=0.45,
            max_fallback_rate=0.02,
            max_duplicate_rate=0.005,
            high_copy_similarity=0.92,
            max_high_copy_rate=1.0,
            input_price_cny_per_million=3.0,
            output_price_cny_per_million=6.0,
        )
    )
    assert report["quality_gate"]["eligible"] is True
    assert report["coverage"] == {
        "requests": 4,
        "responses": 4,
        "missing": 0,
        "unexpected": 0,
    }
    assert report["review_sample"]["entries"] == 4
    assert report["api_accounting"]["normalized_input_tokens"] == 200
    assert report["api_accounting"]["normalized_output_tokens"] == 40


def test_batch_parser_rejects_missing_and_extra_ids():
    expected = ["a", "b"]
    content = json.dumps({"items": [{"request_id": "a", "description": "enough words here"}]})
    try:
        llm_api._strict_items(content, expected, {"description": str})
    except ValueError as error:
        assert "coverage" in str(error)
    else:
        raise AssertionError("batch parser accepted incomplete request coverage")


def test_description_limit_expands_only_for_information_dense_sources():
    short = ["walks left", "waves right"]
    assert description_word_limit(short) == 90
    long = [" ".join(["moves forward"] * 35), " ".join(["turns right"] * 25)]
    assert description_word_limit(long) == 150
    validate_description(long, _source_preserving_fallback(long))


def test_quality_review_semantic_risk_strata_cover_known_failure_modes():
    flags = _semantic_risk_flags(
        [
            "the person picks up a box with the left hand twice",
            "the person turns clockwise",
            "the person picks up a box with the left hand twice",
            "the person walks forward",
        ]
    )
    assert set(flags) == {
        "semantic_direction",
        "semantic_repetition",
        "semantic_explicit_count",
        "semantic_body_part",
        "semantic_object_interaction",
        "semantic_long_sequence",
    }
    assert missing_explicit_count_groups(
        ["the person takes 3 steps and turns once"],
        "The person takes several steps and then turns.",
    ) == ["1:times", "3:steps"]
    assert missing_explicit_count_groups(
        ["the person takes 3 steps and turns once"],
        "The person takes three steps and then turns one time.",
    ) == []
    assert missing_explicit_count_groups(
        ["the person holds an object with two hands and then turns"],
        "The person holds the object with both hands and turns.",
    ) == []
    assert missing_explicit_count_groups(
        ["Once the object is placed, the person turns"],
        "After placing the object, the person turns.",
    ) == []
    assert missing_explicit_count_groups(
        ["the person jumps onto a 2-meter obstacle"],
        "The person jumps onto a 2m obstacle.",
    ) == []


def test_rejected_candidate_gets_repaired_and_rejudged():
    class Client:
        def post(self, payload, *, request_kind, request_ids):
            if request_kind == "generation":
                items = [
                    {
                        "request_id": request_ids[0],
                        "description": "The person walks left and then stands still in place.",
                    }
                ]
            elif request_kind == "semantic_judge":
                items = [{"request_id": request_ids[0], "accepted": False, "reason": "wave omitted"}]
            elif request_kind == "repair_generation":
                items = [
                    {
                        "request_id": request_ids[0],
                        "description": "The person walks left and then waves toward the right.",
                    }
                ]
            else:
                items = [{"request_id": request_ids[0], "accepted": True, "reason": "complete"}]
            return {
                "id": request_kind,
                "choices": [{"message": {"content": json.dumps({"items": items})}}],
            }

    records = llm_api._process_batch(
        [{"request_id": "a", "source_texts": ["walks left", "waves right"]}],
        Client(),
        Namespace(
            model="mimo-v2.5-pro",
            judge_model="mimo-v2.5-pro",
            revision="provider-managed",
            max_completion_tokens=1024,
            judge_max_completion_tokens=512,
            temperature=0.2,
        ),
    )
    assert records[0]["repair_attempted"] is True
    assert records[0]["repair_succeeded"] is True
    assert records[0]["fallback"] is None
    assert records[0]["initial_semantic_judge"]["accepted"] is False
    assert "waves" in records[0]["description"]
