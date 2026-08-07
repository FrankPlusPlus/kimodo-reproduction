from __future__ import annotations

import hashlib
import json

import pytest

from kimodo.training.expert_review_cli import (
    _parse_verdict,
    _validate_bindings,
)


def test_expert_verdict_parser_accepts_strict_json_and_fences():
    expected = {
        "verdict": "major",
        "categories": ["direction", "order"],
        "reason": "The direction and order changed.",
    }
    assert _parse_verdict(json.dumps(expected)) == expected
    assert _parse_verdict(f"```json\n{json.dumps(expected)}\n```") == expected


@pytest.mark.parametrize(
    "value",
    [
        '{"verdict":"unknown","categories":[],"reason":"bad"}',
        '{"verdict":"pass","categories":[],"reason":""}',
        '{"verdict":"pass","categories":[],"reason":"ok","extra":1}',
    ],
)
def test_expert_verdict_parser_rejects_invalid_contracts(value):
    with pytest.raises((TypeError, ValueError)):
        _parse_verdict(value)


def test_expert_review_bindings_require_1200_unique_hash_bound_rows(tmp_path):
    responses = tmp_path / "responses.jsonl"
    responses.write_text('{"request_id":"source"}\n', encoding="utf-8")
    sample = tmp_path / "review.jsonl"
    with sample.open("w", encoding="utf-8") as handle:
        for index in range(1200):
            handle.write(
                json.dumps(
                    {
                        "request_id": f"request-{index}",
                        "event_count": 2 + index % 4,
                        "source_texts": ["A person walks.", "A person stops."],
                        "description": "A person walks and then stops.",
                    }
                )
                + "\n"
            )
    quality = tmp_path / "quality.json"
    quality.write_text(
        json.dumps(
            {
                "quality_gate": {"eligible": True},
                "review_sample": {
                    "sha256": hashlib.sha256(sample.read_bytes()).hexdigest()
                },
                "sources": {
                    "responses": [
                        {"sha256": hashlib.sha256(responses.read_bytes()).hexdigest()}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    rows, report = _validate_bindings(sample, responses, quality)
    assert len(rows) == 1200
    assert report["quality_gate"]["eligible"] is True

    sample.write_text(sample.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="review sample disagrees"):
        _validate_bindings(sample, responses, quality)
