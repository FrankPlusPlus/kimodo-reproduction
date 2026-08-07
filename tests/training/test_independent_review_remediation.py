from __future__ import annotations

import argparse
import json
from pathlib import Path

from kimodo.training.independent_review_remediation_cli import (
    _policy,
    prepare_supplement,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY = (
    PROJECT_ROOT
    / "artifacts/benchmark-metadata/v2/expert-semantic-adjudication.v2.2.json"
)


def test_v2_adjudication_policy_is_an_evidenced_9_15_partition():
    policy, decisions = _policy(POLICY)
    true_major = {
        request_id
        for request_id, row in decisions.items()
        if row["adjudication"] == "true_major"
    }
    false_positive = set(decisions) - true_major
    assert policy["policy_id"] == "kimodo-v2.2-motion-caption-adjudication-2026-08-07"
    assert len(decisions) == 24
    assert len(true_major) == 9
    assert len(false_positive) == 15
    assert true_major.isdisjoint(false_positive)
    assert all(row["issue_codes"] and row["evidence"] for row in decisions.values())


def test_prepare_supplement_carries_only_exact_sample_overlap(tmp_path):
    old_sample = tmp_path / "old-sample.jsonl"
    new_sample = tmp_path / "new-sample.jsonl"
    old_verdicts = tmp_path / "old-verdicts.jsonl"
    partial = tmp_path / "supplement.jsonl.partial"
    old_ids = [f"id-{index}" for index in range(1_200)]
    new_ids = [*old_ids[:-1], "new-id"]
    old_sample.write_text(
        "".join(json.dumps({"request_id": value}) + "\n" for value in old_ids),
        encoding="utf-8",
    )
    new_sample.write_text(
        "".join(json.dumps({"request_id": value}) + "\n" for value in new_ids),
        encoding="utf-8",
    )
    old_verdicts.write_text(
        "".join(
            json.dumps({"request_id": value, "verdict": "pass"}) + "\n"
            for value in old_ids
        ),
        encoding="utf-8",
    )
    result = prepare_supplement(
        argparse.Namespace(
            review_sample=str(new_sample),
            source_review_sample=str(old_sample),
            source_expert_verdicts=str(old_verdicts),
            output_partial=str(partial),
        )
    )
    assert result["carried_forward_rows"] == 1_199
    assert result["supplemental_request_ids"] == ["new-id"]
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 1_199
