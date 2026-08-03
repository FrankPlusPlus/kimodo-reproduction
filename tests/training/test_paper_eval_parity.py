from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmark.evaluate_folder import compute_paper_retrieval_set
from benchmark.parse_folder import _load_paper_protocol_outputs
from kimodo.constraints import EndEffectorConstraintSet, FullBodyConstraintSet, Root2DConstraintSet
from kimodo.metrics import (
    PAPER_END_EFFECTOR_POSITION,
    PAPER_END_EFFECTOR_ROTATION,
    PAPER_FULLBODY_POSITION,
    PAPER_PELVIS_TO_SMOOTH_ROOT_2D,
    PAPER_SMOOTH_ROOT_2D,
    compute_paper_constraint_errors,
    rotation_geodesic_degrees,
)


class _ThreeJointSkeleton:
    root_idx = 0
    nbjoints = 3
    hip_joint_idx = (1, 2)
    bone_index = {"Hips": 0, "LeftHand": 1, "LeftFoot": 2}

    def expand_joint_names(self, names: list[str]) -> tuple[list[str], list[str]]:
        return names, names


def _z_rotation(degrees: float) -> torch.Tensor:
    radians = torch.deg2rad(torch.tensor(degrees))
    c, s = torch.cos(radians), torch.sin(radians)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_rotation_error_is_so3_geodesic_in_degrees() -> None:
    identity = torch.eye(3)
    angles = rotation_geodesic_degrees(
        torch.stack([identity, _z_rotation(90.0), _z_rotation(180.0)]),
        identity.expand(3, -1, -1),
    )
    torch.testing.assert_close(angles, torch.tensor([0.0, 90.0, 180.0]), atol=1e-4, rtol=0.0)


def test_paper_constraint_metrics_keep_smooth_root_and_pelvis_separate() -> None:
    skeleton = _ThreeJointSkeleton()
    posed = torch.zeros(2, 3, 3)
    posed[0, 0, [0, 2]] = torch.tensor([0.3, 0.4])
    posed[1, 1, 0] = 2.0
    generated_smooth_root = torch.tensor([[3.0, 0.0, 4.0], [0.0, 0.0, 0.0]])

    rotations = torch.eye(3).expand(2, 3, 3, 3).clone()
    rotations[1, 1] = _z_rotation(90.0)
    targets_pos = torch.zeros(1, 3, 3)
    targets_rot = torch.eye(3).expand(1, 3, 3, 3).clone()

    constraints = [
        Root2DConstraintSet(skeleton, torch.tensor([0, 1]), torch.zeros(2, 2)),
        FullBodyConstraintSet(
            skeleton,
            torch.tensor([0]),
            targets_pos,
            targets_rot,
            torch.zeros(1, 2),
        ),
        EndEffectorConstraintSet(
            skeleton,
            torch.tensor([1]),
            targets_pos,
            targets_rot,
            torch.zeros(1, 2),
            joint_names=["LeftHand"],
        ),
    ]
    errors = compute_paper_constraint_errors(
        posed_joints=posed,
        global_rot_mats=rotations,
        smooth_root_pos=generated_smooth_root,
        constraints_lst=constraints,
        root_idx=skeleton.root_idx,
    )

    torch.testing.assert_close(errors[PAPER_SMOOTH_ROOT_2D], torch.tensor([5.0, 0.0]))
    torch.testing.assert_close(errors[PAPER_PELVIS_TO_SMOOTH_ROOT_2D], torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(errors[PAPER_END_EFFECTOR_POSITION], torch.tensor([2.0]))
    torch.testing.assert_close(errors[PAPER_END_EFFECTOR_ROTATION], torch.tensor([90.0]), atol=1e-4, rtol=0.0)
    assert errors[PAPER_FULLBODY_POSITION].numel() == 3


def test_paper_root_metric_refuses_pelvis_as_smooth_root_substitute() -> None:
    skeleton = _ThreeJointSkeleton()
    constraint = Root2DConstraintSet(skeleton, torch.tensor([0]), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="smooth_root_pos"):
        compute_paper_constraint_errors(
            posed_joints=torch.zeros(1, 3, 3),
            constraints_lst=[constraint],
            root_idx=0,
        )


def _write_embeddings(sample: Path, motion: np.ndarray, text: np.ndarray, gt: np.ndarray) -> None:
    sample.mkdir(parents=True)
    np.save(sample / "motion_embedding.npy", motion)
    np.save(sample / "text_embedding.npy", text)
    np.save(sample / "gt_motion_embedding.npy", gt)


def test_paper_retrieval_uses_one_matrix_for_the_complete_set(tmp_path: Path) -> None:
    # Any two-sample subgroup has R@3=100 by construction.  The adversarial
    # permutation only becomes visible when all four samples compete.
    text = np.eye(8, dtype=np.float64)
    motion = text[::-1].copy()
    examples = []
    for index in range(8):
        sample = tmp_path / f"case_{index // 2}" / f"sample_{index}"
        _write_embeddings(sample, motion[index], text[index], text[index])
        examples.append((sample, Path(f"case_{index // 2}/sample_{index}")))

    output = compute_paper_retrieval_set(examples)
    assert output["num_motions"] == 8
    assert output["metrics"]["TMR/t2m_R/R03"] < 100.0
    assert output["metrics"]["TMR/t2m_gt_R/R03"] == 100.0
    assert "complete prompt test set" in output["aggregation"]


def test_parser_keeps_paper_protocol_separate_from_legacy_rows(tmp_path: Path) -> None:
    retrieval_path = tmp_path / "content" / "text2motion" / "overview" / "paper_retrieval.json"
    retrieval_path.parent.mkdir(parents=True)
    retrieval_path.write_text(
        json.dumps({"protocol": "paper", "num_motions": 8, "metrics": {"TMR/t2m_R/R03": 75.0}}),
        encoding="utf-8",
    )
    constraint_path = tmp_path / "content" / "paper_constraint_metrics.json"
    constraint_path.write_text(json.dumps({"protocol": "paper", "generated": {}}), encoding="utf-8")

    loaded = _load_paper_protocol_outputs(tmp_path)
    assert loaded["retrieval"][0]["category"] == "overview"
    assert loaded["retrieval"][0]["metrics"]["TMR/t2m_R/R03"] == 75.0
    assert loaded["constraints"][0]["split"] == "content"
