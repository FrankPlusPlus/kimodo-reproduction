from __future__ import annotations

import numpy as np
import pytest
import torch

from kimodo.constraints import (
    EndEffectorConstraintSet,
    FullBodyConstraintSet,
    Root2DConstraintSet,
)
from kimodo.exports.motion_io import (
    _quaternion_slerp,
    load_motion_file,
    resample_motion_dict_to_kimodo_fps,
)
from kimodo.model.backbone import pad_x_and_mask_to_fixed_size
from kimodo.model.kimodo_model import _resolve_batched_constraints
from kimodo.skeleton.registry import build_skeleton


def _identity_motion(frames: int, joints: int = 30) -> dict[str, torch.Tensor]:
    rotations = torch.eye(3).expand(frames, joints, 3, 3).clone()
    roots = torch.zeros(frames, 3)
    roots[:, 1] = 1.0
    return {"local_rot_mats": rotations, "root_positions": roots}


def test_fixed_text_padding_crops_features_and_mask_together():
    features = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    mask = torch.tensor([[True, False, True, False, True, False]])

    cropped, cropped_mask = pad_x_and_mask_to_fixed_size(features, mask, size=4)

    assert torch.equal(cropped, features[:, :4])
    assert torch.equal(cropped_mask, mask[:, :4])


def test_3d_smooth_root_constraints_select_xz_not_xy():
    skeleton = build_skeleton(30)
    frame_indices = torch.tensor([0, 1])
    smooth_root_3d = torch.tensor([[1.0, 100.0, 2.0], [3.0, 200.0, 4.0]])
    expected_xz = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    root = Root2DConstraintSet(skeleton, frame_indices, smooth_root_3d)

    positions = skeleton.neutral_joints.unsqueeze(0).expand(2, -1, -1).clone()
    rotations = torch.eye(3).expand(2, skeleton.nbjoints, 3, 3).clone()
    full_body = FullBodyConstraintSet(
        skeleton,
        frame_indices,
        positions,
        rotations,
        smooth_root_2d=smooth_root_3d,
    )
    end_effector = EndEffectorConstraintSet(
        skeleton,
        frame_indices,
        positions,
        rotations,
        smooth_root_2d=smooth_root_3d,
        joint_names=["LeftHand"],
    )

    assert torch.equal(root.smooth_root_2d, expected_xz)
    assert torch.equal(full_body.smooth_root_2d, expected_xz)
    assert torch.equal(end_effector.smooth_root_2d, expected_xz)


def test_constraint_sets_reject_negative_frame_indices():
    skeleton = build_skeleton(30)
    with pytest.raises(ValueError, match="non-negative"):
        Root2DConstraintSet(
            skeleton,
            torch.tensor([-1]),
            torch.tensor([[0.0, 0.0]]),
        )


def test_per_sample_constraints_are_not_treated_as_constraint_objects():
    first = object()
    second = object()
    resolved = _resolve_batched_constraints([[first], [second]], num_samples=2)
    assert resolved == [[first], [second]]

    shared = _resolve_batched_constraints([first], num_samples=2)
    assert shared == [[first], [first]]

    with pytest.raises(ValueError, match="one list per generated sample"):
        _resolve_batched_constraints([[first]], num_samples=2)


def test_slerp_is_finite_for_identical_and_near_identical_quaternions():
    q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1e-7, 0.0, 0.0]])
    q1 = q1 / torch.linalg.norm(q1, dim=-1, keepdim=True)

    interpolated = _quaternion_slerp(q0, q1, torch.tensor([0.25, 0.75]))

    assert torch.isfinite(interpolated).all()
    assert torch.allclose(
        torch.linalg.norm(interpolated, dim=-1),
        torch.ones(2),
        atol=1e-6,
    )
    assert torch.allclose(interpolated[0], q0[0], atol=1e-7)


def test_fractional_resampling_keeps_constant_rotations_finite():
    skeleton = build_skeleton(30)
    resampled, changed = resample_motion_dict_to_kimodo_fps(
        _identity_motion(6),
        skeleton,
        source_fps=60.0,
        target_fps=24.0,
    )

    assert changed
    assert torch.isfinite(resampled["local_rot_mats"]).all()
    expected = torch.eye(3).expand_as(resampled["local_rot_mats"])
    assert torch.allclose(resampled["local_rot_mats"], expected, atol=1e-6)


def test_load_motion_resamples_small_but_real_fps_difference(tmp_path):
    source = _identity_motion(77)
    path = tmp_path / "motion.npz"
    np.savez(
        path,
        local_rot_mats=source["local_rot_mats"].numpy(),
        root_positions=source["root_positions"].numpy(),
    )

    loaded, num_joints = load_motion_file(
        str(path),
        source_fps=30.4,
        target_fps=30.0,
    )

    assert num_joints == 30
    assert len(loaded["local_rot_mats"]) == 76
    assert torch.isfinite(loaded["local_rot_mats"]).all()


def test_load_motion_does_not_hide_sub_millihertz_duration_drift(tmp_path):
    source = _identity_motion(1113)
    path = tmp_path / "long-motion.npz"
    np.savez(
        path,
        local_rot_mats=source["local_rot_mats"].numpy(),
        root_positions=source["root_positions"].numpy(),
    )

    loaded, _ = load_motion_file(
        str(path),
        source_fps=1.0009,
        target_fps=1.0,
    )

    assert len(loaded["local_rot_mats"]) == 1112
