from __future__ import annotations

import pytest
import torch

from kimodo.evaluation.validation_cli import _parameter_delta


def test_parameter_delta_reports_changed_tensor_and_relative_norm():
    baseline = {
        "first": torch.tensor([1.0, 2.0]),
        "second": torch.tensor([0.0]),
    }
    current = {
        "first": torch.tensor([1.0, 3.0]),
        "second": torch.tensor([0.0]),
    }
    result = _parameter_delta(current, baseline)
    assert result["changed_tensors"] == 1
    assert result["tensor_count"] == 2
    assert result["maximum_absolute_delta"] == 1.0
    assert result["relative_l2"] == pytest.approx(torch.sqrt(torch.tensor(0.2)).item())
