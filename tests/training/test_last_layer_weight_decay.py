from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from kimodo.training.optim import AdamAtan2, build_optimizer


class _FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


class _FakeBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seqTransEncoder = _FakeEncoder()


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body_model = _FakeBody()
        self.head = nn.Linear(2, 2)


def _config(**overrides) -> SimpleNamespace:
    values = dict(
        name="adam_atan2",
        learning_rate=1.0e-5,
        betas=(0.9, 0.999),
        weight_decay=0.3,
        last_layer_weight_decay=None,
        atan2_lambda=8.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_optimizer_keeps_single_group_without_last_layer_decay():
    optimizer = build_optimizer(_FakeModel(), _config())
    assert isinstance(optimizer, AdamAtan2)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["name"] == "rest"
    assert optimizer.param_groups[0]["weight_decay"] == 0.3


def test_build_optimizer_splits_last_layer_weight_decay():
    model = _FakeModel()
    optimizer = build_optimizer(model, _config(last_layer_weight_decay=1.0))
    by_name = {group["name"]: group for group in optimizer.param_groups}
    assert set(by_name) == {"rest", "last_layer"}
    assert by_name["rest"]["weight_decay"] == 0.3
    assert by_name["last_layer"]["weight_decay"] == 1.0
    last_ids = {id(parameter) for parameter in model.body_model.seqTransEncoder.layers[-1].parameters()}
    grouped_last = {id(parameter) for parameter in by_name["last_layer"]["params"]}
    grouped_rest = {id(parameter) for parameter in by_name["rest"]["params"]}
    assert grouped_last == last_ids
    assert grouped_rest.isdisjoint(last_ids)
    assert grouped_rest
