# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午6:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Optimizer factory policy and actual updates for distributed parameters."""

import copy
from datetime import timedelta
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
try:
    from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
except ImportError:
    from torch.distributed._tensor import DTensor, Replicate, distribute_tensor

from parascale.optimizers import build_optimizer  # noqa: E402


@pytest.fixture
def mesh(tmp_path):
    if not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    from torch.distributed.device_mesh import init_device_mesh

    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / "rendezvous").as_uri(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def parameters_model(mesh, *, all_distributed=False, frozen=False):
    model = torch.nn.Module()
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
    model.weight = torch.nn.Parameter(
        distribute_tensor(weight, mesh, [Replicate()]), requires_grad=not frozen
    )
    bias = torch.arange(3, dtype=torch.float32) / 10
    model.bias = torch.nn.Parameter(
        distribute_tensor(bias, mesh, [Replicate()]) if all_distributed else bias
    )
    return model


def local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def test_plain_adamw_retains_native_execution_selection():
    model = torch.nn.Linear(4, 3)
    optimizer = build_optimizer(model, {})
    assert optimizer.defaults["foreach"] is None
    assert optimizer.defaults["fused"] is None


@pytest.mark.parametrize("all_distributed", [False, True])
def test_dtensor_adamw_matches_native_updates_and_state(mesh, all_distributed):
    model = parameters_model(mesh, all_distributed=all_distributed)
    reference = [
        torch.nn.Parameter(local(p).detach().clone()) for p in model.parameters()
    ]
    settings = dict(lr=0.003, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.07)
    optimizer = build_optimizer(model, {"optimizer": {"type": "adamw", **settings}})
    native = torch.optim.AdamW(reference, **settings, foreach=False, fused=False)
    expected_mode = None if all_distributed else False
    assert optimizer.defaults["foreach"] is expected_mode
    assert optimizer.defaults["fused"] is expected_mode
    assert [id(p) for p in optimizer.param_groups[0]["params"]] == [
        id(p) for p in model.parameters()
    ]
    for step in range(3):
        for parameter, expected in zip(model.parameters(), reference):
            parameter.grad = torch.ones_like(parameter) * (step + 1) / 10
            expected.grad = torch.ones_like(expected) * (step + 1) / 10
        optimizer.step()
        native.step()
        for parameter, expected in zip(model.parameters(), reference):
            torch.testing.assert_close(local(parameter), expected, rtol=0, atol=0)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    local(optimizer.state[parameter][key]),
                    native.state[expected][key],
                    rtol=0,
                    atol=0,
                )
        if step == 1:
            state = copy.deepcopy(optimizer.state_dict())
            optimizer = build_optimizer(model, {"optimizer": settings})
            optimizer.load_state_dict(state)


@pytest.mark.parametrize("kind", ["four_bit_adamw", "four_bit_sgd"])
def test_four_bit_factory_rejects_dtensor_before_construction(mesh, monkeypatch, kind):
    import parascale.optimizers.four_bit as four_bit

    model = parameters_model(mesh)
    initial = [local(p).detach().clone() for p in model.parameters()]
    constructor = Mock(side_effect=AssertionError("unsupported optimizer constructed"))
    monkeypatch.setattr(
        four_bit,
        "FourBitAdamW" if kind == "four_bit_adamw" else "FourBitSGD",
        constructor,
    )
    with pytest.raises(ValueError, match="DTensor.*adamw"):
        build_optimizer(model, {"optimizer": {"type": kind}})
    constructor.assert_not_called()
    for parameter, expected in zip(model.parameters(), initial):
        torch.testing.assert_close(local(parameter), expected, rtol=0, atol=0)
        assert parameter.grad is None


@pytest.mark.parametrize("kind", ["four_bit_adamw", "four_bit_sgd", "adamw"])
def test_frozen_dtensor_does_not_change_trainable_optimizer_policy(mesh, kind):
    model = parameters_model(mesh, frozen=True)
    optimizer = build_optimizer(model, {"optimizer": {"type": kind}})
    assert optimizer.param_groups[0]["params"] == [model.bias]
    if kind == "adamw":
        assert optimizer.defaults["foreach"] is None
    model.bias.grad = torch.ones_like(model.bias)
    before = model.bias.detach().clone()
    optimizer.step()
    assert not torch.equal(model.bias, before)
