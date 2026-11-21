# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午1:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Strict semantic declarations without GPU or filesystem side effects."""

import json
from unittest.mock import Mock

import pytest

from parascale.parallel.checkpoint_context import canonical_checkpoint_context


def context():
    return dict(
        model={"width": 32},
        loss="causal-v1",
        data="sha256:example",
        tokenizer="test-tokenizer-v1",
        precision="fp32",
    )


@pytest.mark.parametrize(
    "invalid", [float("nan"), float("inf"), (1, 2), {1, 2}, {1: "key"}]
)
def test_context_rejects_non_json_or_lossy_values(invalid):
    value = context()
    value["model"]["invalid"] = invalid
    with pytest.raises(ValueError, match="context"):
        canonical_checkpoint_context(value)


def test_context_rejects_cycles():
    value = context()
    value["model"]["cycle"] = value
    with pytest.raises(ValueError, match="context"):
        canonical_checkpoint_context(value)


def test_context_canonical_order_and_scalar_types():
    value = context()
    assert canonical_checkpoint_context(
        dict(reversed(list(value.items())))
    ) == canonical_checkpoint_context(value)
    value["model"]["width"] = True
    boolean = canonical_checkpoint_context(value)
    value["model"]["width"] = 1
    assert canonical_checkpoint_context(value) != boolean
    assert json.loads(canonical_checkpoint_context(value)) == value


def test_constructor_detaches_context_before_execution(monkeypatch):
    torch = pytest.importorskip("torch")
    from parascale.parallel import execution
    from parascale.parallel.distributed import DistributedParallelConfig

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(execution, "_validate_constructor", lambda *args: None)
    mesh = Mock()
    monkeypatch.setattr(torch.distributed.device_mesh, "init_device_mesh", mesh)
    value = context()
    trainer = execution.DistributedTrainer(
        torch.nn.Linear(2, 2),
        DistributedParallelConfig(),
        torch.zeros(1, 2),
        torch.nn.MSELoss(),
        checkpoint_context=value,
    )
    value["model"]["width"] = 64
    assert trainer.checkpoint_context["model"]["width"] == 32
    trainer.checkpoint_context["model"]["width"] = 99
    assert trainer.checkpoint_context["model"]["width"] == 32
    mesh.assert_called_once()


def test_constructor_invalid_context_precedes_mesh_creation(monkeypatch):
    torch = pytest.importorskip("torch")
    from parascale.parallel import execution
    from parascale.parallel.distributed import DistributedParallelConfig

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(execution, "_validate_constructor", lambda *args: None)
    mesh = Mock()
    monkeypatch.setattr(torch.distributed.device_mesh, "init_device_mesh", mesh)
    with pytest.raises(ValueError, match="context"):
        execution.DistributedTrainer(
            torch.nn.Linear(2, 2),
            DistributedParallelConfig(),
            torch.zeros(1, 2),
            torch.nn.MSELoss(),
            checkpoint_context={},
        )
    mesh.assert_not_called()
