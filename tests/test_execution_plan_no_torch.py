# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Planning inputs must describe allocation, not the machine's GPU inventory."""

from types import SimpleNamespace

import pytest

from parascale.config import ParaScaleConfig
from parascale.runtime import build_runtime_context, lifecycle
from parascale.runtime.training import TrainEngine
from parascale.strategy import StrategyPlan, build_strategy_plan


@pytest.mark.parametrize("world", [2, 4])
def test_declared_world_size_is_used_without_environment(monkeypatch, world):
    monkeypatch.setenv("WORLD_SIZE", "16")
    plan = build_strategy_plan(
        {}, {"world_size": world}, ParaScaleConfig(training_backend="native_ddp")
    )
    assert plan.dp_size == world


def test_allocation_overrides_inventory_and_reaches_context_plan():
    context = build_runtime_context(
        {"parascale": {"training_backend": "native_ddp"},
         "hardware_profile": {"num_gpus": 5}},
        world_size=2,
    )
    assert context.world_size == context.strategy_plan.dp_size == 2


def test_explicit_execution_world_size_rejects_declared_conflict():
    with pytest.raises(ValueError, match="world_size"):
        build_strategy_plan({}, {"world_size": 4}, ParaScaleConfig(), world_size=2)


@pytest.mark.parametrize("world", [0, -1, True, 1.5])
def test_explicit_execution_size_must_be_a_positive_integer(world):
    with pytest.raises(ValueError, match="world_size"):
        build_strategy_plan({}, {}, ParaScaleConfig(), world_size=world)


def test_native_zero_stage_does_not_require_a_second_enable_flag():
    config = ParaScaleConfig(training_backend="native_ddp", zero_stage=1)
    assert config.zero_optimization is False
    assert build_strategy_plan({}, {"num_gpus": 2}, config).zero_stage == 1


def test_auto_small_multirank_model_uses_a_synchronizing_backend():
    plan = build_strategy_plan(
        {}, {}, ParaScaleConfig(training_backend="auto"), world_size=2
    )
    assert plan.backend == "native_ddp" and plan.dp_size == 2


def test_engine_plan_uses_execution_size_and_rejects_stale_cache(monkeypatch):
    import parascale.runtime.training.engine as module

    monkeypatch.setattr(module, "execution_world_size", lambda: 2, raising=False)
    engine = TrainEngine(ParaScaleConfig(training_backend="native_ddp"))
    assert engine.plan().dp_size == 2
    engine.strategy_plan = StrategyPlan(backend="native_ddp", dp_size=1)
    with pytest.raises(ValueError, match="world_size"):
        engine.plan()


def test_execution_size_prefers_group_and_rejects_launcher_conflict(monkeypatch):
    group = SimpleNamespace(get_world_size=lambda: 2)
    monkeypatch.setattr(lifecycle, "initialized_distributed", lambda: group, raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert lifecycle.execution_world_size() == 2
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        lifecycle.execution_world_size()


def test_single_process_engine_does_not_plan_for_unallocated_inventory(monkeypatch):
    import parascale.runtime.training.engine as module

    monkeypatch.setattr(module, "execution_world_size", lambda: None)
    engine = TrainEngine(ParaScaleConfig(), hardware_profile={"num_gpus": 4})
    assert engine.plan().dp_size == 1
    engine.strategy_plan = StrategyPlan(dp_size=4)
    with pytest.raises(ValueError, match="world_size"):
        engine.plan()


def test_execution_size_before_group_initialization(monkeypatch):
    monkeypatch.setattr(lifecycle, "initialized_distributed", lambda: None, raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert lifecycle.execution_world_size() is None
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert lifecycle.execution_world_size() == 4


def test_product_runtime_rejects_unsynchronized_native_multirank(monkeypatch):
    import parascale.runtime.train_runner as runner

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    def forbidden(_config):
        raise AssertionError("must reject before building the workload")
    monkeypatch.setattr(runner, "build_training_components", forbidden)
    with pytest.raises(ValueError, match="native_ddp"):
        runner.run_train_from_config({"parascale": {"training_backend": "native"}})


def test_identity_does_not_require_optional_zero_optimizer_import(monkeypatch):
    import builtins

    from parascale.runtime.training import identity

    original = builtins.__import__
    def without_zero(name, *args, **kwargs):
        if name == "torch.distributed.optim":
            raise ImportError("optional optimizer module unavailable")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_zero)
    monkeypatch.setattr(identity, "initialized_distributed", lambda: None)
    engine = SimpleNamespace(
        training_backend=SimpleNamespace(optimizer=object()), scheduler=None
    )
    record, = identity.collect_execution_identity(engine)
    assert record["optimizer_class"] == "builtins.object"
    assert record["zero_redundancy_optimizer"] is None
