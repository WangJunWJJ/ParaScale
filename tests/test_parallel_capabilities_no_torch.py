# -*- coding: utf-8 -*-
# @Time : 2026/11/21 下午6:21
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Legacy runtime capability boundaries, without importing torch."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from parascale.parallel import build_parallel_plan
from parascale.runtime.backends.registry import (
    create_runtime_training_backend,
    default_training_backend_registry,
)
from parascale.runtime.training.engine import TrainEngine
from parascale.strategy import StrategyPlan, build_strategy_plan

BACKENDS = ("native", "native_ddp", "fsdp", "deepspeed", "auto")
DIMENSIONS = ("tensor_parallel_size", "pipeline_parallel_size", "model_parallel_size")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dimension", DIMENSIONS)
@pytest.mark.parametrize("as_dict", (False, True))
def test_strategy_rejects_explicit_unsupported_dimensions(backend, dimension, as_dict):
    values = {"training_backend": backend, dimension: 3}
    config = values if as_dict else SimpleNamespace(**values)
    with pytest.raises(ValueError, match=dimension):
        build_strategy_plan({}, {"num_gpus": 2}, config)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("world_size", (1, 8, 32))
def test_strategy_never_invents_model_parallelism(backend, world_size):
    plan = build_strategy_plan(
        {"total_params": 20_000_000_000, "num_layers": 48},
        {"num_gpus": world_size, "gpus_per_node": 8},
        SimpleNamespace(training_backend=backend),
    )
    assert (plan.dp_size, plan.tp_size, plan.pp_size) == (world_size, 1, 1)


@pytest.mark.parametrize("dimension", DIMENSIONS)
@pytest.mark.parametrize("supplied_plan", (False, True))
def test_initialize_rejects_before_process_initialization(dimension, supplied_plan):
    collective = Mock()
    engine = TrainEngine(
        config=SimpleNamespace(**{dimension: 2}),
        strategy_plan=StrategyPlan() if supplied_plan else None,
        collective=collective,
        training_backend=Mock(),
    )
    with pytest.raises(ValueError, match=dimension):
        engine.initialize()
    collective.init_process_group.assert_not_called()
    assert not engine.state.initialized


@pytest.mark.parametrize("dimension", ("tp_size", "pp_size"))
def test_supplied_plan_rejected_before_process_initialization(dimension):
    collective = Mock()
    engine = TrainEngine(
        config=SimpleNamespace(),
        strategy_plan=StrategyPlan(**{dimension: 2}),
        collective=collective,
    )
    with pytest.raises(ValueError, match=dimension):
        engine.initialize()
    collective.init_process_group.assert_not_called()


@pytest.mark.parametrize("dimension", DIMENSIONS + ("tp_size", "pp_size"))
def test_fit_revalidates_before_model_or_optimizer_mutation(dimension):
    config = SimpleNamespace()
    plan = StrategyPlan()
    target = plan if dimension in {"tp_size", "pp_size"} else config
    setattr(target, dimension, 2)
    backend = Mock(model=None)
    builder = Mock()
    engine = TrainEngine(config=config, strategy_plan=plan, training_backend=backend)
    engine.state.initialized = True
    engine.fit_loop = Mock()
    with pytest.raises(ValueError, match=dimension):
        engine.fit(
            [1],
            model=object(),
            optimizer=object(),
            loss_fn=Mock(),
            optimizer_builder=builder,
        )
    backend.setup.assert_not_called()
    builder.assert_not_called()
    engine.fit_loop.run.assert_not_called()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_runtime_factory_rejects_before_config_mutation(backend, dimension):
    config = SimpleNamespace(training_backend=backend, **{dimension: 2})
    with pytest.raises(ValueError, match=dimension) as error:
        create_runtime_training_backend(model=object(), config=config)
    assert config.training_backend == backend
    assert "legacy training runtime" in str(error.value)
    assert "TrainEngine" not in str(error.value)
    assert "model-specific experimental" in str(error.value)
    assert "parascale.parallel.distributed" in str(error.value)
    assert "parascale.parallel.execution" in str(error.value)


@pytest.mark.parametrize("backend", BACKENDS[:-1])
def test_registry_rejects_before_calling_backend_factory(backend):
    registry = default_training_backend_registry()
    factory = Mock()
    registry.register(backend, factory)
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        registry.create(backend, config=SimpleNamespace(tensor_parallel_size=2))
    factory.assert_not_called()


def test_declarative_introspection_still_allows_tensor_and_pipeline_dimensions():
    config = SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=4)
    strategy = StrategyPlan(tp_size=2, pp_size=4)
    payload = build_parallel_plan(config, strategy).to_dict()
    assert payload["dimensions"]["tensor"]["size"] == 2
    assert payload["dimensions"]["pipeline"]["size"] == 4


@pytest.mark.parametrize("backend", BACKENDS[:-1])
def test_supported_defaults_keep_backend_and_precision(backend):
    config = SimpleNamespace(training_backend=backend, precision="fp16")
    plan = build_strategy_plan({}, {"num_gpus": 2}, config)
    assert plan.backend == backend
    assert plan.precision == "fp16"
    assert create_runtime_training_backend(config=config).name == backend
