# -*- coding: utf-8 -*-
# @Time : 2026/11/21 下午6:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Pure validation for the explicit distributed execution API."""

import importlib

import pytest


def config_type():
    package = importlib.import_module("parascale.parallel")
    assert hasattr(package, "DistributedParallelConfig")
    return package.DistributedParallelConfig


@pytest.mark.parametrize(
    "dp,tp,pp",
    [(1, 1, 1), (4, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1), (2, 1, 2), (1, 2, 2)],
)
def test_supported_world_sizes(dp, tp, pp):
    config = config_type()(dp_size=dp, tp_size=tp, pp_size=pp)
    config.validate_world_size(dp * tp * pp)
    assert config.world_size == dp * tp * pp


def test_tp_is_fastest_varying_rank_axis():
    config = config_type()(dp_size=2, tp_size=2)
    assert [config.coordinates(rank) for rank in range(4)] == [
        (0, 0, 0),
        (0, 0, 1),
        (1, 0, 0),
        (1, 0, 1),
    ]


@pytest.mark.parametrize(
    "field,value",
    [("tp_size", 0), ("pp_size", -1), ("dp_size", True), ("microbatches", 1.5)],
)
def test_invalid_dimensions_rejected(field, value):
    with pytest.raises(ValueError, match="positive integer"):
        config_type()(**{field: value})


def test_full_three_dimensional_execution_is_not_certified():
    with pytest.raises(ValueError, match="four"):
        config_type()(dp_size=2, tp_size=2, pp_size=2)


@pytest.mark.parametrize("dimensions", [{"tp_size": 4}, {"pp_size": 4}, {"dp_size": 3}])
def test_unvalidated_topologies_are_not_advertised(dimensions):
    with pytest.raises(ValueError, match="validated"):
        config_type()(**dimensions)


def test_world_size_mismatch_fails_before_collectives():
    with pytest.raises(ValueError, match="world size"):
        config_type()(tp_size=2).validate_world_size(4)


def test_rank_bounds_checked():
    with pytest.raises(ValueError, match="rank"):
        config_type()(tp_size=2).coordinates(2)


def test_global_batch_excludes_model_parallel_ranks():
    config = config_type()(dp_size=2, tp_size=2, microbatches=4)
    assert config.global_batch_size(3) == 24


def test_api_is_lazy_and_does_not_require_torch():
    import subprocess
    import sys

    code = "import sys; sys.modules['torch'] = None; from parascale.parallel import DistributedParallelConfig; assert DistributedParallelConfig().world_size == 1"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_pipeline_schedule_is_explicit_and_frozen():
    from dataclasses import FrozenInstanceError, asdict

    assert config_type()().pipeline_schedule == "gpipe"
    config = config_type()(pp_size=2, pipeline_schedule="1f1b")
    assert asdict(config)["pipeline_schedule"] == "1f1b"
    with pytest.raises(FrozenInstanceError):
        config.pipeline_schedule = "gpipe"


@pytest.mark.parametrize("schedule", ["auto", "1F1B", "", None, True, 1])
def test_invalid_pipeline_schedule_rejected(schedule):
    with pytest.raises(ValueError, match="pipeline_schedule"):
        config_type()(pp_size=2, pipeline_schedule=schedule)


@pytest.mark.parametrize("options", [{}, {"pp_size": 2, "microbatches": 1}])
def test_1f1b_requires_pipeline_and_sufficient_microbatches(options):
    with pytest.raises(ValueError, match="1f1b"):
        config_type()(pipeline_schedule="1f1b", **options)
