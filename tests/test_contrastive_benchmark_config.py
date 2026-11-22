# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午7:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Benchmark configurations must pass product validation before GPU work."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tests.benchmarks.scripts.benchmark_contrastive_training import (
    runtime_config,  # noqa: E402
)


@pytest.mark.parametrize("world", [1, 2, 4])
def test_runtime_configuration_is_valid_and_bounded(world):
    config = runtime_config(world)
    assert config.data_parallel_size == world
    assert config.training_backend == ("native" if world == 1 else "native_ddp")
    assert config.metrics_history_limit == 1
    assert config.gradient_accumulation_steps == 1
    assert config.precision == "fp32"
    assert config.ddp_comm_hook == "none"
    assert config.grad_clip_norm is None
