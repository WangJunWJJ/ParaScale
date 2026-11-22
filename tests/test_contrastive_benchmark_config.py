# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午7:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Benchmark configurations must pass product validation before GPU work."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tests.benchmarks.scripts import (
    benchmark_contrastive_training as benchmark,  # noqa: E402
)
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


def test_profile_disabled_is_noop():
    with benchmark.profile_run(None, "unused", 0) as profiler:
        assert profiler is None


def test_profile_is_bounded_and_rank_specific(monkeypatch, tmp_path):
    captured = {}

    def profile(**kwargs):
        captured.update(kwargs)
        return "profile-context"

    monkeypatch.setattr(benchmark.torch.profiler, "profile", profile)
    assert benchmark.profile_run(tmp_path, "screen", 3) == "profile-context"
    schedule = captured["schedule"]
    action = benchmark.torch.profiler.ProfilerAction
    assert [schedule(i) for i in (0, 9, 10, 11, 12, 13, 14)] == [
        action.NONE,
        action.NONE,
        action.WARMUP,
        action.RECORD,
        action.RECORD,
        action.RECORD_AND_SAVE,
        action.NONE,
    ]

    class Trace:
        def export_chrome_trace(self, path):
            captured["path"] = path

    captured["on_trace_ready"](Trace())
    assert captured["path"] == str(tmp_path / "screen.rank3.trace.json")


@pytest.mark.parametrize("global_batch", [32, 128])
def test_window_throughput_uses_actual_global_batch(global_batch):
    ranks = [
        {"windows": [{"wall_seconds": seconds}] * (benchmark.WINDOWS + 1)}
        for seconds in (1.0, 2.0)
    ]
    windows = benchmark.summarize_windows(ranks, global_batch)
    assert len(windows) == benchmark.WINDOWS
    assert all(w["max_rank_wall_seconds"] == 2.0 for w in windows)
    assert all(
        w["global_pairs_per_second"] == global_batch * benchmark.STEPS / 2.0
        for w in windows
    )
