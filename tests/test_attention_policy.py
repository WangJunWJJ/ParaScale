# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午9:40
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Attention experiments must isolate and restore backend selection."""

from contextlib import nullcontext

import pytest

torch = pytest.importorskip("torch")


def backend_state():
    cuda = torch.backends.cuda
    return {
        "math": cuda.math_sdp_enabled(),
        "flash": cuda.flash_sdp_enabled(),
        "efficient": cuda.mem_efficient_sdp_enabled(),
        "cudnn": cuda.cudnn_sdp_enabled(),
        "math_reduced_precision": cuda.fp16_bf16_reduction_math_sdp_allowed(),
    }


@pytest.mark.parametrize("fail", [False, True])
def test_math_attention_policy_restores_state_after_worker(fail):
    from parascale.runtime.inference.attention import attention_policy

    before = backend_state()
    cuda = torch.backends.cuda
    cuda.allow_fp16_bf16_reduction_math_sdp(True)
    original = backend_state()
    try:
        with (
            pytest.raises(RuntimeError, match="worker failed")
            if fail
            else nullcontext()
        ):
            with attention_policy("math") as state:
                assert state == {"requested": "math", "enabled": backend_state()}
                assert state["enabled"] == {
                    "math": True,
                    "flash": False,
                    "efficient": False,
                    "cudnn": False,
                    "math_reduced_precision": False,
                }
                query = torch.randn(1, 2, 4, 8)
                output = torch.nn.functional.scaled_dot_product_attention(
                    query, query, query, is_causal=True
                )
                assert torch.isfinite(output).all()
                if fail:
                    raise RuntimeError("worker failed")
        assert backend_state() == original
    finally:
        cuda.allow_fp16_bf16_reduction_math_sdp(before["math_reduced_precision"])
    assert backend_state() == before


def test_auto_attention_policy_does_not_change_enabled_backends():
    from parascale.runtime.inference.attention import attention_policy

    before = backend_state()
    with attention_policy("auto") as state:
        assert state == {"requested": "auto", "enabled": before}
        assert backend_state() == before
    assert backend_state() == before


def test_unknown_attention_policy_is_rejected_without_mutation():
    from parascale.runtime.inference.attention import attention_policy

    before = backend_state()
    with pytest.raises(ValueError, match="attention"):
        with attention_policy("invalid"):
            pytest.fail("unsupported attention policy was accepted")
    assert backend_state() == before


def test_worker_executes_within_attention_policy(monkeypatch, tmp_path):
    from tests.benchmarks.scripts import benchmark_quantized_inference as benchmark

    before = backend_state()
    observed = []

    def worker(args, state):
        observed.append(state)
        assert args.attention_backend == "math"
        assert state["enabled"] == backend_state()
        assert state["enabled"]["math"]
        assert not state["enabled"]["flash"]

    monkeypatch.setattr(benchmark, "run_worker", worker)
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        [
            "benchmark",
            "--mode",
            "bf16",
            "--model",
            str(tmp_path),
            "--data",
            str(tmp_path / "eval.txt"),
            "--data-description",
            "test",
            "--output",
            str(tmp_path / "result.json"),
            "--attention-backend",
            "math",
        ],
    )
    benchmark.main()
    assert len(observed) == 1
    assert backend_state() == before
