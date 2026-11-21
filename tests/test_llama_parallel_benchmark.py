# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Portable argument and measurement contracts for real Llama TP benchmarks."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.mark.parametrize(
    "arguments",
    [
        ["--global-batch", "0"],
        ["--microbatch", "0"],
        ["--global-batch", "7"],
        ["--sequence-length", "1"],
        ["--sequence-length", "255"],
        ["--validation-tokens", "0"],
        ["--steps", "8", "--warmup", "8"],
        ["--warmup", "-1"],
        ["--adam-eps", "nan"],
        ["--adam-eps", "0"],
        ["--tp", "3"],
        ["--cpu-affinity", "-1"],
    ],
)
def test_invalid_configuration_fails_before_device_work(arguments):
    from tests.benchmarks.scripts.benchmark_llama_parallel import parse_args

    with pytest.raises(SystemExit) as error:
        parse_args(
            [
                "--model",
                "model",
                "--data",
                "data",
                "--output",
                "out.json",
                "--driver",
                "native",
                *arguments,
            ]
        )
    assert error.value.code == 2


def test_defaults_do_not_pin_machine_specific_devices_or_cpus():
    from tests.benchmarks.scripts.benchmark_llama_parallel import parse_args

    args = parse_args(
        [
            "--model",
            "model",
            "--data",
            "data",
            "--output",
            "out.json",
            "--driver",
            "native",
        ]
    )
    assert args.cpu_affinity is None
    assert args.native_plan == "shared"
    assert args.adam_eps == 1e-8
    assert args.tp == 1


def test_summed_loss_counts_shifted_tokens_without_tp_multiplier():
    from tests.benchmarks.scripts.benchmark_llama_parallel import summed_loss

    logits = torch.zeros(2, 3, 5, requires_grad=True)
    labels = torch.tensor([[1, 2, 3], [2, 3, 4]])
    result = summed_loss(logits, labels)
    torch.testing.assert_close(result, torch.log(torch.tensor(5.0)) * 4)
    result.backward()
    assert torch.count_nonzero(logits.grad[:, -1]) == 0


def test_throughput_uses_total_measured_time_not_median_only():
    from tests.benchmarks.scripts.benchmark_llama_parallel import summarize_timing

    result = summarize_timing([1.0, 2.0, 9.0], batch=2, length=5)
    assert result["tokens_per_second"] == 2.5
    assert result["scored_tokens_per_second"] == 2.0
    assert result["median_step_tokens_per_second"] == 5.0
