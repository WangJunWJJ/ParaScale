# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Portable argument and measurement contracts for real Llama TP benchmarks."""

import copy
import weakref

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
    assert args.precision == "fp32"


def test_bf16_benchmark_is_an_explicit_matched_policy():
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
            "--precision",
            "bf16",
        ]
    )
    assert args.precision == "bf16"


def test_native_reference_plan_contains_shared_input_and_projection_styles():
    from tests.benchmarks.scripts.benchmark_llama_parallel import native_tensor_plan

    plan = native_tensor_plan(2)
    assert len(plan) == 18
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        assert type(plan[f"{prefix}.self_attn"]).__name__ == "PrepareModuleInput"
        assert type(plan[f"{prefix}.mlp"]).__name__ == "PrepareModuleInput"
        assert type(plan[f"{prefix}.self_attn.q_proj"]).__name__ == "ColwiseParallel"
        assert type(plan[f"{prefix}.self_attn.o_proj"]).__name__ == "RowwiseParallel"
    assert len(native_tensor_plan(2, shared=False)) == 14


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


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("microbatch_size", [2, 4])
def test_native_backward_releases_outputs_and_preserves_gradients(
    monkeypatch, precision, microbatch_size
):
    from tests.benchmarks.scripts.benchmark_llama_parallel import (
        native_forward_backward,
        summed_loss,
    )

    torch.manual_seed(37)
    model = torch.nn.Sequential(torch.nn.Embedding(7, 8), torch.nn.Linear(8, 7))
    reference = copy.deepcopy(model)
    batch = torch.tensor([[1, 2, 3], [2, 4, 5], [6, 1, 2], [3, 5, 1]])
    expected_loss = torch.zeros(())
    for part in batch.split(microbatch_size):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
            prediction = reference(part)
            loss = summed_loss(prediction, part) / 8
        loss.backward()
        expected_loss += loss.detach()

    outputs = []
    original_backward = torch.Tensor.backward

    def forward(part):
        prediction = model(part)
        outputs.append(weakref.ref(prediction))
        return prediction

    def check_backward(tensor, *args, **kwargs):
        assert outputs[-1]() is None
        return original_backward(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", check_backward)
    actual_loss = native_forward_backward(
        forward, batch, microbatch_size=microbatch_size, precision=precision
    )
    assert len(outputs) == 4 // microbatch_size
    assert all(output() is None for output in outputs)
    assert not actual_loss.requires_grad
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
