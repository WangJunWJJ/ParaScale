# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Portable argument and measurement contracts for real Llama TP benchmarks."""

import copy
import weakref
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _parse_native(output, *arguments):
    from tests.benchmarks.scripts.benchmark_llama_parallel import parse_args

    return parse_args(
        [
            "--model",
            "model",
            "--data",
            "data",
            "--output",
            str(output),
            "--driver",
            "native",
            *arguments,
        ]
    )


@pytest.mark.parametrize("dp,adam", [(1, "single"), (2, "auto"), (4, "fused")])
def test_native_dp_and_adam_choices(tmp_path, dp, adam):
    args = _parse_native(
        tmp_path / "new.json",
        "--dp",
        str(dp),
        "--adam-implementation",
        adam,
        "--global-batch",
        "24",
        "--microbatch",
        "3",
    )
    assert args.dp == dp
    assert args.adam_implementation == adam


@pytest.mark.parametrize(
    "arguments",
    [
        ("--dp", "3"),
        ("--dp", "2", "--tp", "2"),
        ("--dp", "4", "--global-batch", "12", "--microbatch", "2"),
        ("--adam-implementation", "unknown"),
        ("--adam-implementation", "fused", "--tp", "2"),
    ],
)
def test_invalid_parallel_or_adam_settings_rejected(tmp_path, arguments):
    with pytest.raises(SystemExit) as error:
        _parse_native(tmp_path / "new.json", *arguments)
    assert error.value.code == 2


def test_existing_output_is_rejected_without_modification(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        _parse_native(output)
    assert error.value.code == 2
    assert output.read_text(encoding="utf-8") == "existing"


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
    assert args.dp == 1
    assert args.adam_implementation == "single"
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


def test_native_ddp_no_sync_covers_forward_and_backward(monkeypatch):
    from tests.benchmarks.scripts.benchmark_llama_parallel import (
        native_forward_backward,
    )

    events = []
    active = False

    class SpyDDP:
        class process_group:
            @staticmethod
            def size():
                return 1

        @contextmanager
        def no_sync(self):
            nonlocal active
            assert not active
            events.append("enter")
            active = True
            try:
                yield
            finally:
                active = False
                events.append("exit")

    def forward(part):
        events.append(("forward", active))
        logits = torch.zeros((*part.shape, 7), requires_grad=True)
        logits.register_hook(lambda grad: events.append(("backward", active)))
        return logits

    ddp = SpyDDP()

    def check_group(group):
        assert group is ddp.process_group
        return 1

    monkeypatch.setattr(torch.distributed, "get_world_size", check_group)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: check_group(group),
    )
    native_forward_backward(
        forward,
        torch.ones(6, 3, dtype=torch.long),
        microbatch_size=2,
        precision="fp32",
        ddp_model=ddp,
    )
    assert events == [
        "enter",
        ("forward", True),
        ("backward", True),
        "exit",
        "enter",
        ("forward", True),
        ("backward", True),
        "exit",
        ("forward", False),
        ("backward", False),
    ]


def _native_dp_worker(rank, rendezvous):
    from tests.benchmarks.scripts.benchmark_llama_parallel import (
        native_forward_backward,
        summed_loss,
    )

    dist = torch.distributed
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.set_num_threads(1)
        torch.manual_seed(37)
        model = torch.nn.Sequential(torch.nn.Embedding(7, 8), torch.nn.Linear(8, 7))
        reference = copy.deepcopy(model)
        group = dist.new_group(backend="gloo")
        ddp = torch.nn.parallel.DistributedDataParallel(model, process_group=group)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        batch = torch.tensor(
            [
                [1, 2, 3],
                [2, 4, 5],
                [6, 1, 2],
                [3, 5, 1],
                [1, 3, 6],
                [0, 4, 1],
                [5, 1, 0],
                [3, 0, 2],
            ]
        )
        local = batch.chunk(2)[rank].contiguous()
        expected = summed_loss(reference(batch), batch) / (batch.numel() - len(batch))
        expected.backward()

        actual = native_forward_backward(
            ddp,
            local,
            microbatch_size=2,
            precision="fp32",
            ddp_model=ddp,
        )
        assert not actual.requires_grad
        torch.testing.assert_close(actual, expected.detach(), rtol=1e-6, atol=1e-7)
        for parameter, reference_parameter in zip(
            model.parameters(), reference.parameters()
        ):
            torch.testing.assert_close(
                parameter.grad, reference_parameter.grad, rtol=1e-6, atol=1e-7
            )
        optimizer.step()
        reference_optimizer.step()
        for parameter, reference_parameter in zip(
            model.parameters(), reference.parameters()
        ):
            torch.testing.assert_close(
                parameter, reference_parameter, rtol=1e-6, atol=1e-7
            )
    finally:
        dist.destroy_process_group()


def test_native_dp_matches_unsplit_global_batch(tmp_path):
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("Gloo unavailable")
    rendezvous = (Path(tmp_path) / "rendezvous").as_uri()
    torch.multiprocessing.spawn(
        _native_dp_worker, args=(rendezvous,), nprocs=2, join=True
    )
