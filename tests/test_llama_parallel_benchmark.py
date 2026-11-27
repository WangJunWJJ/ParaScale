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
from types import SimpleNamespace

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


def test_parameter_fingerprint_uses_ordered_values_not_parameter_names():
    from tests.benchmarks.scripts.benchmark_llama_parallel import parameter_fingerprint

    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    clone = copy.deepcopy(model)
    renamed = torch.nn.Sequential(
        torch.nn.ModuleDict({"first": clone[0], "second": clone[1]})
    )
    expected = parameter_fingerprint(model.parameters())
    assert parameter_fingerprint(renamed.parameters()) == expected
    assert parameter_fingerprint(reversed(list(clone.parameters()))) != expected
    with torch.no_grad():
        clone[0].weight[0, 0].add_(1)
    assert parameter_fingerprint(clone.parameters()) != expected


def test_parameter_fingerprint_distinguishes_equal_bytes_with_different_metadata():
    from tests.benchmarks.scripts.benchmark_llama_parallel import parameter_fingerprint

    tensor = torch.zeros(2, 3, dtype=torch.float32)
    expected = parameter_fingerprint([tensor])
    assert parameter_fingerprint([tensor.view(torch.int32)]) != expected
    assert parameter_fingerprint([tensor.reshape(3, 2)]) != expected
    assert parameter_fingerprint([tensor.T]) == parameter_fingerprint(
        [tensor.T.contiguous()]
    )


@pytest.mark.parametrize(
    "dp,local_failure,peer_record,expected_error",
    [
        (2, True, (None, "peer"), "rank 0:.*local copy failed"),
        (1, False, ("peer copy failed", None), "rank 1:.*peer copy failed"),
        (2, False, (None, "peer"), "Initial parameters differ"),
        (1, False, (None, "peer"), None),
    ],
)
def test_initial_fingerprint_coordinates_errors_and_compares_only_dp(
    monkeypatch, dp, local_failure, peer_record, expected_error
):
    from tests.benchmarks.scripts import benchmark_llama_parallel as benchmark

    def fingerprint(parameters):
        if local_failure:
            raise OSError("local copy failed")
        return "local"

    def gather(records, local_record):
        assert local_record == (
            ("OSError: local copy failed", None) if local_failure else (None, "local")
        )
        records[:] = [local_record, peer_record]

    monkeypatch.setattr(benchmark, "parameter_fingerprint", fingerprint)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    if expected_error:
        with pytest.raises((ValueError, RuntimeError), match=expected_error):
            benchmark._initial_parameter_fingerprint([], world=2, dp=dp)
    else:
        assert benchmark._initial_parameter_fingerprint([], world=2, dp=dp) == "local"


def test_benchmark_bf16_hook_requires_pure_dp(tmp_path):
    args = _parse_native(
        tmp_path / "new.json",
        "--dp",
        "4",
        "--microbatch",
        "4",
        "--ddp-comm-hook",
        "bf16_compress",
    )
    assert args.ddp_comm_hook == "bf16_compress"
    for topology in ((), ("--tp", "2"), ("--dp", "2", "--tp", "2")):
        with pytest.raises(SystemExit) as error:
            _parse_native(
                tmp_path / "new.json", *topology, "--ddp-comm-hook", "bf16_compress"
            )
        assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        _parse_native(
            tmp_path / "new.json", "--dp", "2", "--ddp-comm-hook", "fp16_compress"
        )
    assert error.value.code == 2


def test_benchmark_hook_registers_official_hook_with_explicit_group():
    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
        bf16_compress_hook,
    )

    from tests.benchmarks.scripts.benchmark_llama_parallel import (
        register_benchmark_ddp_hook,
    )

    calls = []
    group = object()
    model = SimpleNamespace(
        process_group=group, register_comm_hook=lambda **kwargs: calls.append(kwargs)
    )
    register_benchmark_ddp_hook(None, "none")
    register_benchmark_ddp_hook(model, "bf16_compress")
    assert calls == [{"state": group, "hook": bf16_compress_hook}]
    with pytest.raises(ValueError, match="unsupported"):
        register_benchmark_ddp_hook(model, "fp16_compress")
    model.process_group = None
    with pytest.raises(ValueError, match="process group"):
        register_benchmark_ddp_hook(model, "bf16_compress")
    assert len(calls) == 1


def test_benchmark_default_hook_setup_does_not_touch_cuda_or_collectives(monkeypatch):
    from tests.benchmarks.scripts.benchmark_llama_parallel import (
        _setup_benchmark_ddp_hook,
    )

    def unexpected(*args, **kwargs):
        pytest.fail("uncompressed setup must not check capability or communicate")

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", unexpected)
    monkeypatch.setattr(torch.distributed, "all_gather_object", unexpected)
    _setup_benchmark_ddp_hook(None, "none", torch.device("cpu"))


@pytest.mark.parametrize(
    "failure", ["wrapper", "device", "version", "registration", "peer"]
)
def test_benchmark_candidate_coordinates_setup_errors(monkeypatch, failure):
    from tests.benchmarks.scripts import benchmark_llama_parallel as benchmark

    ddp_type = type("DDP", (), {})
    model = ddp_type() if failure != "wrapper" else object()
    if isinstance(model, ddp_type):
        model.process_group = object()
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", ddp_type)
    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", lambda **kwargs: failure != "device"
    )
    monkeypatch.setattr(
        torch.cuda.nccl,
        "version",
        lambda: (2, 9, 6) if failure == "version" else (2, 10, 0),
    )
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    registrations = []

    def register(wrapper, name):
        registrations.append((wrapper, name))
        if failure == "registration":
            raise RuntimeError("injected registration failure")

    monkeypatch.setattr(
        benchmark, "register_benchmark_ddp_hook", register, raising=False
    )
    gathered = []

    def gather(errors, local_error):
        gathered.append(local_error)
        errors[:] = [
            local_error,
            "injected peer failure" if failure == "peer" else None,
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="rank 1" if failure == "peer" else "rank 0"):
        benchmark._setup_benchmark_ddp_hook(
            model, "bf16_compress", torch.device("cuda:0")
        )
    assert len(gathered) == 1
    assert bool(gathered[0]) == (failure != "peer")
    assert registrations == (
        [(model, "bf16_compress")] if failure in {"registration", "peer"} else []
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
    assert args.ddp_comm_hook == "none"
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
