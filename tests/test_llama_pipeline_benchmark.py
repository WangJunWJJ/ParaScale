# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午6:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""CPU contracts for the source-independent pipeline measurement script."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def arguments():
    return ["--model", "model", "--data-file", "train.txt", "--output", "out.json"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--dp", "2", "--tp", "2"],
        ["--dp", "0"],
        ["--tp", "3"],
        ["--pp", "1"],
        ["--microbatches", "0"],
        ["--microbatches", "1"],
        ["--global-batch", "0"],
        ["--global-batch", "7"],
        ["--dp", "2", "--global-batch", "6"],
        ["--sequence-length", "1"],
        ["--steps", "0"],
        ["--warmup", "-1"],
        ["--steps", "4", "--warmup", "4"],
        ["--precision", "fp16"],
        ["--cpu-affinity", "-1"],
    ],
)
def test_rejects_invalid_budgets_before_cuda(extra, monkeypatch):
    from tests.benchmarks.scripts.benchmark_llama_pipeline import parse_args

    monkeypatch.setattr(
        torch.cuda, "set_device", lambda *a: pytest.fail("CUDA touched")
    )
    with pytest.raises(SystemExit) as error:
        parse_args(arguments() + extra)
    assert error.value.code == 2


def test_defaults_and_explicit_modes():
    from tests.benchmarks.scripts.benchmark_llama_pipeline import parse_args

    args = parse_args(arguments())
    assert (args.dp, args.tp, args.pp, args.microbatches) == (1, 1, 2, 2)
    assert (args.global_batch, args.sequence_length, args.steps, args.warmup) == (
        16,
        256,
        16,
        4,
    )
    assert args.model == Path("model") and args.data_file == Path("train.txt")
    assert args.cpu_affinity is None and not args.return_outputs
    assert args.precision == "fp32"
    args = parse_args(
        arguments()
        + [
            "--tp",
            "2",
            "--precision",
            "bf16",
            "--return-outputs",
            "--cpu-affinity",
            "2",
            "4",
        ]
    )
    assert args.return_outputs and args.cpu_affinity == [2, 4]
    assert not parse_args(arguments() + ["--no-return-outputs"]).return_outputs


def test_refuses_existing_output(tmp_path):
    from tests.benchmarks.scripts.benchmark_llama_pipeline import parse_args

    path = tmp_path / "result.json"
    path.touch()
    with pytest.raises(SystemExit):
        parse_args(arguments() + ["--output", str(path)])


def test_retention_counts_backing_storage_once_per_container():
    from tests.benchmarks.scripts.benchmark_llama_pipeline import cache_memory

    logits = torch.zeros(4, 8, dtype=torch.bfloat16)
    hidden = torch.zeros(2, 8)
    stage = SimpleNamespace(
        output_chunks=[logits, logits[:1], logits],
        fwd_cache={0: ((logits,), [hidden]), 1: ((logits[:2],), [hidden])},
    )
    assert cache_memory(stage) == {
        "output_chunks_len": 3,
        "output_chunks_storage_bytes": 64,
        "fwd_cache_len": 2,
        "fwd_cache_storage_bytes": 128,
    }
    assert logits.requires_grad is False


def test_empty_caches_are_observations_not_requirements():
    from tests.benchmarks.scripts.benchmark_llama_pipeline import cache_memory

    assert cache_memory(SimpleNamespace(output_chunks=[], fwd_cache={})) == {
        "output_chunks_len": 0,
        "output_chunks_storage_bytes": 0,
        "fwd_cache_len": 0,
        "fwd_cache_storage_bytes": 0,
    }


def test_final_fingerprints_match_exact_model_and_optimizer_state():
    from tests.benchmarks.scripts.benchmark_llama_pipeline import (
        final_state_fingerprints,
    )
    from tests.benchmarks.scripts.validate_llama_checkpoint import state_fingerprint

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    expected = {
        "model": state_fingerprint(model.state_dict()),
        "optimizer": state_fingerprint(optimizer.state_dict()),
    }
    assert final_state_fingerprints(model, optimizer) == expected
    with torch.no_grad():
        model.weight.add_(1)
    changed = final_state_fingerprints(model, optimizer)
    assert changed["model"] != expected["model"]
    assert changed["optimizer"] == expected["optimizer"]
    optimizer.state[model.weight]["exp_avg"].add_(1)
    assert (
        final_state_fingerprints(model, optimizer)["optimizer"] != expected["optimizer"]
    )


@pytest.mark.parametrize(
    "invalid", ["parameter_dtype", "optimizer_dtype", "model_nan", "optimizer_inf"]
)
def test_final_state_rejects_wrong_dtype_and_nonfinite(invalid):
    from tests.benchmarks.scripts.benchmark_llama_pipeline import (
        final_state_fingerprints,
    )

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    if invalid == "parameter_dtype":
        model.double()
    elif invalid == "optimizer_dtype":
        optimizer.state[model.weight]["exp_avg"] = optimizer.state[model.weight][
            "exp_avg"
        ].double()
    elif invalid == "model_nan":
        with torch.no_grad():
            model.weight.fill_(float("nan"))
    else:
        optimizer.state[model.weight]["exp_avg"].fill_(float("inf"))
    with pytest.raises(ValueError, match="FP32|nonfinite"):
        final_state_fingerprints(model, optimizer)


def script_arguments(name):
    if name == "validate_pipeline_precision":
        return ["--output", "out.json"]
    if name == "validate_llama_parallel":
        return ["--pp", "2", "--output", "out.json"]
    result = arguments()
    if name == "validate_llama_checkpoint":
        result[result.index("--data-file")] = "--data"
        result += ["--pp", "2", "--checkpoint-dir", "checkpoint"]
    return result


@pytest.mark.parametrize(
    "name",
    [
        "benchmark_llama_pipeline",
        "validate_llama_parallel",
        "validate_llama_checkpoint",
        "validate_pipeline_precision",
    ],
)
@pytest.mark.parametrize("selection", [None, "gpipe", "1f1b"])
def test_schedule_cli_defaults_and_passes_selection_to_config(
    monkeypatch, name, selection
):
    module = importlib.import_module(f"tests.benchmarks.scripts.{name}")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda *a: pytest.fail("CUDA touched")
    )
    constructor = Mock(wraps=module.DistributedParallelConfig)
    monkeypatch.setattr(module, "DistributedParallelConfig", constructor)
    argv = script_arguments(name)
    if selection is not None:
        argv += ["--pipeline-schedule", selection]
    parsed = module.parse_args(argv)
    args = parsed[0] if isinstance(parsed, tuple) else parsed
    assert args.pipeline_schedule == (selection or "gpipe")
    assert constructor.call_args.kwargs["pipeline_schedule"] == args.pipeline_schedule
    if isinstance(parsed, tuple):
        assert parsed[1].pipeline_schedule == args.pipeline_schedule
        if name == "validate_pipeline_precision":
            assert (parsed[1].pp_size, parsed[1].microbatches) == (2, 2)


@pytest.mark.parametrize(
    "name",
    [
        "benchmark_llama_pipeline",
        "validate_llama_parallel",
        "validate_llama_checkpoint",
        "validate_pipeline_precision",
    ],
)
@pytest.mark.parametrize(
    "extra",
    [
        ["--pipeline-schedule", "interleaved"],
        ["--pipeline-schedule", "1f1b", "--pp", "1"],
        ["--pipeline-schedule", "1f1b", "--microbatches", "1"],
    ],
)
def test_schedule_cli_rejects_invalid_selection_before_cuda(monkeypatch, name, extra):
    module = importlib.import_module(f"tests.benchmarks.scripts.{name}")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda *a: pytest.fail("CUDA touched")
    )
    with pytest.raises(SystemExit) as error:
        module.parse_args(script_arguments(name) + extra)
    assert error.value.code == 2


@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
def test_boundary_validator_supports_dp2_with_either_schedule(schedule):
    from tests.benchmarks.scripts.validate_pipeline_precision import parse_args

    args, config = parse_args(
        ["--dp", "2", "--pipeline-schedule", schedule, "--output", "out.json"]
    )
    assert args.pipeline_schedule == config.pipeline_schedule == schedule
    assert (config.dp_size, config.pp_size, config.microbatches, config.world_size) == (
        2,
        2,
        2,
        4,
    )
