# -*- coding: utf-8 -*-
# @Time : 2026/11/24 下午9:18
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Diffusion policies reject unsupported work before loading any GPU assets."""

import argparse
import json

import pytest


def config(tmp_path, **options):
    return {
        "model": {"pretrained_model_name_or_path": str(tmp_path / "model")},
        "data": {"type": "prompts_jsonl", "path": str(tmp_path / "prompts.jsonl")},
        "runtime": {"device": "cuda:0", "run_dir": str(tmp_path / "run")},
        "inference": {
            "workload": "diffusion",
            "output_dir": str(tmp_path / "run" / "images"),
            **options,
        },
    }


def test_diffusion_default_preserves_sampling_budget(tmp_path):
    from parascale.workloads.specs.diffusion import DiffusionInferenceSpec

    spec = DiffusionInferenceSpec.from_config(config(tmp_path))
    assert (spec.height, spec.width, spec.num_inference_steps) == (512, 512, 30)
    assert spec.precision == "fp16" and spec.compile_mode == "off"
    assert spec.seed == 42 and spec.batch_size == 1


@pytest.mark.parametrize(
    "option,value",
    [
        ("precision", "int8"),
        ("compile_mode", "auto"),
        ("height", 513),
        ("width", 0),
        ("batch_size", True),
        ("seed", -1),
        ("num_inference_steps", 0),
        ("guidance_scale", float("nan")),
        ("guidance_scale", True),
        ("warmup_steps", -1),
        ("scheduler", "faster"),
        ("safety_checker", False),
        ("max_requests", 0),
        ("max_line_bytes", 0),
    ],
)
def test_invalid_diffusion_options_fail(tmp_path, option, value):
    from parascale.workloads.specs.diffusion import DiffusionInferenceSpec

    with pytest.raises(ValueError, match=option):
        DiffusionInferenceSpec.from_config(config(tmp_path, **{option: value}))


@pytest.mark.parametrize(
    "case", ["model", "input", "source", "summary", "existing", "run"]
)
def test_diffusion_cli_rejects_paths_before_writes(tmp_path, monkeypatch, case):
    from parascale.commands import run

    data = config(tmp_path)
    source = tmp_path / "source.json"
    summary = tmp_path / "run" / "summary.json"
    if case == "model":
        data["inference"]["output_dir"] = str(tmp_path / "model" / "images")
    elif case == "input":
        data["inference"]["output_dir"] = str(tmp_path / "prompts.jsonl")
    elif case == "source":
        data["inference"]["output_dir"] = str(source)
    elif case == "summary":
        summary = tmp_path / "run" / "images" / "summary.json"
    elif case == "run":
        data["runtime"]["run_dir"] = str(tmp_path)
    else:
        (tmp_path / "run" / "images").mkdir(parents=True)
    source.write_text(json.dumps(data), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("validation must precede writes and model loading")

    monkeypatch.setattr(run, "_write_command_config_artifacts", forbidden)
    monkeypatch.setattr(run, "run_inference_from_config", forbidden)
    with pytest.raises((ValueError, FileExistsError), match="overlap|exist"):
        run.cmd_infer(argparse.Namespace(config=str(source), output=str(summary)))


def test_diffusion_rejects_distributed_launch_before_import(tmp_path, monkeypatch):
    from parascale.workloads.inference import build_inference_components

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single.device|distributed"):
        build_inference_components(config(tmp_path))


def test_diffusion_requires_explicit_output_directory(tmp_path):
    from parascale.workloads.specs.diffusion import DiffusionInferenceSpec

    data = config(tmp_path)
    del data["inference"]["output_dir"]
    with pytest.raises(ValueError, match="output_dir"):
        DiffusionInferenceSpec.from_config(data)


@pytest.mark.parametrize(
    "precision,compile_mode", [("fp16", "off"), ("fp32", "max-autotune")]
)
def test_cpu_only_allows_eager_fp32(tmp_path, precision, compile_mode):
    from parascale.workloads.specs.diffusion import DiffusionInferenceSpec

    data = config(tmp_path, precision=precision, compile_mode=compile_mode)
    data["runtime"]["device"] = "cpu"
    with pytest.raises(ValueError, match="CPU"):
        DiffusionInferenceSpec.from_config(data)


# Batched execution must preserve the same model and sampling budget.
def test_batched_example_changes_only_batch_size():
    import json
    from pathlib import Path

    from parascale.workloads.specs.diffusion import DiffusionInferenceSpec

    root = (
        Path(__file__).resolve().parents[1]
        / "examples/gpu/example_016_diffusion_inference"
    )
    eager = json.loads((root / "eager.json").read_text(encoding="utf-8"))
    batched = json.loads((root / "batched.json").read_text(encoding="utf-8"))
    assert DiffusionInferenceSpec.from_config(batched).batch_size == 8
    batched["inference"]["batch_size"] = eager["inference"]["batch_size"]
    assert batched == eager
