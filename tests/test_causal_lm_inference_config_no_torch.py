# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午10:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Strict workload-owned local generation configuration."""

import argparse
import json

import pytest


def parse(inference=None, **sections):
    from parascale.workloads.specs.causal_lm_inference import CausalLmInferenceSpec

    return CausalLmInferenceSpec.from_config(
        {
            "model": {"pretrained_model_name_or_path": "/local/model"},
            "data": {"type": "prompts_jsonl", "path": "/local/prompts.jsonl"},
            "inference": {"workload": "causal_lm", **(inference or {})},
            **sections,
        }
    )


def test_default_float_policy_is_explicit_and_bounded():
    spec = parse()
    assert spec.execution == "float"
    assert spec.device == "cuda:0" and spec.precision == "bf16"
    assert spec.batch_size == 1 and spec.max_outputs == 32
    assert spec.max_input_tokens == 256 and spec.max_new_tokens == 128
    assert spec.cache_implementation == "dynamic" and spec.compile_mode == "off"


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", True),
        ("max_outputs", None),
        ("max_input_tokens", 0),
        ("max_new_tokens", -1),
        ("warmup_steps", 1.5),
        ("max_requests", "10"),
        ("max_line_bytes", 0),
        ("do_sample", True),
        ("unknown", 1),
        ("execution", "auto"),
        ("precision", "fp8"),
    ],
)
def test_invalid_options_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        parse({field: value})


@pytest.mark.parametrize(
    "execution,artifact",
    [
        ("reference", None),
        ("gemlite_int8", None),
        ("float", "/artifact.pt"),
    ],
)
def test_artifact_and_execution_must_agree(execution, artifact):
    with pytest.raises(ValueError, match="artifact|execution"):
        parse({"execution": execution, "artifact_path": artifact})


def test_explicit_cpu_functionality_requires_fp32():
    assert parse({"precision": "fp32"}, runtime={"device": "cpu"}).device == "cpu"
    with pytest.raises(ValueError, match="precision|CPU"):
        parse(runtime={"device": "cpu"})


def test_cuda_float_fp16_is_an_explicit_inference_policy():
    spec = parse({"precision": "fp16"})
    assert spec.precision == "fp16" and spec.execution == "float"
    assert spec.cache_implementation == "dynamic"


@pytest.mark.parametrize(
    "policy,runtime",
    [
        ({}, {"device": "cpu"}),
        ({"cache_implementation": "static"}, {}),
        ({"execution": "reference", "artifact_path": "/artifact.pt"}, {}),
        ({"execution": "gemlite_int8", "artifact_path": "/artifact.pt"}, {}),
    ],
)
def test_fp16_does_not_enable_unqualified_execution_combinations(policy, runtime):
    with pytest.raises(ValueError, match="FP16|fp16|CPU|BF16"):
        parse({"precision": "fp16", **policy}, runtime=runtime)


@pytest.mark.parametrize(
    "policy",
    [
        {"compile_mode": "max-autotune"},
        {"cache_implementation": "static", "batch_size": 2},
        {
            "cache_implementation": "static",
            "compile_mode": "max-autotune",
            "precision": "fp32",
        },
    ],
)
def test_unqualified_optimized_tuples_reject(policy):
    with pytest.raises(ValueError, match="static|batch_size|precision|BF16"):
        parse(policy)


def test_supported_optimized_policy_has_no_silent_defaults():
    spec = parse(
        {
            "cache_implementation": "static",
            "compile_mode": "max-autotune",
            "attention_policy": "math",
            "max_input_tokens": 128,
        }
    )
    assert spec.attention_policy == "math" and spec.max_input_tokens == 128


@pytest.mark.parametrize("device", ["auto", "npu:0", "cuda:-1", "cuda:1junk"])
def test_device_is_explicit(device):
    with pytest.raises(ValueError, match="device"):
        parse(runtime={"device": device})


@pytest.mark.parametrize(
    "field",
    [
        "execution",
        "precision",
        "attention_policy",
        "cache_implementation",
        "compile_mode",
    ],
)
@pytest.mark.parametrize("value", [[], {}])
def test_non_scalar_policies_have_actionable_errors(field, value):
    with pytest.raises(ValueError, match=field):
        parse({field: value})


@pytest.mark.parametrize("accelerator", [[], {}])
def test_non_scalar_accelerator_has_actionable_error(accelerator):
    with pytest.raises(ValueError, match="accelerator"):
        parse(runtime={"accelerator": accelerator})


@pytest.mark.parametrize(
    "collision",
    [
        "summary_prompt",
        "summary_config",
        "summary_model",
        "summary_artifact",
        "summary_stream",
        "summary_partial",
        "run_model",
        "run_prompt",
        "run_config",
        "stream_config_artifact",
        "stream_config",
        "existing_summary",
    ],
)
def test_cli_rejects_output_collisions_before_any_write(
    tmp_path, monkeypatch, collision
):
    from parascale.commands import run

    model = tmp_path / "model"
    model.mkdir()
    source = tmp_path / "input.json"
    prompt = tmp_path / "prompts.jsonl"
    prompt.write_text('{"prompt":"hello"}\n', encoding="utf-8")
    summary = tmp_path / "summary.json"
    stream = tmp_path / "records.jsonl"
    run_dir = tmp_path / "run"
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"protected artifact")
    config = {
        "model": {"pretrained_model_name_or_path": str(model)},
        "data": {"type": "prompts_jsonl", "path": str(prompt)},
        "runtime": {"device": "cpu", "run_dir": str(run_dir)},
        "inference": {
            "workload": "causal_lm",
            "precision": "fp32",
            "output_jsonl": str(stream),
        },
    }
    if collision.startswith("summary_"):
        summary = {
            "summary_prompt": prompt,
            "summary_config": source,
            "summary_model": model / "new.json",
            "summary_artifact": artifact,
            "summary_stream": stream,
            "summary_partial": stream.with_suffix(".jsonl.partial"),
        }[collision]
    if collision == "summary_artifact":
        config["inference"].update(execution="reference", artifact_path=str(artifact))
    if collision == "run_model":
        config["runtime"]["run_dir"] = str(model)
    if collision in {"run_prompt", "run_config"}:
        config["runtime"]["run_dir"] = str(tmp_path)
    if collision == "stream_config_artifact":
        config["inference"]["output_jsonl"] = str(run_dir / "config.resolved.json")
    if collision == "stream_config":
        config["inference"]["output_jsonl"] = str(source)
    if collision == "existing_summary":
        summary.write_text("previous result", encoding="utf-8")
    source.write_text(json.dumps(config), encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Path validation must precede artifact writes and model loading")

    monkeypatch.setattr(run, "_write_command_config_artifacts", forbidden)
    monkeypatch.setattr(run, "run_inference_from_config", forbidden)
    with pytest.raises(
        (ValueError, FileExistsError), match="overlap|exist|output|directory"
    ):
        run.cmd_infer(argparse.Namespace(config=str(source), output=str(summary)))
    assert {path: path.read_bytes() for path in before} == before
