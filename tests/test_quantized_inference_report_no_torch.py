# -*- coding: utf-8 -*-
# @Time : 2026/11/21 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Acceptance must be based on comparable measurements, never conversion alone."""

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "benchmarks/scripts/quantized_inference_report.py"
spec = importlib.util.spec_from_file_location("quantized_report", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def result(mode, latency, ppl):
    return {
        "mode": mode,
        "contract": {"model_sha256": "abc", "data_sha256": "def", "batch_size": 1},
        "environment": {"gpu_uuid": "GPU-1", "torch": "2.9.1"},
        "quality": {"perplexity": ppl, "tokens": 1000},
        "timing": {"generation": {"wall_ms": [latency] * 20}},
        "conversion": (
            {"converted_layers": ["linear"], "execution": "int4_weight_only_fused_bf16"}
            if mode == "int4"
            else None
        ),
        "kernel_verified": mode == "int4",
    }


def test_real_speedup_and_quality_must_both_pass():
    baseline = result("bf16", 100, 10)
    candidate = result("int4", 75, 10.2)
    summary = report.compare_results(baseline, candidate)
    assert summary["engineering_gate_passed"]
    assert summary["speedup"] == pytest.approx(4 / 3)
    candidate["quality"]["perplexity"] = 11
    assert not report.compare_results(baseline, candidate)["engineering_gate_passed"]


def test_mislabeled_precision_is_rejected():
    baseline, candidate = result("bf16", 100, 10), result("int4", 75, 10)
    candidate["mode"] = "int8"
    with pytest.raises(ValueError, match="actual conversion"):
        report.compare_results(baseline, candidate)


@pytest.mark.parametrize(
    "latency,ppl", [(110, 10), (75, float("nan")), (75, float("inf")), (0, 10)]
)
def test_bad_or_slower_measurements_do_not_pass(latency, ppl):
    assert not report.compare_results(
        result("bf16", 100, 10), result("int4", latency, ppl)
    )["engineering_gate_passed"]


@pytest.mark.parametrize("field", ["contract", "environment"])
def test_different_inputs_or_environment_are_not_comparable(field):
    baseline = result("bf16", 100, 10)
    candidate = result("int4", 75, 10)
    candidate[field] = copy.deepcopy(baseline[field])
    candidate[field][next(iter(candidate[field]))] = "different"
    with pytest.raises(ValueError, match="comparable"):
        report.compare_results(baseline, candidate)


def test_no_fused_kernel_or_no_coverage_cannot_pass():
    baseline, candidate = result("bf16", 100, 10), result("int4", 75, 10)
    candidate["kernel_verified"] = False
    assert not report.compare_results(baseline, candidate)["engineering_gate_passed"]
    candidate["kernel_verified"] = True
    candidate["conversion"]["converted_layers"] = []
    assert not report.compare_results(baseline, candidate)["engineering_gate_passed"]


def test_requested_compile_requires_observed_compiled_regions():
    baseline, candidate = result("bf16", 100, 10), result("int4", 75, 10)
    baseline["contract"]["compile"] = candidate["contract"]["compile"] = True
    assert not report.compare_results(baseline, candidate)["engineering_gate_passed"]
    baseline["profiled_compiled_regions"] = candidate["profiled_compiled_regions"] = [
        "Torch-Compiled Region"
    ]
    assert report.compare_results(baseline, candidate)["engineering_gate_passed"]


def test_empty_provenance_is_rejected():
    baseline, candidate = result("bf16", 100, 10), result("int4", 75, 10)
    baseline["contract"]["model_sha256"] = candidate["contract"]["model_sha256"] = ""
    with pytest.raises(ValueError, match="provenance"):
        report.compare_results(baseline, candidate)


def test_benchmark_resolves_paths_before_launch_from_other_directory(
    tmp_path, monkeypatch
):
    import sys

    benchmark_spec = importlib.util.spec_from_file_location(
        "quantized_benchmark", SCRIPT.with_name("benchmark_quantized_inference.py")
    )
    benchmark = importlib.util.module_from_spec(benchmark_spec)
    benchmark_spec.loader.exec_module(benchmark)
    captured = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            "model",
            "--data",
            "eval.txt",
            "--data-description",
            "held-out",
            "--output",
            "runs/result.json",
        ],
    )
    monkeypatch.setattr(benchmark, "run_comparison", captured.append)
    benchmark.main()
    assert captured[0].model == tmp_path / "model"
    assert captured[0].data == tmp_path / "eval.txt"
    assert captured[0].output == tmp_path / "runs/result.json"


def test_benchmark_exposes_selective_layer_quantization():
    benchmark_spec = importlib.util.spec_from_file_location(
        "quantized_benchmark_selective",
        SCRIPT.with_name("benchmark_quantized_inference.py"),
    )
    benchmark = importlib.util.module_from_spec(benchmark_spec)
    benchmark_spec.loader.exec_module(benchmark)
    args = benchmark.parser().parse_args(
        [
            "--model",
            "model",
            "--data",
            "eval.txt",
            "--data-description",
            "heldout",
            "--output",
            "result.json",
            "--exclude-modules",
            "lm_head",
            "self_attn",
        ]
    )
    assert args.exclude_modules == ["lm_head", "self_attn"]
