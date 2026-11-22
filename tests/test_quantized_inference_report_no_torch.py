# -*- coding: utf-8 -*-
# @Time : 2026/11/21 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Acceptance must be based on comparable measurements, never conversion alone."""

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parent / "benchmarks/scripts/quantized_inference_report.py"
spec = importlib.util.spec_from_file_location("quantized_report", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def profile_event(key, device, count, cpu_us=0, device_us=0, annotation=False):
    return SimpleNamespace(
        key=key,
        device_type=SimpleNamespace(name=device),
        count=count,
        self_cpu_time_total=cpu_us,
        self_device_time_total=device_us,
        is_user_annotation=annotation,
    )


def test_profile_separates_cuda_activities_from_cpu_linked_device_time():
    events = [
        profile_event("gemm", "CPU", 2, cpu_us=40, device_us=999),
        profile_event("gemm", "CUDA", 2, device_us=80),
        profile_event("attention", "CUDA", 1, device_us=20),
        profile_event("annotation", "CUDA", 1, device_us=1000, annotation=True),
        profile_event("cudaDeviceSynchronize", "CPU", 1, cpu_us=60),
    ]
    actual = report.summarize_profile(events, wall_ms=2, top_k=1)
    assert actual["cuda_activities_observed"]
    assert actual["cuda"]["count"] == 3
    assert actual["cuda"]["self_us"] == 100
    assert actual["cuda"]["top"][0] == {
        "name": "gemm",
        "count": 2,
        "self_us": 80,
        "mean_self_us": 40,
        "fraction_of_activity_time": 0.8,
    }
    assert actual["cuda"]["other_count"] == 1
    assert actual["cuda"]["other_self_us"] == 20
    assert actual["cpu"]["self_us"] == 100
    assert actual["cpu"]["top"][0]["name"] == "cudaDeviceSynchronize"
    assert actual["timing_used_for_throughput"] is False
    assert actual["iterations"] == 1


def test_empty_profile_does_not_claim_cuda_evidence():
    actual = report.summarize_profile([], wall_ms=1)
    assert not actual["cuda_activities_observed"]
    assert actual["cuda"]["top"] == []
    assert actual["cuda"]["self_us"] == 0


def test_quantized_kernel_evidence_requires_matching_cuda_activity():
    events = [
        profile_event("gemm_INT_kernel", "CPU", 1, cpu_us=10),
        profile_event("attention", "CUDA", 1, device_us=100),
        profile_event("gemm_INT_kernel", "CUDA", 1, device_us=0),
        profile_event("gemv_INT_kernel", "CUDA", 1, device_us=10, annotation=True),
    ]
    assert report.quantized_cuda_kernels(events, backend="gemlite_int8") == []
    events.append(profile_event("gemv_INT_kernel", "CUDA", 1, device_us=1))
    assert report.quantized_cuda_kernels(events, backend="gemlite_int8") == [
        "gemv_INT_kernel"
    ]
    assert report.quantized_cuda_kernels(events, backend="torchao_int4") == []


def test_quantized_kernel_evidence_is_not_limited_to_top_k():
    events = [profile_event("attention", "CUDA", 100, device_us=1000)]
    events.append(profile_event("gemm_INT_kernel", "CUDA", 1, device_us=1))
    assert (
        report.summarize_profile(events, wall_ms=2, top_k=1)["cuda"]["top"][0]["name"]
        == "attention"
    )
    assert report.quantized_cuda_kernels(events, backend="gemlite_int8") == [
        "gemm_INT_kernel"
    ]


def test_int4_kernel_evidence_excludes_cpu_operator_and_packing():
    events = [
        profile_event("aten::_weight_int4pack_mm", "CPU", 1, cpu_us=10),
        profile_event("matrix_to_m16n8k16_Bint4_layout", "CUDA", 1, device_us=2),
    ]
    assert report.quantized_cuda_kernels(events, backend="torchao_int4") == []
    kernel = "void at::native::tinygemm_m16n8k16_chunk_kernel<...>()"
    events.append(profile_event(kernel, "CUDA", 1, device_us=1))
    assert report.quantized_cuda_kernels(events, backend="torchao_int4") == [kernel]
    assert report.quantized_cuda_kernels(events, backend="unknown") == []


@pytest.mark.parametrize("kernel", ["gemv_INT_splitK_kernel", "gemm_splitK_INT_kernel"])
def test_quantized_kernel_evidence_accepts_split_k(kernel):
    events = [profile_event(kernel, "CUDA", 1, device_us=2)]
    assert report.quantized_cuda_kernels(events, backend="gemlite_int8") == [kernel]


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_profile_rejects_invalid_activity_time(value):
    with pytest.raises(ValueError, match="profile"):
        report.summarize_profile(
            [profile_event("kernel", "CUDA", 1, device_us=value)], wall_ms=1
        )


def test_benchmark_profile_trace_is_opt_in():
    from tests.benchmarks.scripts.benchmark_quantized_inference import parser

    required = [
        "--model",
        "m",
        "--data",
        "d",
        "--data-description",
        "test",
        "--output",
        "o",
    ]
    assert not parser().parse_args(required).export_profile_trace
    assert (
        parser().parse_args([*required, "--export-profile-trace"]).export_profile_trace
    )


@pytest.mark.parametrize("mode", ["fast", "max"])
def test_gemlite_tuning_is_explicit_and_resets_cache_before_search(monkeypatch, mode):
    from tests.benchmarks.scripts import benchmark_quantized_inference as benchmark

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "parascale.quantization.inference_gemlite",
        SimpleNamespace(_load_gemlite=lambda: calls.append("validate")),
    )
    monkeypatch.setitem(
        sys.modules,
        "gemlite",
        SimpleNamespace(
            reset_config=lambda: calls.append("reset"),
            set_autotune=lambda config: calls.append(config),
        ),
    )
    assert benchmark.configure_gemlite_tuning(None) is None
    assert calls == []
    policy = benchmark.configure_gemlite_tuning(mode)
    assert calls == ["validate", "reset", {"GEMV_SPLITK": mode}]
    assert policy == {"gemv_splitk": mode, "initial_cache": "empty_all_modes"}


def test_gemlite_tuning_rejects_unrelated_backend(monkeypatch):
    from tests.benchmarks.scripts import benchmark_quantized_inference as benchmark

    args = [
        "benchmark",
        "--model",
        "m",
        "--data",
        "d",
        "--data-description",
        "test",
        "--output",
        "o",
        "--gemlite-gemv-autotune",
        "max",
    ]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="requires --backend gemlite_int8"):
        benchmark.main()


def test_gemlite_tuning_evidence_validates_and_hashes_winners(monkeypatch):
    from tests.benchmarks.scripts import benchmark_quantized_inference as benchmark

    configs = {"GEMV_SPLITK": {"(1, 256, 512, 512, 1, 108)": {"BLOCK_SIZE_K": 64}}}
    validated = []
    monkeypatch.setitem(
        sys.modules,
        "gemlite",
        SimpleNamespace(
            cache_config=lambda path: Path(path).write_text(json.dumps(configs))
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "parascale.quantization.inference_gemlite",
        SimpleNamespace(_validate_kernel_cache=validated.append),
    )
    evidence = benchmark.gemlite_tuning_evidence({(512, 256)})
    assert evidence["winning_configs"] == configs
    assert validated == [configs]
    assert len(evidence["config_sha256"]) == 64
    with pytest.raises(RuntimeError, match="Missing GEMV tuning evidence"):
        benchmark.gemlite_tuning_evidence({(1024, 256)})
    configs["GEMV_SPLITK"] = {"(1, 256, 512, 128, 1, 108)": {"BLOCK_SIZE_K": 64}}
    with pytest.raises(RuntimeError, match="Missing GEMV tuning evidence"):
        benchmark.gemlite_tuning_evidence({(512, 256)})


def test_profile_summary_observes_real_cuda_graph_activities():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA profiler")
    inputs = torch.randn(256, 256, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            inputs @ inputs
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = inputs @ inputs
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as trace:
        graph.replay()
        torch.cuda.synchronize()
    actual = report.summarize_profile(trace.key_averages(), wall_ms=1)
    assert output.shape == (256, 256)
    assert actual["cuda_activities_observed"]
    assert actual["cuda"]["count"] > 0
    assert actual["cuda"]["self_us"] > 0
    assert actual["cpu"]["count"] > 0


def result(mode, latency, ppl):
    generation_quality = {
        "segments": 1,
        "discarded_tokens": 0,
        "prompt_sha256": "prompt",
        "target_sha256": "targets",
        "generation_config": {"do_sample": False, "max_new_tokens": 4},
        "execution_evidence": {
            "prefill_excluded": True,
            "compiled_regions": [],
            "quantized_cuda_kernels": ["tinygemm_m16n8k16_chunk_kernel"]
            if mode == "int4"
            else [],
        },
    }
    for name, count in (("first_token", 1), ("decode", 3), ("all", 4)):
        generation_quality[name] = {
            "tokens": count,
            "nll_sum": count * math.log(ppl),
            "perplexity": ppl,
        }
    return {
        "mode": mode,
        "contract": {
            "model_sha256": "abc",
            "data_sha256": "def",
            "batch_size": 1,
            "new_tokens": 4,
            "sequence_length": 2,
            "eval_tokens_requested": 6,
        },
        "environment": {"gpu_uuid": "GPU-1", "torch": "2.9.1"},
        "quality": {"perplexity": ppl, "tokens": 1000},
        "generation_quality": generation_quality,
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


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "decode",
        "hash",
        "tokens",
        "kernel",
        "truncated",
        "negative_tail",
        "missing_length",
        "prefill_only",
        "annotation",
        "configuration",
        "null_evidence",
        "list_evidence",
    ],
)
def test_generation_quality_is_required_even_when_prefill_quality_passes(invalid):
    baseline, candidate = result("bf16", 100, 10), result("int4", 75, 10)
    quality = candidate["generation_quality"]
    if invalid == "missing":
        del candidate["generation_quality"]
    elif invalid == "decode":
        quality["decode"]["perplexity"] = 12
        quality["decode"]["nll_sum"] = 3 * math.log(12)
        quality["all"]["nll_sum"] = (
            quality["first_token"]["nll_sum"] + quality["decode"]["nll_sum"]
        )
        quality["all"]["perplexity"] = math.exp(quality["all"]["nll_sum"] / 4)
    elif invalid == "hash":
        quality["target_sha256"] = "other"
    elif invalid == "tokens":
        quality["decode"]["tokens"] = 2
    elif invalid == "kernel":
        quality["execution_evidence"]["quantized_cuda_kernels"] = []
    elif invalid == "truncated":
        for item in (baseline, candidate):
            item["contract"]["eval_tokens_requested"] = 600
    elif invalid == "negative_tail":
        for item in (baseline, candidate):
            item["generation_quality"]["discarded_tokens"] = -1
    elif invalid == "missing_length":
        for item in (baseline, candidate):
            del item["contract"]["sequence_length"]
    elif invalid == "prefill_only":
        quality["execution_evidence"]["prefill_excluded"] = False
    elif invalid == "configuration":
        quality["generation_config"]["do_sample"] = True
    elif invalid == "null_evidence":
        quality["execution_evidence"] = None
    elif invalid == "list_evidence":
        quality["execution_evidence"] = []
    else:
        for item in (baseline, candidate):
            item["contract"]["compile"] = True
            item["profiled_compiled_regions"] = ["Torch-Compiled Region"]
            item["generation_quality"]["execution_evidence"]["compiled_regions"] = [
                "user compiled annotation"
            ]
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
    for item in (baseline, candidate):
        item["generation_quality"]["execution_evidence"]["compiled_regions"] = [
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


@pytest.mark.parametrize("missing", ["hash", "codes", "requantized"])
def test_artifact_comparison_requires_preserved_stored_quantization(missing):
    baseline, candidate = result("bf16", 100, 10), result("int4", 75, 10)
    candidate["mode"] = "int8"
    candidate["conversion"].update(
        execution="int8_weight_only_fused_bf16",
        source_format="parascale.reference_quantization",
        stored_codes_and_scales_preserved=True,
        requantized=False,
    )
    candidate["generation_quality"]["execution_evidence"]["quantized_cuda_kernels"] = [
        "gemv_INT_splitK_kernel"
    ]
    for item in (baseline, candidate):
        item["contract"]["reference_artifact"] = {"sha256": "artifact-hash"}
    assert report.compare_results(baseline, candidate)["engineering_gate_passed"]
    if missing == "hash":
        for item in (baseline, candidate):
            item["contract"]["reference_artifact"]["sha256"] = ""
    elif missing == "codes":
        candidate["conversion"]["stored_codes_and_scales_preserved"] = False
    else:
        candidate["conversion"]["requantized"] = True
    with pytest.raises(ValueError, match="artifact"):
        report.compare_results(baseline, candidate)
