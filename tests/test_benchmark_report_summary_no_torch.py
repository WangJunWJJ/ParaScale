# -*- coding: utf-8 -*-
# @Time : 2026/7/20 下午3:05
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import json
from pathlib import Path

from tests.benchmarks.tools.build_benchmark_report import (
    build_report_markdown,
    load_summaries,
)


def test_quantization_report_distinguishes_smoke_from_quality():
    from tests.benchmarks.tools.build_benchmark_report import _quantization_section

    text = "\n".join(
        _quantization_section(
            {
                "reports": [
                    {
                        "arguments": {"bits": 4},
                        "ptq": {"perplexity": 64.0, "roundtrip_max_abs_error": 0.0},
                        "qat": {"training": {"nonzero_gradient_steps": 16}},
                    }
                ]
            }
        )
    )
    assert "synthetic" in text
    assert "not pretrained" in text
    assert "4-bit PTQ" in text
    assert "4-bit QAT" in text


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_clip_recovery_report_retains_real_data_and_timing_limits(tmp_path):
    summaries = load_summaries(tmp_path)
    summaries["clip_recovery"] = {
        "dataset": "COCO subset, NOT DataComp",
        "parity_checks": [
            {
                "name": "Continuous vs resumed",
                "model_max_abs": 1e-6,
                "model_relative_l2": 1e-8,
                "optimizer_max_abs": 1e-5,
                "evidence": "clip_recovery/comparison.json",
            }
        ],
        "parity_notes": ["Opt-in mitigation; not a cross-version guarantee."],
        "dataset_checks": [
            {
                "dataset": "Existing Flickr8k-labeled WDS",
                "notes": ["Original provenance not verified."],
                "evidence": "clip_recovery/flickr/summary.json",
                "runs": [
                    {"world_size": 1, "status": "passed", "weights_bitwise_equal": True}
                ],
            }
        ],
        "runs": [
            {
                "world_size": 2,
                "devices": "0,1",
                "status": "passed",
                "baseline_wall_seconds": 64,
                "weights_bitwise_equal": True,
            }
        ],
    }
    text = build_report_markdown(summaries, report_root=tmp_path)
    assert "COCO subset, NOT DataComp" in text
    assert "10.00" in text
    assert "no warmup exclusion" in text
    assert "not DataComp quality" in text
    assert "Continuous vs resumed" in text
    assert "clip_recovery/comparison.json" in text
    assert "not a cross-version guarantee" in text
    assert "Existing Flickr8k-labeled WDS" in text
    assert "Original provenance not verified" in text


def test_clip_validation_rejects_synthetic_fallback(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    from tests.benchmarks.scripts import validate_clip_recovery as validation

    output = tmp_path / "result.json"
    output.write_text(
        json.dumps({"workload": "synthetic_regression", "synthetic": True})
    )
    monkeypatch.setattr(
        validation,
        "execute_training",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    with pytest.raises(AssertionError):
        validation.run_case(tmp_path / "config.json", output, "0")


def test_clip_timeout_terminates_training_process_tree(monkeypatch):
    import os
    import subprocess
    from types import SimpleNamespace

    import pytest

    from tests.benchmarks.scripts import validate_clip_recovery as validation

    waits = []
    cleanup = []

    class Process:
        pid = 123456
        returncode = 0

        def wait(self, timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired("training", timeout)
            return 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(validation.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(
        validation.subprocess,
        "run",
        lambda command, **kw: cleanup.append(command) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        validation.os, "killpg", lambda *args: cleanup.append(args), raising=False
    )
    with pytest.raises(subprocess.TimeoutExpired):
        validation.execute_training(["training"], {}, None)
    assert len(waits) == 2
    if os.name == "nt":
        assert cleanup == [["taskkill", "/PID", "123456", "/T", "/F"]]
    else:
        assert cleanup[0][0] == 123456


def test_compression_examples_report_keeps_random_and_real_data_scope():
    from tests.benchmarks.tools.build_benchmark_report import (
        _compression_examples_section,
    )

    text = "\n".join(
        _compression_examples_section(
            {
                "scope": "Entry-point validation, not a quality claim.",
                "runs": [
                    {
                        "example": "Llama PTQ",
                        "device": "cuda",
                        "source": "random Llama / synthetic tokens",
                        "status": "passed",
                    },
                    {
                        "example": "YOLO KD",
                        "device": "cuda",
                        "source": "pretrained YOLO / COCO128",
                        "status": "passed",
                    },
                ],
                "notes": ["Pretrained Llama quality was not evaluated."],
            }
        )
    )
    assert "random Llama / synthetic tokens" in text
    assert "pretrained YOLO / COCO128" in text
    assert "Pretrained Llama quality was not evaluated." in text
    assert "not a quality claim" in text
    assert _compression_examples_section({}) == []


def test_normalization_optimization_report_keeps_comparison_scope():
    from tests.benchmarks.tools.build_benchmark_report import (
        _architecture_contracts_section,
    )

    text = "\n".join(
        _architecture_contracts_section(
            {
                "scope": "Original short-window measurement",
                "performance": [
                    {
                        "precision": "fp32",
                        "baseline_median": 100,
                        "fixed_median": 95,
                        "change_percent": -5,
                    }
                ],
                "normalization_optimization": {
                    "scope": "Synthetic MLP; 2000 steps; not real-model throughput",
                    "performance": [
                        {
                            "precision": "fp32",
                            "before_median": 95,
                            "after_median": 100,
                            "change_percent": 5.26,
                        }
                    ],
                },
                "accumulation_optimization": {
                    "scope": "Profiler counts are not throughput measurements",
                    "performance": [],
                },
            }
        )
    )
    assert "Original short-window measurement" in text
    assert "not real-model throughput" in text
    assert "+5.26%" in text
    assert "-5.00%" in text
    assert "Accumulation Statistics Optimization" in text
    assert "Profiler counts are not throughput measurements" in text


def test_distillation_report_distinguishes_smoke_from_quality():
    from tests.benchmarks.tools.build_benchmark_report import _distillation_section

    text = "\n".join(
        _distillation_section(
            {
                "reports": [
                    {
                        "parameter_counts": {"teacher": 100, "student": 50},
                        "distillation": {"after": {"perplexity": 64.0}},
                        "ce_baseline": {"after": {"perplexity": 65.0}},
                    }
                ]
            }
        )
    )
    assert "synthetic" in text
    assert "not pretrained" in text
    assert "CE baseline" in text
    assert "Distillation" in text
    assert "excludes batch transfer" in text


def test_detection_report_separates_throughput_and_quality():
    from tests.benchmarks.tools.build_benchmark_report import (
        _detection_distillation_section,
    )

    aggregate = {
        "steady_state_images_per_second": {"mean": 100, "stdev": 2},
        "end_to_end_images_per_second": {"mean": 75},
        "cuda_peak_allocated_bytes": {"mean": 1024**2 * 300},
    }
    text = "\n".join(
        _detection_distillation_section(
            {
                "aggregates": {"baseline": aggregate, "kd": aggregate},
                "evaluation": {
                    "baseline_0": {"mAP50_95": 0.4},
                    "kd_0": {"mAP50_95": 0.39},
                },
                "notes": ["No demonstrated accuracy gain."],
            }
        )
    )
    assert "End-to-end" in text
    assert "40.00" in text and "39.00" in text
    assert "No demonstrated accuracy gain" in text
    assert _detection_distillation_section({}) == []


def test_benchmark_report_combines_suite_summaries(tmp_path):
    _write_json(
        tmp_path / "dual_4090_full_validation" / "summary.json",
        {
            "passed": True,
            "hardware": "dual RTX 4090D",
            "image": "image-a",
            "summaries": [
                {
                    "model": "clip",
                    "ok_runs": 1,
                    "total_runs": 1,
                    "best_backend": "native_ddp",
                    "best_throughput": 10.0,
                    "best_loss": 1.5,
                    "best_peak_memory_bytes": 1024**3,
                }
            ],
        },
    )
    _write_json(
        tmp_path / "direct_pytorch_clip_comparison" / "summary.json",
        {
            "passed": True,
            "hardware": "dual RTX 4090D",
            "image": "image-a",
            "comparisons": [
                {
                    "direct": "torch_ddp",
                    "parascale": "parascale_native_ddp",
                    "direct_throughput": 8.0,
                    "parascale_throughput": 10.0,
                    "parascale_vs_direct": 1.25,
                }
            ],
            "deepspeed_comparisons": [],
        },
    )
    _write_json(
        tmp_path / "ascend_validation" / "summary.json",
        {
            "passed": True,
            "hardware": "Ascend 910B4",
            "image": "image-b",
            "runs": [
                {
                    "run_id": "doctor",
                    "ok": True,
                    "mode": "doctor",
                    "torch_npu": True,
                    "npu_device_count": 8,
                }
            ],
        },
    )
    _write_json(
        tmp_path / "ascend_parallel_matrix" / "summary.json",
        {
            "passed": True,
            "hardware": "Ascend 910B4",
            "image": "image-b",
            "scenarios": [
                {
                    "scenario": "single_docker_2card",
                    "containers": 1,
                    "cards": 2,
                    "ok": True,
                    "aggregate_throughput": 12.0,
                    "throughput_per_card": 6.0,
                }
            ],
        },
    )
    _write_json(
        tmp_path / "cross_hardware_clip_datacomp" / "summary.json",
        {
            "passed": True,
            "dataset": "datacomp",
            "model": "clip",
            "precision": "fp32",
            "runs": [
                {
                    "label": "rtx4090",
                    "hardware": "dual RTX 4090D",
                    "backend": "native_ddp",
                    "throughput": 10.0,
                    "runtime_status": "real_local",
                    "capability_level": "local_native_clip_datacomp",
                    "measurement_window": {
                        "warmup_steps_effective": 10,
                        "measured_batches": 100,
                    },
                }
            ],
            "comparisons": [
                {
                    "label": "rtx4090",
                    "relative_to_baseline": 1.0,
                }
            ],
        },
    )
    _write_json(
        tmp_path / "rtx4090_clip_precision_datacomp" / "summary.json",
        {
            "hardware": "dual RTX 4090D",
            "image": "image-a",
            "runs": [
                {
                    "precision": "fp32",
                    "backend": "native_ddp",
                    "throughput": 10.0,
                    "relative_to_fp32": 1.0,
                }
            ],
        },
    )
    _write_json(
        tmp_path / "a6000_native_ddp_scaling" / "summary.json",
        {
            "hardware": "5x RTX A6000",
            "image": "image-c",
            "dataset": "datacomp",
            "model": "clip_medium",
            "steps": 120,
            "warmup_steps": 20,
            "scaling": [
                {
                    "precision": "bf16",
                    "one_gpu_throughput": 100.0,
                    "two_gpu_throughput": 160.0,
                    "four_gpu_throughput": 280.0,
                    "scale_1_to_2": 1.6,
                    "scale_2_to_4": 1.75,
                    "scale_1_to_4": 2.8,
                    "efficiency_1_to_4": 0.7,
                }
            ],
            "hook_comparisons": [
                {
                    "gpus": 4,
                    "precision": "bf16",
                    "hook": "bf16_compress",
                    "baseline_throughput": 280.0,
                    "hook_throughput": 300.0,
                    "relative_to_none": 1.0714,
                }
            ],
            "bucket_comparisons": [
                {
                    "bucket_cap_mb": 100,
                    "throughput": 310.0,
                    "relative_to_default": 1.107,
                    "dataloader_wait_ms": 4.2,
                }
            ],
            "topology_comparisons": [
                {
                    "visible_devices": "1,2,3,4",
                    "bucket_cap_mb": 100,
                    "throughput": 320.0,
                    "dataloader_wait_ms": 4.1,
                }
            ],
            "best_dataloader": {
                "run_id": "data_4gpu_bf16_none_b8_w4_p4_persist",
                "throughput": 300.0,
                "dataloader_wait_ms": 4.0,
            },
        },
    )

    markdown = build_report_markdown(load_summaries(tmp_path), report_root=tmp_path)

    assert "# ParaScale Benchmark Report" in markdown
    assert "Dual RTX 4090 Validation" in markdown
    assert "parascale_native_ddp" in markdown
    assert "Evidence Quality" in markdown
    assert "local_native_clip_datacomp" in markdown
    assert "10/100" in markdown
    assert "Ascend Parallel Matrix" in markdown
    assert "rtx4090_clip_precision_datacomp/summary.json" in markdown
    assert "A6000 Native-DDP Scaling" in markdown
    assert "bf16_compress" in markdown
    assert "CUDA_VISIBLE_DEVICES" in markdown
    assert "1,2,3,4" in markdown
    assert "a6000_native_ddp_scaling/summary.json" in markdown
