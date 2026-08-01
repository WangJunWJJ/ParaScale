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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
