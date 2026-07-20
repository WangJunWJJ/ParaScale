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

    markdown = build_report_markdown(load_summaries(tmp_path), report_root=tmp_path)

    assert "# ParaScale Benchmark Report" in markdown
    assert "Dual RTX 4090 Validation" in markdown
    assert "parascale_native_ddp" in markdown
    assert "Ascend Parallel Matrix" in markdown
    assert "rtx4090_clip_precision_datacomp/summary.json" in markdown
