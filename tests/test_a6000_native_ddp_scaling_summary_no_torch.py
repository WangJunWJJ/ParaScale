# -*- coding: utf-8 -*-
# @Time : 2026/7/24 下午4:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import json
from pathlib import Path

from tests.benchmarks.tools.summarize_a6000_native_ddp_scaling import build_report


def _write_result(path: Path, throughput: float, wait_ms: float) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "train": {
                    "backend": "native_ddp",
                    "global_step": 120,
                },
                "metrics": {
                    "stable_end_to_end_image_text_pairs_per_second": throughput,
                    "stable_dataloader_wait_ms": wait_ms,
                    "stable_peak_memory_bytes": 2 * 1024**3,
                    "stable_loss": 2.0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_a6000_native_ddp_scaling_summary_parses_matrix(tmp_path):
    _write_result(tmp_path / "scale_1gpu_bf16_none_b8_w2.json", 100.0, 5.0)
    _write_result(tmp_path / "scale_2gpu_bf16_none_b8_w2.json", 160.0, 5.5)
    _write_result(tmp_path / "scale_4gpu_bf16_none_b8_w2.json", 280.0, 6.0)
    _write_result(tmp_path / "hook_4gpu_bf16_bf16_compress_b8_w2.json", 300.0, 6.2)
    _write_result(
        tmp_path / "bucket_4gpu_bf16_bf16_compress_bucket100_b8_w2.json",
        330.0,
        6.1,
    )
    _write_result(
        tmp_path / "topo_4gpu_bf16_bf16_compress_bucket100_cuda1234_b8_w2.json",
        340.0,
        5.9,
    )
    _write_result(tmp_path / "data_4gpu_bf16_none_b8_w4_p4_persist.json", 310.0, 3.0)

    report = build_report(
        tmp_path,
        hardware="5x RTX A6000",
        image="parascale-ci:a6000-cu126-torch25",
        dataset="/dataset/datacomp_subsets/final/datacomp_10k_wds",
        model="clip_medium",
        steps=120,
        warmup_steps=20,
        batch_per_gpu=8,
    )

    bf16 = next(item for item in report["scaling"] if item["precision"] == "bf16")
    hook = next(
        item
        for item in report["hook_comparisons"]
        if item["gpus"] == 4 and item["precision"] == "bf16"
    )

    assert report["passed"] is True
    assert bf16["scale_1_to_4"] == 2.8
    assert bf16["efficiency_1_to_4"] == 0.7
    assert hook["relative_to_none"] > 1.07
    assert report["bucket_comparisons"][0]["bucket_cap_mb"] == 100
    assert report["bucket_comparisons"][0]["relative_to_default"] == 1.1
    assert report["topology_comparisons"][0]["visible_devices"] == "1,2,3,4"
    assert report["best_dataloader"]["run_id"] == "data_4gpu_bf16_none_b8_w4_p4_persist"
