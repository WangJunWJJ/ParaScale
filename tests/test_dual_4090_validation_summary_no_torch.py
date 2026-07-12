# -*- coding: utf-8 -*-
# @Time : 2026/7/12 下午1:20
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import json
from pathlib import Path

from tests.benchmarks.tools.summarize_dual_4090_validation import build_report


def _write_payload(path: Path, *, backend: str, loss: float, throughput: float) -> None:
    payload = {
        "mode": "benchmark",
        "runtime_status": "real_local",
        "capability_level": "cuda_real_data",
        "metrics": {
            "end_to_end_image_text_pairs_per_second": throughput,
            "peak_memory_bytes": 2 * 1024**3,
            "dataloader_wait_ms": 7.5,
            "loss": loss,
        },
        "train": {
            "backend": backend,
            "global_step": 8,
            "last_metrics": {
                "loss": loss,
                "end_to_end_image_text_pairs_per_second": throughput,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_dual_4090_report_extracts_training_effect_and_speed(tmp_path):
    _write_payload(
        tmp_path / "clip_native_ddp.json",
        backend="native_ddp",
        loss=1.23,
        throughput=155.0,
    )
    _write_payload(
        tmp_path / "clip_fsdp.json",
        backend="fsdp",
        loss=1.31,
        throughput=101.0,
    )

    report = build_report(
        tmp_path,
        suite_id="unit_dual_4090",
        hardware="dual RTX 4090D",
        image="parascale-ci:cu121-torch24",
    )

    assert report["passed"] is True
    assert report["hardware"] == "dual RTX 4090D"
    assert report["summaries"][0]["model"] == "clip"
    assert report["summaries"][0]["best_backend"] == "native_ddp"
    assert report["summaries"][0]["best_throughput"] == 155.0
    runs_by_backend = {run["backend"]: run for run in report["runs"]}
    assert runs_by_backend["native_ddp"]["loss"] == 1.23
