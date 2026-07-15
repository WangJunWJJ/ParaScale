# -*- coding: utf-8 -*-

import json
from pathlib import Path

from tests.benchmarks.tools.summarize_direct_pytorch_comparison import build_report


def _write_payload(path: Path, *, backend: str, throughput: float, loss: float) -> None:
    payload = {
        "mode": "benchmark",
        "metrics": {
            "stable_end_to_end_image_text_pairs_per_second": throughput,
            "peak_memory_bytes": 3 * 1024**3,
            "dataloader_wait_ms": 1.5,
            "loss": loss,
        },
        "train": {
            "backend": backend,
            "global_step": 80,
            "last_metrics": {
                "loss": loss,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_direct_pytorch_comparison_reports_ratios(tmp_path):
    _write_payload(
        tmp_path / "parascale_native_ddp.json",
        backend="native_ddp",
        throughput=120.0,
        loss=2.0,
    )
    _write_payload(
        tmp_path / "parascale_fsdp.json",
        backend="fsdp",
        throughput=90.0,
        loss=2.1,
    )
    _write_payload(
        tmp_path / "torch_ddp.json",
        backend="torch_ddp",
        throughput=100.0,
        loss=2.2,
    )
    _write_payload(
        tmp_path / "torch_fsdp.json",
        backend="torch_fsdp",
        throughput=80.0,
        loss=2.3,
    )
    _write_payload(
        tmp_path / "parascale_deepspeed.json",
        backend="deepspeed",
        throughput=95.0,
        loss=2.4,
    )
    _write_payload(
        tmp_path / "deepspeed.json",
        backend="direct_deepspeed",
        throughput=76.0,
        loss=2.5,
    )

    report = build_report(
        tmp_path,
        suite_id="unit_direct_compare",
        hardware="dual RTX 4090D",
        image="parascale-ci:cu121-torch24",
    )

    assert report["passed"] is True
    assert report["missing"] == []
    ratios = {
        item["parascale"]: item["parascale_vs_direct"]
        for item in report["comparisons"]
    }
    assert ratios["parascale_native_ddp"] == 1.2
    assert ratios["parascale_fsdp"] == 1.125
    assert ratios["parascale_deepspeed"] == 1.25
    deepspeed_ratios = {
        item["baseline"]: item["deepspeed_vs_baseline"]
        for item in report["deepspeed_comparisons"]
    }
    assert deepspeed_ratios["parascale_native_ddp"] == 95.0 / 120.0
    assert deepspeed_ratios["torch_ddp"] == 0.95
