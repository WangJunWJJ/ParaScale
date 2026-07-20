# -*- coding: utf-8 -*-
# @Time : 2026/7/20 下午1:45
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import json
from pathlib import Path

from tests.benchmarks.tools.summarize_ascend_validation import build_report


def _write_payload(path: Path, *, backend: str, throughput: float) -> None:
    payload = {
        "mode": "train",
        "runtime_status": "real_local",
        "metrics": {
            "samples_per_second": throughput,
            "loss": 0.25,
        },
        "train": {
            "backend": backend,
            "global_step": 5,
            "last_metrics": {
                "loss": 0.25,
                "samples_per_second": throughput,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_ascend_validation_summary_requires_tiny_runs(tmp_path):
    doctor = {
        "mode": "doctor",
        "status": "ok",
        "dependencies": {"torch_npu": True},
        "ascend_runtime": {"available": True, "device_count": 8},
        "diagnostics": {"ok": True},
    }
    (tmp_path / "doctor.json").write_text(json.dumps(doctor), encoding="utf-8")
    _write_payload(tmp_path / "tiny_single.json", backend="ascend_native", throughput=12.5)
    _write_payload(tmp_path / "tiny_hccl.json", backend="native_ddp", throughput=18.0)

    report = build_report(
        tmp_path,
        suite_id="unit_ascend",
        hardware="Ascend 910B4",
        image="ascend-image",
    )

    assert report["passed"] is True
    assert report["missing"] == []
    runs = {run["run_id"]: run for run in report["runs"]}
    assert runs["doctor"]["torch_npu"] is True
    assert runs["tiny_hccl"]["throughput"] == 18.0


def test_ascend_validation_summary_can_mark_access_blocked(tmp_path):
    report = build_report(
        tmp_path,
        suite_id="unit_ascend",
        hardware="Ascend 910B4",
        image="ascend-image",
        blocked_reason="ssh password required",
    )

    assert report["passed"] is False
    assert report["blocked"] is True
    assert report["blocked_reason"] == "ssh password required"


def test_ascend_validation_summary_reads_top_level_train_payload(tmp_path):
    payload = {
        "mode": "train",
        "backend": "ascend_native",
        "global_step": 5,
        "steps_per_second": 2.5,
        "last_metrics": {"loss": 0.125},
    }
    (tmp_path / "tiny_single.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    doctor = {
        "mode": "doctor",
        "dependencies": {"torch_npu": True},
        "ascend_runtime": {"available": True, "device_count": 2},
        "diagnostics": {"ok": True},
    }
    (tmp_path / "doctor.json").write_text(json.dumps(doctor), encoding="utf-8")
    _write_payload(tmp_path / "tiny_hccl.json", backend="native_ddp", throughput=18.0)

    report = build_report(
        tmp_path,
        suite_id="unit_ascend",
        hardware="Ascend 910B4",
        image="ascend-image",
    )

    runs = {run["run_id"]: run for run in report["runs"]}
    assert runs["tiny_single"]["throughput"] == 2.5
    assert runs["tiny_single"]["loss"] == 0.125
