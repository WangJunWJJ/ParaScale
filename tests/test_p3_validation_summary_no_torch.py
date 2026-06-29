# -*- coding: utf-8 -*-
# @Time : 2026/6/15 下午3:46
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import json
import tempfile
from pathlib import Path

from tests.benchmarks.tools.summarize_p3_validation import build_report


def _workspace_tmp(name):
    path = Path(tempfile.gettempdir()) / "parascale-test-runs" / name
    path.mkdir(parents=True, exist_ok=True)
    for child in path.glob("*.json"):
        child.unlink()
    return path


def _write_payload(path, *, backend="native", step=1, resumed=False):
    payload = {
        "mode": "train",
        "backend": backend,
        "global_step": step,
        "checkpoint_validation": {"ok": True},
        "last_metrics": {
            "peak_memory_bytes": 1024,
            "end_to_end_image_text_pairs_per_second": 3.0,
        },
    }
    if resumed:
        payload["resumed_from"] = {"global_step": 2}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_p3_summary_marks_required_checks_passed():
    tmp_path = _workspace_tmp("p3_summary_passed")
    _write_payload(tmp_path / "native_bf16_train.json", step=4)
    _write_payload(tmp_path / "native_bf16_resume.json", step=6, resumed=True)
    _write_payload(tmp_path / "deepspeed_zero3_train.json", backend="deepspeed", step=4)
    _write_payload(
        tmp_path / "deepspeed_zero3_resume.json",
        backend="deepspeed",
        step=6,
        resumed=True,
    )
    _write_payload(
        tmp_path / "deepspeed_zero3_activation_ckpt_train.json",
        backend="deepspeed",
        step=2,
    )
    _write_payload(tmp_path / "hf_clip_pretrained_offline_smoke.json", step=1)

    report = build_report(tmp_path)

    assert report["passed"] is True
    assert len(report["checks"]) == 6
    assert all(item["ok"] for item in report["checks"])


def test_p3_summary_allows_hf_smoke_skip():
    tmp_path = _workspace_tmp("p3_summary_hf_skip")
    _write_payload(tmp_path / "native_bf16_train.json", step=4)
    _write_payload(tmp_path / "native_bf16_resume.json", step=6, resumed=True)
    _write_payload(tmp_path / "deepspeed_zero3_train.json", backend="deepspeed", step=4)
    _write_payload(
        tmp_path / "deepspeed_zero3_resume.json",
        backend="deepspeed",
        step=6,
        resumed=True,
    )
    _write_payload(
        tmp_path / "deepspeed_zero3_activation_ckpt_train.json",
        backend="deepspeed",
        step=2,
    )
    (tmp_path / "hf_clip_pretrained_offline_smoke.json").write_text(
        json.dumps({"status": "skipped", "reason": "missing model"}),
        encoding="utf-8",
    )

    report = build_report(tmp_path)

    assert report["passed"] is True
    assert report["checks"][-1]["skipped"] is True
