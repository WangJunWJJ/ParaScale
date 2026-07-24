# -*- coding: utf-8 -*-
# @Time : 2026/6/15 下午5:13
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import json
import tempfile
from pathlib import Path

from parascale.reporting.matrix import build_report, recommend_backends
from tests.benchmarks.tools.summarize_p3_validation import (
    build_report as build_checkpoint_stress_report,
)


def _workspace_tmp(name):
    path = Path(tempfile.gettempdir()) / "parascale-test-runs" / name
    path.mkdir(parents=True, exist_ok=True)
    for child in path.glob("*.json"):
        child.unlink()
    return path


def _write_benchmark(path, *, backend, throughput, memory_gb):
    payload = {
        "mode": "benchmark",
        "config": {
            "parascale": {
                "precision": "bf16",
                "task_type": "multimodal",
                "model_family": "clip",
                "gradient_accumulation_steps": 2,
            }
        },
        "metrics": {
            "stable_end_to_end_images_per_second": throughput,
            "stable_step_time_ms": 1000.0 / max(throughput, 1e-9),
            "peak_memory_bytes": memory_gb * 1024**3,
            "dataloader_wait_ms": 8.0,
        },
        "train": {
            "backend": backend,
            "config_artifacts": {
                "run_dir": str(path.parent / path.stem),
                "resolved_config": str(path.parent / path.stem / "config.resolved.json"),
                "deepspeed_final_config": None,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_checkpoint_payload(path, *, backend="native", step=1, resumed=False):
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


def test_checkpoint_stress_summary_marks_required_checks_passed():
    tmp_path = _workspace_tmp("checkpoint_stress_summary_passed")
    _write_checkpoint_payload(tmp_path / "native_bf16_train.json", step=4)
    _write_checkpoint_payload(
        tmp_path / "native_bf16_resume.json", step=6, resumed=True
    )
    _write_checkpoint_payload(
        tmp_path / "deepspeed_zero3_train.json", backend="deepspeed", step=4
    )
    _write_checkpoint_payload(
        tmp_path / "deepspeed_zero3_resume.json",
        backend="deepspeed",
        step=6,
        resumed=True,
    )
    _write_checkpoint_payload(
        tmp_path / "deepspeed_zero3_activation_ckpt_train.json",
        backend="deepspeed",
        step=2,
    )
    _write_checkpoint_payload(tmp_path / "hf_clip_pretrained_offline_smoke.json")

    report = build_checkpoint_stress_report(tmp_path)

    assert report["passed"] is True
    assert len(report["checks"]) == 6
    assert all(item["ok"] for item in report["checks"])


def test_checkpoint_stress_summary_allows_hf_smoke_skip():
    tmp_path = _workspace_tmp("checkpoint_stress_summary_hf_skip")
    _write_checkpoint_payload(tmp_path / "native_bf16_train.json", step=4)
    _write_checkpoint_payload(
        tmp_path / "native_bf16_resume.json", step=6, resumed=True
    )
    _write_checkpoint_payload(
        tmp_path / "deepspeed_zero3_train.json", backend="deepspeed", step=4
    )
    _write_checkpoint_payload(
        tmp_path / "deepspeed_zero3_resume.json",
        backend="deepspeed",
        step=6,
        resumed=True,
    )
    _write_checkpoint_payload(
        tmp_path / "deepspeed_zero3_activation_ckpt_train.json",
        backend="deepspeed",
        step=2,
    )
    (tmp_path / "hf_clip_pretrained_offline_smoke.json").write_text(
        json.dumps({"status": "skipped", "reason": "missing model"}),
        encoding="utf-8",
    )

    report = build_checkpoint_stress_report(tmp_path)

    assert report["passed"] is True
    assert report["checks"][-1]["skipped"] is True


def test_balanced_recommendation_prefers_lower_memory_when_throughput_is_close():
    rows = [
        {
            "run_id": "yolo_l",
            "backend": "native_ddp",
            "status": "ok",
            "throughput": 50.0,
            "peak_memory_gb": 2.0,
        },
        {
            "run_id": "yolo_l",
            "backend": "fsdp",
            "status": "ok",
            "throughput": 49.0,
            "peak_memory_gb": 1.2,
        },
        {
            "run_id": "yolo_l",
            "backend": "deepspeed",
            "status": "ok",
            "throughput": 40.0,
            "peak_memory_gb": 3.0,
        },
    ]

    recommendations = recommend_backends(rows, optimize_for="balanced")

    assert recommendations[0]["selected_backend"] == "fsdp"
    assert (
        recommendations[0]["recommended_config_updates"]["training_backend"] == "fsdp"
    )
    candidates = {
        item["backend"]: item
        for item in recommendations[0]["candidate_evaluations"]
    }
    assert candidates["fsdp"]["selected"] is True
    assert candidates["native_ddp"]["selected"] is False
    assert "higher_memory_than_selected" in candidates["native_ddp"][
        "rejection_reasons"
    ]
    assert recommendations[0]["expected_trade_off"]


def test_balanced_recommendation_prefers_faster_backend_when_gap_is_large():
    rows = [
        {
            "run_id": "vlm_lora",
            "backend": "native_ddp",
            "status": "ok",
            "throughput": 165.0,
            "peak_memory_gb": 1.1,
        },
        {
            "run_id": "vlm_lora",
            "backend": "fsdp",
            "status": "ok",
            "throughput": 23.0,
            "peak_memory_gb": 0.9,
        },
        {
            "run_id": "vlm_lora",
            "backend": "deepspeed",
            "status": "ok",
            "throughput": 136.0,
            "peak_memory_gb": 2.4,
        },
    ]

    recommendations = recommend_backends(rows, optimize_for="balanced")

    assert recommendations[0]["selected_backend"] == "native_ddp"


def test_recommendation_exposes_failed_candidates_and_low_confidence():
    rows = [
        {
            "run_id": "clip",
            "backend": "native_ddp",
            "status": "ok",
            "throughput": 100.0,
            "peak_memory_gb": 2.0,
        },
        {
            "run_id": "clip",
            "backend": "fsdp",
            "status": "error",
            "error": "benchmark failed",
        },
    ]

    recommendation = recommend_backends(rows)[0]
    candidates = {
        item["backend"]: item
        for item in recommendation["candidate_evaluations"]
    }

    assert recommendation["confidence"] == "low"
    assert recommendation["actionable"] is False
    assert recommendation["evidence"]["valid_candidate_count"] == 1
    assert candidates["fsdp"]["rejection_reasons"] == ["benchmark_failed"]


def test_backend_matrix_report_includes_recommendations():
    tmp_path = _workspace_tmp("backend_matrix_recommendations")
    _write_benchmark(
        tmp_path / "model_native_ddp.json",
        backend="native_ddp",
        throughput=100.0,
        memory_gb=2.0,
    )
    _write_benchmark(
        tmp_path / "model_fsdp.json",
        backend="fsdp",
        throughput=98.0,
        memory_gb=1.0,
    )
    _write_benchmark(
        tmp_path / "model_deepspeed.json",
        backend="deepspeed",
        throughput=80.0,
        memory_gb=3.0,
    )

    report = build_report(
        tmp_path,
        title="test",
        workload_label="synthetic matrix",
        optimize_for="balanced",
    )

    assert report["recommendations"][0]["selected_backend"] == "fsdp"
    assert report["comparisons"]
    assert report["results"][0]["config_artifacts"]["resolved_config"].endswith(
        "config.resolved.json"
    )
    assert report["recommendations"][0]["communication_plan"]["backend"] == "fsdp"
    assert report["evidence_summary"]["recommendation_count"] == 1
    assert report["evidence_summary"]["selected_backends"] == ["fsdp"]
    assert (
        report["recommendations"][0]["communication_plan"]["evidence"][
            "dataloader_wait_ms"
        ]
        == 8.0
    )


def test_backend_matrix_markdown_reports_communication_plan():
    tmp_path = _workspace_tmp("backend_matrix_communication_markdown")
    markdown_path = tmp_path / "report.md"
    _write_benchmark(
        tmp_path / "model_native_ddp.json",
        backend="native_ddp",
        throughput=100.0,
        memory_gb=2.0,
    )

    report = build_report(
        tmp_path,
        title="test",
        workload_label="synthetic matrix",
        optimize_for="throughput",
    )
    from parascale.reporting.matrix import write_markdown

    write_markdown(report, markdown_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert "Communication Plan" in markdown
    assert "native_ddp" in markdown
    assert "Candidate Evaluation" in markdown
    assert "Expected trade-off" in markdown


def test_backend_matrix_report_marks_oom_retry_recovered():
    tmp_path = _workspace_tmp("backend_matrix_oom_recovery")
    _write_benchmark(
        tmp_path / "model_b4_native_ddp.json",
        backend="native_ddp",
        throughput=6.0,
        memory_gb=14.0,
    )
    (tmp_path / "model_b4_fsdp.error.json").write_text(
        json.dumps(
            {
                "backend": "fsdp",
                "error": "benchmark failed with OOM",
                "returncode": 1,
            }
        ),
        encoding="utf-8",
    )
    failed_run_dir = tmp_path / "model_b4_fsdp"
    failed_run_dir.mkdir(exist_ok=True)
    (failed_run_dir / "config.resolved.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model_b4_oom_retry1_fsdp.error.json").write_text(
        json.dumps(
            {
                "backend": "fsdp",
                "error": "checkpoint shape mismatch",
                "returncode": 1,
                "attempt": 1,
                "retry_trigger": "oom",
                "retry_terminated": True,
                "retry_termination_reason": "non_oom_failure",
                "config_artifacts": {
                    "resolved_config": "/runs/retry/config.resolved.json"
                },
            }
        ),
        encoding="utf-8",
    )
    _write_benchmark(
        tmp_path / "model_b4_oom_retry2_deepspeed_zero2.json",
        backend="deepspeed_zero2",
        throughput=7.0,
        memory_gb=5.0,
    )

    report = build_report(
        tmp_path,
        title="test",
        workload_label="synthetic matrix",
        optimize_for="balanced",
    )

    assert report["oom_recovery"][0]["run_id"] == "model_b4"
    assert report["oom_recovery"][0]["recovered"] is True
    assert report["oom_recovery"][0]["selected_backend"] == "deepspeed_zero2"
    first_attempt = report["oom_recovery"][0]["attempts"][0]
    assert first_attempt["attempt"] == 1
    assert first_attempt["retry_trigger"] == "oom"
    assert first_attempt["retry_terminated"] is True
    assert first_attempt["retry_termination_reason"] == "non_oom_failure"
    assert first_attempt["config_artifacts"]["resolved_config"] == (
        "/runs/retry/config.resolved.json"
    )
    failed_row = next(
        row
        for row in report["results"]
        if row["run_id"] == "model_b4" and row["backend"] == "fsdp"
    )
    assert failed_row["config_artifacts"]["resolved_config"].endswith(
        "config.resolved.json"
    )
