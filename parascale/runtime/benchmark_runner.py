# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午9:04
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Benchmark execution runner."""

from __future__ import annotations

import time
from typing import Any, Dict

from parascale.reporting.aggregation import aggregate_stable_metrics
from parascale.reporting.benchmark import benchmark_result_from_train_payload
from parascale.runtime.evidence import attach_runtime_evidence
from parascale.runtime.runner_common import _section
from parascale.runtime.train_runner import run_train_from_config


def run_benchmark_from_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    training = dict(_section(config_data, "training"))
    training.setdefault("max_steps", int(training.get("benchmark_steps", 3) or 3))
    benchmark_config = dict(config_data)
    benchmark_config["training"] = training
    start = time.perf_counter()
    train_payload = run_train_from_config(benchmark_config, resume_step=None)
    elapsed = max(time.perf_counter() - start, 1e-9)
    validation = _build_benchmark_validation_payload(
        benchmark_config, train_payload, training
    )
    last_metrics = dict(train_payload.get("last_metrics", {}))
    warmup_steps = int(training.get("warmup_steps", 0) or 0)
    stable_metrics = aggregate_stable_metrics(
        train_payload.get("metrics_history", []), warmup_steps=warmup_steps
    )
    benchmark_result = benchmark_result_from_train_payload(train_payload)
    task_type = _section(config_data, "parascale").get("task_type", "generic")
    workload = str(training.get("workload", "synthetic_regression"))
    return attach_runtime_evidence({
        "mode": "benchmark",
        "dry_run": False,
        "runtime_status": train_payload.get("runtime_status", "real_local"),
        "capability_level": train_payload.get("capability_level", "local_benchmark"),
        "synthetic": bool(train_payload.get("synthetic", False)),
        "benchmark_type": f"{workload}_train",
        "elapsed_seconds": elapsed,
        "metrics": {
            "steps_per_second": train_payload["steps_per_second"],
            "step_time_ms": 1000.0 / max(train_payload["steps_per_second"], 1e-9),
            "warmup_steps": warmup_steps,
            "measured_steps": int(stable_metrics.get("measured_steps", 0)),
            "samples_per_second": benchmark_result.metric("samples_per_second"),
            "tokens_per_second": float(
                last_metrics.get("tokens_per_second", 0.0) or 0.0
            ),
            "end_to_end_tokens_per_second": float(
                last_metrics.get("end_to_end_tokens_per_second", 0.0) or 0.0
            ),
            "images_per_second": float(
                last_metrics.get("images_per_second", 0.0) or 0.0
            ),
            "end_to_end_images_per_second": float(
                last_metrics.get("end_to_end_images_per_second", 0.0) or 0.0
            ),
            "patch_tokens_per_second": float(
                last_metrics.get("patch_tokens_per_second", 0.0) or 0.0
            ),
            "end_to_end_patch_tokens_per_second": float(
                last_metrics.get("end_to_end_patch_tokens_per_second", 0.0) or 0.0
            ),
            "image_text_pairs_per_second": float(
                last_metrics.get("image_text_pairs_per_second", 0.0) or 0.0
            ),
            "end_to_end_image_text_pairs_per_second": float(
                last_metrics.get("end_to_end_image_text_pairs_per_second", 0.0) or 0.0
            ),
            "padding_ratio": float(last_metrics.get("padding_ratio", 0.0) or 0.0),
            "peak_memory_bytes": float(
                last_metrics.get("peak_memory_bytes", 0.0) or 0.0
            ),
            "allocated_memory_bytes": float(
                last_metrics.get("allocated_memory_bytes", 0.0) or 0.0
            ),
            "dataloader_wait_ms": float(
                last_metrics.get("dataloader_wait_ms", 0.0) or 0.0
            ),
            "adapter_params": int(last_metrics.get("adapter_params", 0) or 0),
            "trainable_params": int(last_metrics.get("trainable_params", 0) or 0),
            "total_params": int(last_metrics.get("total_params", 0) or 0),
            "trainable_ratio": float(last_metrics.get("trainable_ratio", 0.0) or 0.0),
            "lora_rank": int(last_metrics.get("lora_rank", 0) or 0),
            **stable_metrics,
        },
        "validation": validation,
        "benchmark_result": benchmark_result.to_dict(),
        "comparison_contract": {
            "primary_metrics": [
                "samples_per_second",
                "images_per_second",
                "patch_tokens_per_second",
                "image_text_pairs_per_second",
                "step_time_ms",
            ],
            "baseline_backends": ["fsdp", "deepspeed"],
            "requires_same_workload": True,
            "requires_same_hardware": True,
            "task_type": task_type,
            "workload": workload,
        },
        "config_artifacts": train_payload.get("config_artifacts", {}),
        "train": train_payload,
    })

def _build_benchmark_validation_payload(
    config_data: Dict[str, Any],
    train_payload: Dict[str, Any],
    training: Dict[str, Any],
) -> Dict[str, Any]:
    checkpoint_validation = train_payload.get("checkpoint_validation", {})
    validation: Dict[str, Any] = {
        "checkpoint": {
            "ok": bool(checkpoint_validation.get("ok", False)),
            "skipped": bool(checkpoint_validation.get("skipped", False)),
            "details": checkpoint_validation,
        },
        "resume": {
            "ok": None,
            "skipped": True,
            "reason": "set training.validate_resume=true to run resume validation",
        },
    }
    if not bool(training.get("validate_resume", False)):
        return validation

    checkpoint = train_payload.get("checkpoint")
    global_step = int(train_payload.get("global_step", 0) or 0)
    if not checkpoint or checkpoint_validation.get("skipped"):
        validation["resume"] = {
            "ok": False,
            "skipped": False,
            "reason": "resume validation requires a saved checkpoint",
        }
        return validation

    resume_steps = int(training.get("resume_validation_steps", 1) or 1)
    resume_config = dict(config_data)
    resume_training = dict(training)
    resume_training["max_steps"] = max(1, resume_steps)
    resume_training["benchmark_steps"] = max(1, resume_steps)
    resume_training["skip_final_checkpoint"] = bool(
        training.get("skip_resume_final_checkpoint", True)
    )
    resume_config["training"] = resume_training

    resume_start = time.perf_counter()
    try:
        resumed = run_train_from_config(resume_config, resume_step=global_step)
        validation["resume"] = {
            "ok": int(resumed.get("global_step", 0) or 0) >= global_step + resume_steps,
            "skipped": False,
            "elapsed_seconds": time.perf_counter() - resume_start,
            "resumed_from_step": global_step,
            "global_step": resumed.get("global_step"),
            "backend_state_loaded": bool(
                resumed.get("resumed_from", {})
                .get("metadata", {})
                .get("backend_state_loaded", False)
            ),
        }
    except Exception as exc:
        validation["resume"] = {
            "ok": False,
            "skipped": False,
            "elapsed_seconds": time.perf_counter() - resume_start,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return validation
