# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午7:24
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Stability benchmark result collection and Markdown reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from parascale.reporting.aggregation import aggregate_stable_metrics


def collect_stability_results(
    output_dir: Path, *, warmup_steps: int = 0
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        if (
            path.name.endswith(".config.json")
            or path.name.endswith(".error.json")
            or path.name in {"payload.json", "summary.json"}
        ):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        last = dict(payload.get("last_metrics", {}))
        history = payload.get("metrics_history", [])
        stable = aggregate_stable_metrics(history, warmup_steps=warmup_steps)
        row = build_stability_row(path, payload, last, stable)
        rows.append(row)
    for path in sorted(output_dir.glob("*.error.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        rows.append(
            {
                "run_id": path.name[: -len(".error.json")],
                "status": "error",
                "backend": payload.get("backend"),
                "error": payload.get("error", "benchmark failed"),
                "log": payload.get("log"),
            }
        )
    total = len(rows)
    failures = len([row for row in rows if row.get("status") != "ok"])
    for row in rows:
        row["failure_rate"] = failures / total if total else 0.0
    return rows


def build_stability_row(
    path: Path,
    payload: Dict[str, Any],
    last: Dict[str, Any],
    stable: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "run_id": path.stem,
        "status": "ok",
        "backend": payload.get("backend"),
        "global_step": payload.get("global_step"),
        "resumed": bool(payload.get("resumed_from")),
        "checkpoint_ok": bool(payload.get("checkpoint_validation", {}).get("ok")),
        "throughput": float(
            last.get("end_to_end_image_text_pairs_per_second")
            or last.get("end_to_end_images_per_second")
            or last.get("samples_per_second")
            or 0.0
        ),
        "peak_memory_gb": float(last.get("peak_memory_bytes", 0.0) or 0.0) / 1024**3,
        "dataloader_wait_ms": float(last.get("dataloader_wait_ms", 0.0) or 0.0),
        "stable_step_time_ms": stable.get("stable_step_time_ms", 0.0),
        "stable_max_step_time_ms": stable.get("stable_max_step_time_seconds", 0.0)
        * 1000.0,
        "stable_min_step_time_ms": stable.get("stable_min_step_time_seconds", 0.0)
        * 1000.0,
        "stable_dataloader_wait_ms": stable.get("stable_dataloader_wait_ms", 0.0),
        "measured_steps": stable.get("measured_steps", 0.0),
        "stable_loss": stable.get("stable_loss", 0.0),
    }
    row["stable_throughput"] = float(
        stable.get("stable_end_to_end_image_text_pairs_per_second")
        or stable.get("stable_end_to_end_images_per_second")
        or stable.get("stable_end_to_end_tokens_per_second")
        or row["throughput"]
        or 0.0
    )
    for key in pipeline_profile_metric_names(include_counts=True):
        row[key] = stable.get(key, 0.0)
    step = float(row["stable_step_time_ms"] or 0.0)
    if step > 0:
        row["step_time_jitter_ratio"] = (
            float(row["stable_max_step_time_ms"])
            - float(row["stable_min_step_time_ms"])
        ) / step
    else:
        row["step_time_jitter_ratio"] = 0.0
    return row


def build_resume_continuity(
    rows: list[Dict[str, Any]],
    *,
    max_loss_ratio: float = 1.5,
    min_throughput_ratio: float = 0.8,
) -> list[Dict[str, Any]]:
    """Compare each successful training phase with its resume phase."""
    by_run_id = {str(row.get("run_id")): row for row in rows}
    continuity: list[Dict[str, Any]] = []
    for resume_run_id, resume in sorted(by_run_id.items()):
        if not resume_run_id.endswith("_resume"):
            continue
        run_id = resume_run_id[: -len("_resume")]
        initial = by_run_id.get(run_id)
        if initial is None:
            continue
        initial_loss = float(initial.get("stable_loss", 0.0) or 0.0)
        resume_loss = float(resume.get("stable_loss", 0.0) or 0.0)
        initial_throughput = float(initial.get("stable_throughput", 0.0) or 0.0)
        resume_throughput = float(resume.get("stable_throughput", 0.0) or 0.0)
        checkpoint_ok = bool(initial.get("checkpoint_ok")) and bool(
            resume.get("checkpoint_ok")
        )
        progressed = int(resume.get("global_step", 0) or 0) > int(
            initial.get("global_step", 0) or 0
        )
        loss_ratio = resume_loss / initial_loss if initial_loss > 0.0 else None
        throughput_ratio = (
            resume_throughput / initial_throughput
            if initial_throughput > 0.0
            else None
        )
        reasons = []
        if initial.get("status") != "ok" or resume.get("status") != "ok":
            reasons.append("phase_failed")
        if not bool(resume.get("resumed")):
            reasons.append("resume_not_confirmed")
        if not checkpoint_ok:
            reasons.append("checkpoint_invalid")
        if not progressed:
            reasons.append("step_not_progressed")
        if loss_ratio is None or loss_ratio > max_loss_ratio:
            reasons.append("loss_jump")
        if throughput_ratio is None or throughput_ratio < min_throughput_ratio:
            reasons.append("throughput_drop")
        continuity.append(
            {
                "run_id": run_id,
                "resume_run_id": resume_run_id,
                "status": "error" if reasons else "ok",
                "initial_step": initial.get("global_step"),
                "final_step": resume.get("global_step"),
                "loss_ratio": loss_ratio,
                "throughput_ratio": throughput_ratio,
                "checkpoint_ok": checkpoint_ok,
                "thresholds": {
                    "max_loss_ratio": max_loss_ratio,
                    "min_throughput_ratio": min_throughput_ratio,
                },
                "reasons": reasons,
            }
        )
    return continuity


def build_restart_validation(
    results: list[Dict[str, Any]],
    stability_rows: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Validate intentional process-tree kills followed by fresh-launcher resume."""
    results_by_id = {str(item.get("run_id")): item for item in results}
    rows_by_id = {str(item.get("run_id")): item for item in stability_rows}
    validations: list[Dict[str, Any]] = []
    for run_id, interrupted in sorted(results_by_id.items()):
        if interrupted.get("status") != "interrupted":
            continue
        resume_run_id = f"{run_id}_resume"
        resume_result = results_by_id.get(resume_run_id, {})
        resume_row = rows_by_id.get(resume_run_id, {})
        checkpoint_step = int(interrupted.get("checkpoint_step", 0) or 0)
        final_step = int(resume_row.get("global_step", 0) or 0)
        intentional_kill = bool(interrupted.get("intentional_kill"))
        checkpoint_ok = bool(interrupted.get("checkpoint_ok")) and bool(
            resume_row.get("checkpoint_ok")
        )
        resumed = bool(resume_row.get("resumed"))
        reasons = []
        if not intentional_kill:
            reasons.append("kill_not_confirmed")
        if not checkpoint_ok:
            reasons.append("checkpoint_invalid")
        if resume_result.get("status") != "ok" or not resumed:
            reasons.append("resume_failed")
        if final_step <= checkpoint_step:
            reasons.append("step_not_progressed")
        validations.append(
            {
                "run_id": run_id,
                "resume_run_id": resume_run_id,
                "status": "error" if reasons else "ok",
                "checkpoint_step": checkpoint_step,
                "final_step": final_step,
                "intentional_kill": intentional_kill,
                "checkpoint_ok": checkpoint_ok,
                "resumed": resumed,
                "reasons": reasons,
            }
        )
    return validations


def write_stability_markdown(payload: Dict[str, Any], path: Path) -> None:
    lines = [
        "# ParaScale P3 Stability Report",
        "",
        f"- Scenario: {payload.get('scenario')}",
        f"- Output dir: {payload.get('output_dir')}",
        "",
        "## Results",
        "",
        (
            "| Run | Status | Backend | Step | Loss | Throughput | Peak GB | Wait ms "
            "| Jitter | Resumed |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload.get("stability", []):
        lines.append(format_stability_result_row(row))
    pipeline_rows = [
        row
        for row in payload.get("stability", [])
        if any(
            float(row.get(key, 0.0) or 0.0) > 0.0
            for key in pipeline_profile_metric_names()
        )
    ]
    if pipeline_rows:
        lines.extend(pipeline_markdown_header())
        for row in pipeline_rows:
            lines.append(format_pipeline_result_row(row))
    continuity = payload.get("resume_continuity", [])
    if continuity:
        lines.extend(
            [
                "",
                "## Resume Continuity",
                "",
                "| Run | Status | Steps | Loss ratio | Throughput ratio | Reasons |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for item in continuity:
            lines.append(
                "| {run} | {status} | {initial} -> {final} | {loss:.3f} "
                "| {throughput:.3f} | {reasons} |".format(
                    run=item.get("run_id", ""),
                    status=item.get("status", ""),
                    initial=item.get("initial_step", ""),
                    final=item.get("final_step", ""),
                    loss=float(item.get("loss_ratio", 0.0) or 0.0),
                    throughput=float(item.get("throughput_ratio", 0.0) or 0.0),
                    reasons=", ".join(item.get("reasons", [])) or "none",
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_stability_result_row(row: Mapping[str, Any]) -> str:
    return (
        "| {run} | {status} | {backend} | {step} | {loss:.6f} | {throughput:.3f} "
        "| {memory:.3f} | {wait:.3f} | {jitter:.3f} | {resumed} |"
    ).format(
        run=row.get("run_id", ""),
        status=row.get("status", ""),
        backend=row.get("backend", ""),
        step=row.get("global_step", ""),
        loss=float(row.get("stable_loss", 0.0) or 0.0),
        throughput=float(row.get("throughput", 0.0) or 0.0),
        memory=float(row.get("peak_memory_gb", 0.0) or 0.0),
        wait=float(row.get("stable_dataloader_wait_ms", 0.0) or 0.0),
        jitter=float(row.get("step_time_jitter_ratio", 0.0) or 0.0),
        resumed=bool(row.get("resumed", False)),
    )


def pipeline_profile_metric_names(*, include_counts: bool = False) -> list[str]:
    names = [
        "stable_pipeline_shard_read_ms",
        "stable_pipeline_tar_open_ms",
        "stable_pipeline_sample_decode_ms",
        "stable_pipeline_sample_tensor_build_ms",
        "stable_pipeline_sample_build_ms",
        "stable_pipeline_collate_ms",
        "stable_pipeline_image_decode_ms",
        "stable_pipeline_tokenizer_ms",
        "stable_pipeline_image_processor_ms",
        "stable_pipeline_processor_ms",
        "stable_pipeline_host_to_device_ms",
        "stable_pipeline_cuda_prefetch_h2d_ms",
        "stable_pipeline_cuda_prefetch_wait_ms",
        "stable_pipeline_cache_hit",
    ]
    if include_counts:
        names.extend(
            [
                "stable_pipeline_cache_hit_count",
                "stable_pipeline_cache_sample_count",
            ]
        )
    return names


def pipeline_markdown_header() -> list[str]:
    return [
        "",
        "## Input Pipeline Breakdown",
        "",
        (
            "| Run | Shard read ms | Sample decode ms | Collate ms "
            "| Decode ms | Prompt ms | Tokenizer ms | Image processor ms "
            "| Processor ms | H2D ms | Prefetch H2D ms "
            "| Prefetch wait ms | Cache hit |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
            "| ---: | ---: | ---: | ---: | ---: |"
        ),
    ]


def format_pipeline_result_row(row: Mapping[str, Any]) -> str:
    return (
        "| {run} | {shard:.3f} | {sample_decode:.3f} "
        "| {collate:.3f} | {decode:.3f} | {prompt:.3f} "
        "| {tokenizer:.3f} | {image_processor:.3f} "
        "| {processor:.3f} | {h2d:.3f} | {prefetch_h2d:.3f} "
        "| {prefetch_wait:.3f} | {cache:.3f} |"
    ).format(
        run=row.get("run_id", ""),
        shard=float(row.get("stable_pipeline_shard_read_ms", 0.0) or 0.0),
        sample_decode=float(row.get("stable_pipeline_sample_decode_ms", 0.0) or 0.0),
        collate=float(row.get("stable_pipeline_collate_ms", 0.0) or 0.0),
        decode=float(row.get("stable_pipeline_image_decode_ms", 0.0) or 0.0),
        prompt=float(row.get("stable_pipeline_prompt_template_ms", 0.0) or 0.0),
        tokenizer=float(row.get("stable_pipeline_tokenizer_ms", 0.0) or 0.0),
        image_processor=float(
            row.get("stable_pipeline_image_processor_ms", 0.0) or 0.0
        ),
        processor=float(row.get("stable_pipeline_processor_ms", 0.0) or 0.0),
        h2d=float(row.get("stable_pipeline_host_to_device_ms", 0.0) or 0.0),
        prefetch_h2d=float(row.get("stable_pipeline_cuda_prefetch_h2d_ms", 0.0) or 0.0),
        prefetch_wait=float(
            row.get("stable_pipeline_cuda_prefetch_wait_ms", 0.0) or 0.0
        ),
        cache=float(row.get("stable_pipeline_cache_hit", 0.0) or 0.0),
    )
