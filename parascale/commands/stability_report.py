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
    }
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
            "| Run | Status | Backend | Step | Throughput | Peak GB | Wait ms "
            "| Jitter | Resumed |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_stability_result_row(row: Mapping[str, Any]) -> str:
    return (
        "| {run} | {status} | {backend} | {step} | {throughput:.3f} "
        "| {memory:.3f} | {wait:.3f} | {jitter:.3f} | {resumed} |"
    ).format(
        run=row.get("run_id", ""),
        status=row.get("status", ""),
        backend=row.get("backend", ""),
        step=row.get("global_step", ""),
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
