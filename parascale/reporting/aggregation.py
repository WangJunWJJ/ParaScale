# -*- coding: utf-8 -*-
# @Time : 2026/6/25 下午9:36
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Benchmark metric aggregation helpers."""

from __future__ import annotations

from typing import Any, Dict


def aggregate_stable_metrics(
    history: Any,
    *,
    warmup_steps: int = 0,
) -> Dict[str, float]:
    if not isinstance(history, list):
        return {"measured_steps": 0}
    window = [
        item for item in history[max(0, int(warmup_steps)) :] if isinstance(item, dict)
    ]
    if not window:
        return {"measured_steps": 0}
    metric_names = [
        "loss",
        "step_time_seconds",
        "images_per_second",
        "end_to_end_images_per_second",
        "tokens_per_second",
        "end_to_end_tokens_per_second",
        "patch_tokens_per_second",
        "end_to_end_patch_tokens_per_second",
        "image_text_pairs_per_second",
        "end_to_end_image_text_pairs_per_second",
        "dataloader_wait_ms",
        "peak_memory_bytes",
        "allocated_memory_bytes",
        "padding_ratio",
        "adapter_params",
        "trainable_params",
        "total_params",
        "trainable_ratio",
        "lora_rank",
        "pipeline_shard_read_ms",
        "pipeline_tar_open_ms",
        "pipeline_sample_decode_ms",
        "pipeline_sample_tensor_build_ms",
        "pipeline_sample_build_ms",
        "pipeline_collate_ms",
        "pipeline_image_decode_ms",
        "pipeline_prompt_template_ms",
        "pipeline_processor_ms",
        "pipeline_tokenizer_ms",
        "pipeline_image_processor_ms",
        "pipeline_host_to_device_ms",
        "pipeline_cuda_prefetch_h2d_ms",
        "pipeline_cuda_prefetch_wait_ms",
        "pipeline_label_build_ms",
        "pipeline_processor_unaccounted_ms",
        "pipeline_cache_hit",
        "pipeline_cache_hit_count",
        "pipeline_cache_sample_count",
    ]
    aggregated: Dict[str, float] = {"measured_steps": float(len(window))}
    for name in metric_names:
        values = []
        for item in window:
            value = item.get(name)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            aggregated[f"stable_{name}"] = sum(values) / len(values)
            aggregated[f"stable_max_{name}"] = max(values)
            aggregated[f"stable_min_{name}"] = min(values)
    if "stable_step_time_seconds" in aggregated:
        aggregated["stable_step_time_ms"] = (
            aggregated["stable_step_time_seconds"] * 1000.0
        )
    return aggregated


__all__ = ["aggregate_stable_metrics"]
