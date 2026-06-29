# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:54
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Metric and profile contracts used by benchmark and tuner code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class ProfileMetric:
    name: str
    value: float
    unit: str = ""
    source: str = "runtime"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
        }


@dataclass(frozen=True)
class MetricContract:
    """Names that ParaScale treats as stable cross-backend metrics."""

    throughput_metrics: tuple[str, ...] = (
        "samples_per_second",
        "images_per_second",
        "image_text_pairs_per_second",
        "tokens_per_second",
        "patch_tokens_per_second",
        "end_to_end_images_per_second",
        "end_to_end_image_text_pairs_per_second",
    )
    memory_metrics: tuple[str, ...] = (
        "peak_memory_bytes",
        "allocated_memory_bytes",
    )
    pipeline_metrics: tuple[str, ...] = (
        "pipeline_shard_read_ms",
        "pipeline_tar_open_ms",
        "pipeline_sample_decode_ms",
        "pipeline_sample_build_ms",
        "pipeline_collate_ms",
        "pipeline_image_decode_ms",
        "pipeline_prompt_template_ms",
        "pipeline_processor_ms",
        "pipeline_image_processor_ms",
        "pipeline_tokenizer_ms",
        "pipeline_sample_tensor_build_ms",
        "pipeline_label_build_ms",
        "pipeline_processor_unaccounted_ms",
        "pipeline_host_to_device_ms",
        "pipeline_cuda_prefetch_h2d_ms",
        "pipeline_cuda_prefetch_wait_ms",
        "pipeline_cache_hit",
        "pipeline_cache_hit_count",
        "pipeline_cache_sample_count",
    )

    def stable_metric_names(self) -> tuple[str, ...]:
        return self.throughput_metrics + self.memory_metrics + self.pipeline_metrics

    def filter_stable(self, metrics: Mapping[str, Any]) -> Dict[str, float]:
        stable: Dict[str, float] = {}
        for name in self.stable_metric_names():
            if name not in metrics:
                continue
            try:
                stable[name] = float(metrics[name])
            except (TypeError, ValueError):
                continue
        return stable
