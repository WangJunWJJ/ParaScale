# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午9:07
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Benchmark profile persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


class BenchmarkProfileStore:
    def runtime_profile_from_config(
        self, config_data: Mapping[str, Any]
    ) -> Dict[str, Any]:
        explicit = config_data.get("runtime_profile")
        if isinstance(explicit, Mapping):
            return self.runtime_profile_from_metrics(explicit)

        benchmark_profile = config_data.get("benchmark_profile")
        if isinstance(benchmark_profile, Mapping):
            path = benchmark_profile.get("path") or benchmark_profile.get("result_path")
            if path:
                return self.runtime_profile_from_path(path)
            metrics = benchmark_profile.get("metrics", benchmark_profile)
            if isinstance(metrics, Mapping):
                return self.runtime_profile_from_metrics(metrics)

        path = config_data.get("benchmark_result_path") or config_data.get(
            "benchmark_profile_path"
        )
        if path:
            return self.runtime_profile_from_path(path)
        return {}

    def runtime_profile_from_path(self, path: str | Path) -> Dict[str, Any]:
        return self.runtime_profile_from_payload(self.read(path))

    def runtime_profile_from_payload(
        self, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for section_name in ["metrics", "benchmark_result"]:
            section = payload.get(section_name)
            if not isinstance(section, Mapping):
                continue
            if section_name == "benchmark_result":
                section = section.get("metrics", {})
            if isinstance(section, Mapping):
                metrics.update(section)
        train = payload.get("train")
        if isinstance(train, Mapping):
            last_metrics = train.get("last_metrics")
            if isinstance(last_metrics, Mapping):
                metrics.update({key: value for key, value in last_metrics.items()})
        return self.runtime_profile_from_metrics(metrics)

    def runtime_profile_from_metrics(
        self, metrics: Mapping[str, Any]
    ) -> Dict[str, Any]:
        keys = [
            "peak_memory_per_gpu",
            "tokens_per_second",
            "samples_per_second",
            "images_per_second",
            "patch_tokens_per_second",
            "padding_ratio",
            "oom_count",
            "step_time_seconds",
            "batch_tokens",
            "dataloader_wait_ms",
            "peak_memory_ratio",
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
        runtime: Dict[str, Any] = {}
        for key in keys:
            value = metrics.get(key)
            if value is None:
                value = metrics.get(f"stable_{key}")
            if isinstance(value, (int, float)):
                runtime[key] = value
        memory = metrics.get("peak_memory_bytes")
        if isinstance(memory, (int, float)) and "peak_memory_per_gpu" not in runtime:
            runtime["peak_memory_per_gpu"] = int(memory)
        tokens = metrics.get("tokens")
        if isinstance(tokens, (int, float)) and "batch_tokens" not in runtime:
            runtime["batch_tokens"] = int(tokens)
        return runtime

    def write(self, path: str | Path, profile: Mapping[str, Any]) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(dict(profile), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output

    def read(self, path: str | Path) -> Dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
