# -*- coding: utf-8 -*-
# @Time : 2026/5/3 下午9:56
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Torch-free runtime profile structures for strategy feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .utils import get_value


@dataclass
class RuntimeProfile:
    peak_memory_per_gpu: int = 0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    images_per_second: float = 0.0
    patch_tokens_per_second: float = 0.0
    padding_ratio: float = 0.0
    oom_count: int = 0
    step_time_seconds: float = 0.0
    batch_tokens: int = 0
    dataloader_wait_ms: float = 0.0
    peak_memory_ratio: float = 0.0
    pipeline_shard_read_ms: float = 0.0
    pipeline_tar_open_ms: float = 0.0
    pipeline_sample_decode_ms: float = 0.0
    pipeline_sample_tensor_build_ms: float = 0.0
    pipeline_sample_build_ms: float = 0.0
    pipeline_collate_ms: float = 0.0
    pipeline_image_decode_ms: float = 0.0
    pipeline_prompt_template_ms: float = 0.0
    pipeline_processor_ms: float = 0.0
    pipeline_tokenizer_ms: float = 0.0
    pipeline_image_processor_ms: float = 0.0
    pipeline_host_to_device_ms: float = 0.0
    pipeline_cuda_prefetch_h2d_ms: float = 0.0
    pipeline_cuda_prefetch_wait_ms: float = 0.0
    pipeline_label_build_ms: float = 0.0
    pipeline_processor_unaccounted_ms: float = 0.0
    pipeline_cache_hit: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_memory_per_gpu": self.peak_memory_per_gpu,
            "tokens_per_second": self.tokens_per_second,
            "samples_per_second": self.samples_per_second,
            "images_per_second": self.images_per_second,
            "patch_tokens_per_second": self.patch_tokens_per_second,
            "padding_ratio": self.padding_ratio,
            "oom_count": self.oom_count,
            "step_time_seconds": self.step_time_seconds,
            "batch_tokens": self.batch_tokens,
            "dataloader_wait_ms": self.dataloader_wait_ms,
            "peak_memory_ratio": self.peak_memory_ratio,
            "pipeline_shard_read_ms": self.pipeline_shard_read_ms,
            "pipeline_tar_open_ms": self.pipeline_tar_open_ms,
            "pipeline_sample_decode_ms": self.pipeline_sample_decode_ms,
            "pipeline_sample_tensor_build_ms": self.pipeline_sample_tensor_build_ms,
            "pipeline_sample_build_ms": self.pipeline_sample_build_ms,
            "pipeline_collate_ms": self.pipeline_collate_ms,
            "pipeline_image_decode_ms": self.pipeline_image_decode_ms,
            "pipeline_prompt_template_ms": self.pipeline_prompt_template_ms,
            "pipeline_processor_ms": self.pipeline_processor_ms,
            "pipeline_tokenizer_ms": self.pipeline_tokenizer_ms,
            "pipeline_image_processor_ms": self.pipeline_image_processor_ms,
            "pipeline_host_to_device_ms": self.pipeline_host_to_device_ms,
            "pipeline_cuda_prefetch_h2d_ms": self.pipeline_cuda_prefetch_h2d_ms,
            "pipeline_cuda_prefetch_wait_ms": self.pipeline_cuda_prefetch_wait_ms,
            "pipeline_label_build_ms": self.pipeline_label_build_ms,
            "pipeline_processor_unaccounted_ms": (
                self.pipeline_processor_unaccounted_ms
            ),
            "pipeline_cache_hit": self.pipeline_cache_hit,
        }


@dataclass
class BatchRuntimeStats:
    batch_tokens: int = 0
    valid_tokens: int = 0
    samples: int = 0
    step_time_seconds: float = 0.0
    peak_memory_per_gpu: int = 0
    oom_count: int = 0
    dataloader_wait_ms: float = 0.0
    pipeline_shard_read_ms: float = 0.0
    pipeline_tar_open_ms: float = 0.0
    pipeline_sample_decode_ms: float = 0.0
    pipeline_sample_tensor_build_ms: float = 0.0
    pipeline_sample_build_ms: float = 0.0
    pipeline_collate_ms: float = 0.0
    pipeline_image_decode_ms: float = 0.0
    pipeline_prompt_template_ms: float = 0.0
    pipeline_processor_ms: float = 0.0
    pipeline_tokenizer_ms: float = 0.0
    pipeline_image_processor_ms: float = 0.0
    pipeline_host_to_device_ms: float = 0.0
    pipeline_cuda_prefetch_h2d_ms: float = 0.0
    pipeline_cuda_prefetch_wait_ms: float = 0.0
    pipeline_label_build_ms: float = 0.0
    pipeline_processor_unaccounted_ms: float = 0.0
    pipeline_cache_hit: float = 0.0

    def to_runtime_profile(self) -> RuntimeProfile:
        return build_runtime_profile(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_tokens": self.batch_tokens,
            "valid_tokens": self.valid_tokens,
            "samples": self.samples,
            "step_time_seconds": self.step_time_seconds,
            "peak_memory_per_gpu": self.peak_memory_per_gpu,
            "oom_count": self.oom_count,
            "dataloader_wait_ms": self.dataloader_wait_ms,
            "pipeline_shard_read_ms": self.pipeline_shard_read_ms,
            "pipeline_tar_open_ms": self.pipeline_tar_open_ms,
            "pipeline_sample_decode_ms": self.pipeline_sample_decode_ms,
            "pipeline_sample_tensor_build_ms": self.pipeline_sample_tensor_build_ms,
            "pipeline_sample_build_ms": self.pipeline_sample_build_ms,
            "pipeline_collate_ms": self.pipeline_collate_ms,
            "pipeline_image_decode_ms": self.pipeline_image_decode_ms,
            "pipeline_prompt_template_ms": self.pipeline_prompt_template_ms,
            "pipeline_processor_ms": self.pipeline_processor_ms,
            "pipeline_tokenizer_ms": self.pipeline_tokenizer_ms,
            "pipeline_image_processor_ms": self.pipeline_image_processor_ms,
            "pipeline_host_to_device_ms": self.pipeline_host_to_device_ms,
            "pipeline_cuda_prefetch_h2d_ms": self.pipeline_cuda_prefetch_h2d_ms,
            "pipeline_cuda_prefetch_wait_ms": self.pipeline_cuda_prefetch_wait_ms,
            "pipeline_label_build_ms": self.pipeline_label_build_ms,
            "pipeline_processor_unaccounted_ms": (
                self.pipeline_processor_unaccounted_ms
            ),
            "pipeline_cache_hit": self.pipeline_cache_hit,
        }


def build_runtime_profile(stats: Any) -> RuntimeProfile:
    batch_tokens = max(0, int(get_value(stats, "batch_tokens", 0) or 0))
    valid_tokens = max(0, int(get_value(stats, "valid_tokens", batch_tokens) or 0))
    samples = max(0, int(get_value(stats, "samples", 0) or 0))
    images = max(0, int(get_value(stats, "images", samples) or 0))
    patch_tokens = max(0, int(get_value(stats, "patch_tokens", 0) or 0))
    step_time = max(0.0, float(get_value(stats, "step_time_seconds", 0.0) or 0.0))

    padding_tokens = max(0, batch_tokens - valid_tokens)
    padding_ratio = padding_tokens / batch_tokens if batch_tokens > 0 else 0.0
    tokens_per_second = valid_tokens / step_time if step_time > 0 else 0.0
    samples_per_second = samples / step_time if step_time > 0 else 0.0
    images_per_second = images / step_time if step_time > 0 else 0.0
    patch_tokens_per_second = patch_tokens / step_time if step_time > 0 else 0.0

    return RuntimeProfile(
        peak_memory_per_gpu=int(get_value(stats, "peak_memory_per_gpu", 0) or 0),
        tokens_per_second=tokens_per_second,
        samples_per_second=samples_per_second,
        images_per_second=images_per_second,
        patch_tokens_per_second=patch_tokens_per_second,
        padding_ratio=padding_ratio,
        oom_count=int(get_value(stats, "oom_count", 0) or 0),
        step_time_seconds=step_time,
        batch_tokens=batch_tokens,
        dataloader_wait_ms=float(get_value(stats, "dataloader_wait_ms", 0.0) or 0.0),
        pipeline_shard_read_ms=float(
            get_value(stats, "pipeline_shard_read_ms", 0.0) or 0.0
        ),
        pipeline_tar_open_ms=float(
            get_value(stats, "pipeline_tar_open_ms", 0.0) or 0.0
        ),
        pipeline_sample_decode_ms=float(
            get_value(stats, "pipeline_sample_decode_ms", 0.0) or 0.0
        ),
        pipeline_sample_tensor_build_ms=float(
            get_value(stats, "pipeline_sample_tensor_build_ms", 0.0) or 0.0
        ),
        pipeline_sample_build_ms=float(
            get_value(stats, "pipeline_sample_build_ms", 0.0) or 0.0
        ),
        pipeline_collate_ms=float(get_value(stats, "pipeline_collate_ms", 0.0) or 0.0),
        peak_memory_ratio=float(get_value(stats, "peak_memory_ratio", 0.0) or 0.0),
        pipeline_image_decode_ms=float(
            get_value(stats, "pipeline_image_decode_ms", 0.0) or 0.0
        ),
        pipeline_prompt_template_ms=float(
            get_value(stats, "pipeline_prompt_template_ms", 0.0) or 0.0
        ),
        pipeline_processor_ms=float(
            get_value(stats, "pipeline_processor_ms", 0.0) or 0.0
        ),
        pipeline_tokenizer_ms=float(
            get_value(stats, "pipeline_tokenizer_ms", 0.0) or 0.0
        ),
        pipeline_image_processor_ms=float(
            get_value(stats, "pipeline_image_processor_ms", 0.0) or 0.0
        ),
        pipeline_host_to_device_ms=float(
            get_value(stats, "pipeline_host_to_device_ms", 0.0) or 0.0
        ),
        pipeline_cuda_prefetch_h2d_ms=float(
            get_value(stats, "pipeline_cuda_prefetch_h2d_ms", 0.0) or 0.0
        ),
        pipeline_cuda_prefetch_wait_ms=float(
            get_value(stats, "pipeline_cuda_prefetch_wait_ms", 0.0) or 0.0
        ),
        pipeline_label_build_ms=float(
            get_value(stats, "pipeline_label_build_ms", 0.0) or 0.0
        ),
        pipeline_processor_unaccounted_ms=float(
            get_value(stats, "pipeline_processor_unaccounted_ms", 0.0) or 0.0
        ),
        pipeline_cache_hit=float(get_value(stats, "pipeline_cache_hit", 0.0) or 0.0),
    )
