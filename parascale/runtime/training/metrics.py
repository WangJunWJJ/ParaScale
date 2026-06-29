# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午5:37
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Runtime metric and pipeline profile helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RuntimeMetrics:
    world_size: int = 1

    def with_throughput_metrics(
        self,
        metrics: Dict[str, Any],
        batch: Any,
        elapsed_seconds: float,
    ) -> Dict[str, Any]:
        elapsed = max(float(elapsed_seconds), 1e-9)
        if isinstance(batch, dict):
            images = batch.get("num_images")
            pairs = batch.get("num_pairs")
            patch_tokens = batch.get("patch_tokens")
            if images is None and hasattr(batch.get("pixel_values"), "shape"):
                images = int(batch["pixel_values"].shape[0])
            batch_size = pairs if pairs is not None else images
            scale = max(1, int(self.world_size))
            if batch_size is not None:
                metrics["batch_size"] = int(batch_size) * scale
            if images is not None:
                images = int(images) * scale
                metrics.setdefault("step_time_seconds", elapsed)
                metrics["images"] = images
                metrics["images_per_second"] = float(images) / elapsed
            if patch_tokens is not None:
                patch_tokens = sum_metric_value(patch_tokens) * scale
                metrics.setdefault("step_time_seconds", elapsed)
                metrics["patch_tokens"] = int(patch_tokens)
                metrics["patch_tokens_per_second"] = float(patch_tokens) / elapsed
            tokens = batch.get("tokens")
            if tokens is not None:
                tokens = int(tokens) * scale
                metrics.setdefault("step_time_seconds", elapsed)
                metrics["tokens"] = int(tokens)
                metrics["tokens_per_second"] = float(tokens) / elapsed
            if pairs is not None:
                pairs = int(pairs) * scale
                metrics.setdefault("step_time_seconds", elapsed)
                metrics["image_text_pairs"] = int(pairs)
                metrics["image_text_pairs_per_second"] = float(pairs) / elapsed
            if batch.get("padding_ratio") is not None:
                metrics["padding_ratio"] = float(batch["padding_ratio"])
            for key in [
                "adapter_params",
                "trainable_params",
                "total_params",
                "trainable_ratio",
                "lora_rank",
            ]:
                if batch.get(key) is not None:
                    value = batch[key]
                    metrics[key] = (
                        float(value)
                        if key == "trainable_ratio"
                        else int(sum_metric_value(value))
                    )
            add_pipeline_profile_metrics(metrics, batch)
        return metrics

    def add_end_to_end_metrics(
        self,
        metrics: Dict[str, Any],
        batch: Any,
        dataloader_wait_seconds: float,
    ) -> None:
        wait = max(
            (
                float(metrics.get("dataloader_wait_ms", 0.0) or 0.0) / 1000.0
                if "dataloader_wait_ms" in metrics
                else float(dataloader_wait_seconds)
            ),
            0.0,
        )
        metrics["dataloader_wait_ms"] = wait * 1000.0
        step = float(metrics.get("step_time_seconds", 0.0) or 0.0)
        total = max(step + wait, 1e-9)
        if isinstance(batch, dict):
            images = metrics.get("images")
            tokens = metrics.get("tokens")
            patch_tokens = metrics.get("patch_tokens")
            pairs = metrics.get("image_text_pairs")
            if images is not None:
                metrics["end_to_end_images_per_second"] = float(images) / total
            if tokens is not None:
                metrics["end_to_end_tokens_per_second"] = float(tokens) / total
            if patch_tokens is not None:
                metrics["end_to_end_patch_tokens_per_second"] = (
                    float(patch_tokens) / total
                )
            if pairs is not None:
                metrics["end_to_end_image_text_pairs_per_second"] = float(pairs) / total


def add_pipeline_profile_metrics(
    metrics: Dict[str, Any],
    batch: Dict[str, Any],
) -> None:
    profile = batch.get("pipeline_profile")
    if not isinstance(profile, dict):
        return
    for key, value in profile.items():
        if not isinstance(value, (int, float)):
            continue
        metric_name = f"pipeline_{key}"
        metrics[metric_name] = float(value)
    processor = float(metrics.get("pipeline_processor_ms", 0.0) or 0.0)
    tokenizer = float(metrics.get("pipeline_tokenizer_ms", 0.0) or 0.0)
    image_processor = float(metrics.get("pipeline_image_processor_ms", 0.0) or 0.0)
    unaccounted = processor - tokenizer - image_processor
    metrics["pipeline_processor_unaccounted_ms"] = max(0.0, unaccounted)


def metric_value(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return value


def sum_metric_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return sum(value)
    if hasattr(value, "sum"):
        summed = value.sum()
        item = getattr(summed, "item", None)
        return item() if callable(item) else summed
    return value


def merge_accumulated_batches(batches: list[Any]) -> Any:
    if not batches:
        return {}
    if not all(isinstance(batch, dict) for batch in batches):
        return batches[0]
    merged: Dict[str, Any] = dict(batches[0])
    for key in [
        "num_images",
        "num_pairs",
        "patch_tokens",
        "tokens",
        "images",
        "image_text_pairs",
    ]:
        values = [batch.get(key) for batch in batches if batch.get(key) is not None]
        if values:
            merged[key] = sum(int(sum_metric_value(value)) for value in values)
    profiles = [
        batch.get("pipeline_profile")
        for batch in batches
        if isinstance(batch.get("pipeline_profile"), dict)
    ]
    if profiles:
        merged["pipeline_profile"] = merge_pipeline_profiles(profiles)
    return merged


def merge_pipeline_profiles(profiles: list[Dict[str, Any]]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    cache_hits: list[float] = []
    for profile in profiles:
        for key, value in profile.items():
            if isinstance(value, (int, float)):
                if key == "cache_hit":
                    cache_hits.append(float(value))
                else:
                    merged[key] = merged.get(key, 0.0) + float(value)
    if cache_hits:
        hit_count = sum(max(0.0, min(1.0, value)) for value in cache_hits)
        sample_count = float(len(cache_hits))
        merged["cache_hit"] = hit_count / max(sample_count, 1.0)
        merged["cache_hit_count"] = hit_count
        merged["cache_sample_count"] = sample_count
    return merged
