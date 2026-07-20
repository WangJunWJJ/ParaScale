# -*- coding: utf-8 -*-
# @Time : 2026/7/20 下午2:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Shared parsing helpers for benchmark summary tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

IMAGE_TEXT_THROUGHPUT_KEYS = (
    "stable_end_to_end_image_text_pairs_per_second",
    "end_to_end_image_text_pairs_per_second",
    "stable_end_to_end_images_per_second",
    "end_to_end_images_per_second",
    "stable_samples_per_second",
    "samples_per_second",
    "steps_per_second",
)

CLIP_THROUGHPUT_KEYS = (
    "stable_end_to_end_image_text_pairs_per_second",
    "end_to_end_image_text_pairs_per_second",
    "stable_end_to_end_images_per_second",
    "end_to_end_images_per_second",
    "image_text_pairs_per_second",
    "images_per_second",
)

SAMPLE_THROUGHPUT_KEYS = (
    "stable_end_to_end_samples_per_second",
    "end_to_end_samples_per_second",
    "stable_samples_per_second",
    "samples_per_second",
    "steps_per_second",
)


def read_json_payload(
    path: Path,
    *,
    tolerate_errors: bool = True,
    path_key: str = "path",
) -> Dict[str, Any]:
    """Read a benchmark JSON payload and annotate it with the source path."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        if not tolerate_errors:
            raise
        return {
            path_key: str(path),
            "status": "error",
            "error": str(exc),
        }
    payload.setdefault(path_key, str(path))
    return payload


def first_metric(
    metrics: Dict[str, Any],
    keys: Iterable[str],
) -> tuple[float, str | None]:
    """Return the first positive metric value and the field that supplied it."""

    if not isinstance(metrics, dict):
        return 0.0, None
    for key in keys:
        try:
            value = float(metrics.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value, key
    return 0.0, None


def metric_value(metrics: Dict[str, Any], keys: Iterable[str]) -> float:
    """Return the first positive metric value for a list of candidate keys."""

    value, _ = first_metric(metrics, keys)
    return value


def train_section(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return the nested train payload when present."""

    train = payload.get("train", {})
    return train if isinstance(train, dict) else {}


def merged_train_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Merge top-level, train-level, and metrics fields for summary extraction."""

    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    train = train_section(payload)
    last_metrics = train.get("last_metrics", {})
    if not isinstance(last_metrics, dict):
        last_metrics = {}
    top_last_metrics = payload.get("last_metrics", {})
    if not isinstance(top_last_metrics, dict):
        top_last_metrics = {}
    merged = {**top_last_metrics, **last_metrics, **metrics}
    if train.get("steps_per_second") is not None:
        merged.setdefault("steps_per_second", train.get("steps_per_second"))
    if payload.get("steps_per_second") is not None:
        merged.setdefault("steps_per_second", payload.get("steps_per_second"))
    return merged


def loss_value(
    payload: Dict[str, Any],
    *,
    include_top_level: bool = False,
    keys: Iterable[str] = ("loss", "stable_loss", "last_loss"),
) -> float | None:
    """Extract a scalar loss from common benchmark payload shapes."""

    metrics = payload.get("metrics", {})
    train = train_section(payload)
    last_metrics = train.get("last_metrics", {})
    top_last_metrics = payload.get("last_metrics", {})
    sources = [metrics, last_metrics]
    if include_top_level:
        sources.extend([top_last_metrics, payload])
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source:
                try:
                    return float(source[key])
                except (TypeError, ValueError):
                    return None
    return None


def run_id_from_path(path: Path) -> str:
    """Return a stable run id for normal and *.error.json payloads."""

    stem = path.stem
    return stem[: -len(".error")] if stem.endswith(".error") else stem


__all__ = [
    "CLIP_THROUGHPUT_KEYS",
    "IMAGE_TEXT_THROUGHPUT_KEYS",
    "SAMPLE_THROUGHPUT_KEYS",
    "first_metric",
    "loss_value",
    "merged_train_metrics",
    "metric_value",
    "read_json_payload",
    "run_id_from_path",
    "train_section",
]
