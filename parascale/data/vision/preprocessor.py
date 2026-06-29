# -*- coding: utf-8 -*-
# @Time : 2026/6/22 上午10:49
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Generic vision sample preprocessing with cache-aware profiling."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol, Sequence

from .cache import DiskTensorCache


@dataclass
class VisionSample:
    image: Path | str | bytes
    sample_id: str | None = None
    annotation: Path | str | Mapping[str, Any] | None = None
    text: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisionTransformConfig:
    image_size: int | tuple[int, int] = 224
    normalize: str = "unit"
    resize_mode: str = "square"
    cache_format: str = "vision_tensor_v1"

    def size_hw(self) -> tuple[int, int]:
        if isinstance(self.image_size, tuple):
            return int(self.image_size[0]), int(self.image_size[1])
        return int(self.image_size), int(self.image_size)

    def cache_extra(self) -> Dict[str, Any]:
        height, width = self.size_hw()
        return {
            "height": height,
            "width": width,
            "normalize": self.normalize,
            "resize_mode": self.resize_mode,
            "format": self.cache_format,
        }


@dataclass
class ProcessedVisionSample:
    pixel_values: Any
    target: Any = None
    sample_id: str | None = None
    text: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    profile: Dict[str, float] = field(default_factory=dict)


class VisionTargetAdapter(Protocol):
    def cache_paths(self, sample: VisionSample) -> Sequence[Path | str]: ...

    def cache_extra(self, transform: VisionTransformConfig) -> Mapping[str, Any]: ...

    def build_target(self, sample: VisionSample) -> Any: ...


class NullVisionTargetAdapter:
    def cache_paths(self, sample: VisionSample) -> Sequence[Path | str]:
        return []

    def cache_extra(self, transform: VisionTransformConfig) -> Mapping[str, Any]:
        return {}

    def build_target(self, sample: VisionSample) -> Any:
        return None


class VisionPreprocessor:
    """Build model-neutral vision tensors and optional targets."""

    def __init__(
        self,
        *,
        transform: VisionTransformConfig,
        target_adapter: VisionTargetAdapter | None = None,
        tensor_cache_dir: str | Path | None = None,
        tensor_cache: bool = False,
    ) -> None:
        self.transform = transform
        self.target_adapter = target_adapter or NullVisionTargetAdapter()
        self.cache = DiskTensorCache(tensor_cache_dir, enabled=tensor_cache)

    def process(self, sample: VisionSample) -> ProcessedVisionSample:
        torch = self._require_torch()
        np = self._require_numpy()
        pil_image = self._require_pil_image()
        cache_key = self._cache_key(sample)
        cached = self.cache.load(cache_key, torch)
        if isinstance(cached, dict) and "pixel_values" in cached:
            profile = self._empty_profile()
            profile["tensor_cache_hit_count"] = 1.0
            profile["tensor_cache_sample_count"] = 1.0
            profile["cache_hit_count"] = 1.0
            profile["cache_sample_count"] = 1.0
            profile["cache_hit"] = 1.0
            return ProcessedVisionSample(
                pixel_values=cached["pixel_values"],
                target=cached.get("target"),
                sample_id=sample.sample_id,
                text=sample.text,
                metadata=dict(sample.metadata),
                profile=profile,
            )

        profile = self._empty_profile()
        decode_start = time.perf_counter()
        image = self._open_image(pil_image, sample.image).convert("RGB")
        profile["image_decode_ms"] = (time.perf_counter() - decode_start) * 1000.0

        resize_start = time.perf_counter()
        height, width = self.transform.size_hw()
        image = image.resize((width, height), pil_image.BILINEAR)
        profile["image_resize_ms"] = (time.perf_counter() - resize_start) * 1000.0
        profile["image_processor_ms"] = profile["image_resize_ms"]

        tensor_start = time.perf_counter()
        array = np.asarray(image, dtype="uint8").copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()
        if self.transform.normalize == "unit":
            tensor = tensor.div(255.0)
        profile["image_tensor_build_ms"] = (time.perf_counter() - tensor_start) * 1000.0
        profile["sample_tensor_build_ms"] = profile["image_tensor_build_ms"]

        target_start = time.perf_counter()
        target = self.target_adapter.build_target(sample)
        profile["target_build_ms"] = (time.perf_counter() - target_start) * 1000.0
        profile["label_build_ms"] = profile["target_build_ms"]

        payload = {"pixel_values": tensor, "target": target, "profile": profile}
        self.cache.save(cache_key, payload, torch)
        return ProcessedVisionSample(
            pixel_values=tensor,
            target=target,
            sample_id=sample.sample_id,
            text=sample.text,
            metadata=dict(sample.metadata),
            profile=profile,
        )

    def _cache_key(self, sample: VisionSample) -> str:
        paths: list[Path | str] = []
        if not isinstance(sample.image, bytes):
            paths.append(sample.image)
        paths.extend(self.target_adapter.cache_paths(sample))
        extra = dict(self.transform.cache_extra())
        if isinstance(sample.image, bytes):
            extra["image_bytes_sha256"] = hashlib.sha256(sample.image).hexdigest()
        extra.update(self.target_adapter.cache_extra(self.transform))
        return self.cache.key_for_paths(*paths, extra=extra)

    @staticmethod
    def _open_image(pil_image: Any, image: Path | str | bytes) -> Any:
        if isinstance(image, bytes):
            return pil_image.open(BytesIO(image))
        return pil_image.open(image)

    @staticmethod
    def _empty_profile() -> Dict[str, float]:
        return {
            "image_decode_ms": 0.0,
            "image_resize_ms": 0.0,
            "image_processor_ms": 0.0,
            "image_tensor_build_ms": 0.0,
            "sample_tensor_build_ms": 0.0,
            "target_build_ms": 0.0,
            "label_build_ms": 0.0,
            "tensor_cache_hit_count": 0.0,
            "tensor_cache_sample_count": 1.0,
            "cache_hit_count": 0.0,
            "cache_sample_count": 1.0,
            "cache_hit": 0.0,
        }

    @staticmethod
    def _require_torch() -> Any:
        try:
            import torch

            return torch
        except Exception as exc:
            raise ImportError("VisionPreprocessor requires torch.") from exc

    @staticmethod
    def _require_numpy() -> Any:
        try:
            import numpy as np

            return np
        except Exception as exc:
            raise ImportError("VisionPreprocessor requires numpy.") from exc

    @staticmethod
    def _require_pil_image() -> Any:
        try:
            from PIL import Image

            return Image
        except Exception as exc:
            raise ImportError("VisionPreprocessor requires Pillow.") from exc
