# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Image-folder profiling utilities for real vision data smoke tests."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from parascale.runtime.backends.devices import select_torch_device

from .transforms import estimate_patch_tokens

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageFolderProfile:
    data_dir: str
    images: int = 0
    batches: int = 0
    patch_tokens: int = 0
    elapsed_seconds: float = 0.0
    decode_time_seconds: float = 0.0
    augment_time_seconds: float = 0.0
    host_to_device_time_seconds: float = 0.0
    image_size: int = 224
    patch_size: int = 16
    device: str = "cpu"
    warnings: List[str] = field(default_factory=list)

    @property
    def images_per_second(self) -> float:
        return self.images / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def patch_tokens_per_second(self) -> float:
        return (
            self.patch_tokens / self.elapsed_seconds
            if self.elapsed_seconds > 0
            else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "images": self.images,
            "batches": self.batches,
            "patch_tokens": self.patch_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "decode_time_ms": self.decode_time_seconds * 1000.0,
            "augment_time_ms": self.augment_time_seconds * 1000.0,
            "host_to_device_time_ms": self.host_to_device_time_seconds * 1000.0,
            "images_per_second": self.images_per_second,
            "patch_tokens_per_second": self.patch_tokens_per_second,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "device": self.device,
            "warnings": list(self.warnings),
        }


def find_image_files(data_dir: str | Path) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"image data directory does not exist: {root}")
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def profile_image_folder(
    data_dir: str | Path,
    *,
    batch_size: int = 8,
    max_batches: int = 4,
    image_size: int = 224,
    patch_size: int = 16,
    device: str = "auto",
) -> ImageFolderProfile:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_batches < 1:
        raise ValueError("max_batches must be >= 1")

    torch = _require_torch()
    Image = _require_pillow()
    files = find_image_files(data_dir)
    if not files:
        raise FileNotFoundError(f"no images found under {data_dir}")

    selected_device = select_torch_device(torch, device)
    profile = ImageFolderProfile(
        data_dir=str(data_dir),
        image_size=image_size,
        patch_size=patch_size,
        device=str(selected_device),
    )
    start = time.perf_counter()
    for batch_paths in _batched(files, batch_size):
        if profile.batches >= max_batches:
            break
        tensors = []
        metadata = []
        for path in batch_paths:
            decode_start = time.perf_counter()
            with Image.open(path) as image:
                image = image.convert("RGB")
                original_width, original_height = image.size
                profile.decode_time_seconds += time.perf_counter() - decode_start

                augment_start = time.perf_counter()
                image = image.resize((image_size, image_size))
                tensor = _pil_to_tensor(torch, image)
                profile.augment_time_seconds += time.perf_counter() - augment_start
            tensors.append(tensor)
            metadata.append({"height": original_height, "width": original_width})

        batch = torch.stack(tensors)
        transfer_start = time.perf_counter()
        batch = batch.to(selected_device, non_blocking=True)
        if str(selected_device).startswith("cuda"):
            torch.cuda.synchronize(selected_device)
        profile.host_to_device_time_seconds += time.perf_counter() - transfer_start
        _ = float(batch.mean().detach().cpu())

        profile.images += len(tensors)
        profile.patch_tokens += sum(
            estimate_patch_tokens(item["height"], item["width"], patch_size)
            for item in metadata
        )
        profile.batches += 1
    profile.elapsed_seconds = max(time.perf_counter() - start, 1e-9)
    return profile


def _batched(items: Sequence[Path], batch_size: int) -> Iterable[List[Path]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def _pil_to_tensor(torch: Any, image: Any):
    import numpy as np

    array = np.asarray(image, dtype="float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise ImportError("vision image-folder profiling requires PyTorch.") from exc
    return torch


def _require_pillow():
    try:
        from PIL import Image
    except Exception as exc:
        raise ImportError("vision image-folder profiling requires Pillow/PIL.") from exc
    return Image

