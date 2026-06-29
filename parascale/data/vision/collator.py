# -*- coding: utf-8 -*-
# @Time : 2026/5/4 上午1:08
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Vision collators."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from .transforms import estimate_patch_tokens, normalize_vision_sample


class VisionCollator:
    """Collate vision samples into a model-ready batch when torch is available.

    Non-tensor payloads are kept as lists so lightweight no-torch tests and
    metadata-only planning paths remain usable.
    """

    def __init__(
        self,
        *,
        patch_size: int = 16,
        pad_value: float = 0.0,
        pad_to_multiple: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> None:
        self.patch_size = int(patch_size)
        self.pad_value = float(pad_value)
        self.pad_to_multiple = pad_to_multiple
        self.return_tensors = return_tensors

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        if not samples:
            return batch
        normalized = [normalize_vision_sample(sample) for sample in samples]
        for sample in normalized:
            self._fill_tensor_resolution(sample)
        keys = set().union(*(sample.keys() for sample in normalized))
        for sample in normalized:
            sample.setdefault(
                "patch_tokens",
                estimate_patch_tokens(
                    int(sample.get("height", 224)),
                    int(sample.get("width", 224)),
                    self.patch_size,
                ),
            )
        for key in keys:
            values = [sample.get(key) for sample in normalized]
            if key == "pixel_values":
                batch[key] = self._collate_pixel_values(values)
            elif key in {"label", "labels"}:
                batch["labels"] = self._collate_labels(values)
            else:
                batch[key] = list(values)
        image_sizes = [
            (int(sample.get("height", 224)), int(sample.get("width", 224)))
            for sample in normalized
        ]
        per_sample_patch_tokens = [int(sample["patch_tokens"]) for sample in normalized]
        batch["image_sizes"] = image_sizes
        batch["per_sample_patch_tokens"] = per_sample_patch_tokens
        batch["patch_tokens"] = sum(per_sample_patch_tokens)
        batch["num_images"] = len(normalized)
        batch.setdefault(
            "metadata",
            [
                {
                    "height": height,
                    "width": width,
                    "patch_tokens": patch_tokens,
                }
                for (height, width), patch_tokens in zip(
                    image_sizes, per_sample_patch_tokens
                )
            ],
        )
        return batch

    def _collate_pixel_values(self, values: Sequence[Any]) -> Any:
        if not values:
            return []
        first = values[0]
        if not hasattr(first, "shape"):
            return list(values)
        torch = self._optional_torch()
        if torch is None:
            return list(values)
        tensors = [self._as_chw_tensor(value, torch) for value in values]
        channels = {int(tensor.shape[0]) for tensor in tensors if tensor.ndim == 3}
        if len(channels) != 1 or any(tensor.ndim != 3 for tensor in tensors):
            return torch.stack(tensors, dim=0)
        target_height, target_width = self._target_hw(tensors)
        padded = [
            self._pad_chw_tensor(torch, tensor, target_height, target_width)
            for tensor in tensors
        ]
        return torch.stack(padded, dim=0)

    def _collate_labels(self, values: Sequence[Any]) -> Any:
        torch = self._optional_torch()
        if (
            torch is not None
            and values
            and all(isinstance(value, (int, bool)) for value in values)
        ):
            return torch.tensor(list(values), dtype=torch.long)
        if (
            torch is not None
            and values
            and all(isinstance(value, float) for value in values)
        ):
            return torch.tensor(list(values), dtype=torch.float32)
        return list(values)

    def _target_hw(self, tensors: Sequence[Any]) -> Tuple[int, int]:
        height = max(int(tensor.shape[-2]) for tensor in tensors)
        width = max(int(tensor.shape[-1]) for tensor in tensors)
        if self.pad_to_multiple:
            height = self._round_up(height, int(self.pad_to_multiple))
            width = self._round_up(width, int(self.pad_to_multiple))
        return height, width

    @staticmethod
    def _round_up(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    def _pad_chw_tensor(self, torch: Any, tensor: Any, height: int, width: int) -> Any:
        pad_height = height - int(tensor.shape[-2])
        pad_width = width - int(tensor.shape[-1])
        if pad_height < 0 or pad_width < 0:
            raise ValueError("target image size must be greater than sample size.")
        if pad_height == 0 and pad_width == 0:
            return tensor
        return torch.nn.functional.pad(
            tensor,
            (0, pad_width, 0, pad_height),
            value=self.pad_value,
        )

    def _as_chw_tensor(self, value: Any, torch: Any) -> Any:
        tensor = value if hasattr(value, "detach") else torch.as_tensor(value)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError("vision pixel_values must be CHW tensors or arrays.")
        return tensor

    @staticmethod
    def _fill_tensor_resolution(sample: Dict[str, Any]) -> None:
        pixel_values = sample.get("pixel_values")
        if not hasattr(pixel_values, "shape") or len(pixel_values.shape) < 2:
            return
        if (
            sample.get("height") == 224
            and sample.get("width") == 224
            and tuple(pixel_values.shape[-2:]) != (224, 224)
        ):
            sample["height"] = int(pixel_values.shape[-2])
            sample["width"] = int(pixel_values.shape[-1])

    @staticmethod
    def _optional_torch() -> Any:
        try:
            import torch

            return torch
        except Exception:
            return None
