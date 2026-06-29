# -*- coding: utf-8 -*-
# @Time : 2026/6/22 上午10:50
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""YOLO detection adapters for generic vision preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from parascale.data.vision.preprocessor import (
    ProcessedVisionSample,
    VisionSample,
    VisionTransformConfig,
)


class YoloDetectionTargetAdapter:
    """Parse normalized YOLO label files into framework-neutral targets."""

    def cache_paths(self, sample: VisionSample) -> Sequence[Path | str]:
        if isinstance(sample.annotation, (str, Path)):
            return [sample.annotation]
        return []

    def cache_extra(self, transform: VisionTransformConfig) -> Mapping[str, Any]:
        return {"target_format": "yolo_detection_v1"}

    def build_target(self, sample: VisionSample) -> List[Dict[str, Any]]:
        if sample.annotation is None:
            return []
        if isinstance(sample.annotation, Mapping):
            return list(sample.annotation.get("labels", []))
        return read_yolo_label_rows(Path(sample.annotation), batch_index=0)


class YoloOfficialBatchAdapter:
    """Build the batch shape expected by Ultralytics official detection loss."""

    def __init__(self, *, image_size: int) -> None:
        self.image_size = int(image_size)

    def collate(self, samples: Sequence[ProcessedVisionSample]) -> Dict[str, Any]:
        torch = self._require_torch()
        images = [sample.pixel_values for sample in samples]
        labels: List[Dict[str, Any]] = []
        for batch_index, sample in enumerate(samples):
            for row in sample.target or []:
                labels.append(
                    {
                        "batch_idx": batch_index,
                        "cls": row["cls"],
                        "bbox": row["bbox"],
                    }
                )

        batch_images = torch.stack(images, dim=0)
        if labels:
            cls = torch.tensor(
                [[float(row["cls"])] for row in labels],
                dtype=torch.float32,
            )
            bboxes = torch.tensor(
                [row["bbox"] for row in labels],
                dtype=torch.float32,
            )
            batch_idx = torch.tensor(
                [float(row["batch_idx"]) for row in labels],
                dtype=torch.float32,
            )
        else:
            cls = torch.zeros((0, 1), dtype=torch.float32)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            batch_idx = torch.zeros((0,), dtype=torch.float32)

        batch_size = int(batch_images.shape[0])
        return {
            "official_loss": True,
            "pixel_values": batch_images,
            "img": batch_images,
            "cls": cls,
            "bboxes": bboxes,
            "batch_idx": batch_idx,
            "images": batch_size,
            "image_text_pairs": batch_size,
            "patch_tokens": int(batch_size * (self.image_size // 32) ** 2),
            "tokens": batch_size,
            "num_images": batch_size,
            "num_pairs": batch_size,
        }

    @staticmethod
    def _require_torch() -> Any:
        try:
            import torch

            return torch
        except Exception as exc:
            raise ImportError("YoloOfficialBatchAdapter requires torch.") from exc


def read_yolo_label_rows(label_path: Path, batch_index: int) -> List[Dict[str, Any]]:
    rows = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id, x_center, y_center, width, height = parts[:5]
            rows.append(
                {
                    "batch_idx": batch_index,
                    "cls": int(float(cls_id)),
                    "bbox": [
                        float(x_center),
                        float(y_center),
                        float(width),
                        float(height),
                    ],
                }
            )
    return rows
