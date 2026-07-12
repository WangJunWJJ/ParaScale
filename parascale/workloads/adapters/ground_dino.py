# -*- coding: utf-8 -*-
# @Time : 2026/7/11 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""GroundingDINO adapters for generic detection batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from parascale.data.vision.preprocessor import (
    ProcessedVisionSample,
    VisionSample,
    VisionTransformConfig,
)


class GroundDinoBatchAdapter:
    """Build a lightweight GroundingDINO training batch.

    The adapter keeps model-specific text/image processor output separate from
    framework-neutral detection targets. The training proxy consumes the target
    tensors and computes a compact proxy loss suitable for smoke and throughput
    validation.
    """

    def __init__(
        self,
        *,
        prompt: str,
        image_size: int,
        loss_type: str = "proxy",
        tokenizer: Any | None = None,
    ) -> None:
        self.image_size = int(image_size)
        self.loss_type = str(loss_type).lower()
        self.prompt = self._normalize_prompt(prompt)
        self.tokenizer = tokenizer

    def collate(self, samples: Sequence[ProcessedVisionSample]) -> Dict[str, Any]:
        torch = self._require_torch()
        images = [sample.pixel_values for sample in samples]
        batch_images = torch.stack(images, dim=0)
        target_boxes = []
        target_mask = []
        target_classes = []
        for sample in samples:
            rows = list(sample.target or [])
            if rows:
                first = rows[0]
                target_boxes.append(first["bbox"])
                target_classes.append(int(first.get("cls", 0)))
                target_mask.append(1.0)
            else:
                target_boxes.append([0.5, 0.5, 0.1, 0.1])
                target_classes.append(0)
                target_mask.append(0.0)

        batch_size = int(batch_images.shape[0])
        batch: Dict[str, Any] = {
            "pixel_values": batch_images,
            "target_boxes": torch.tensor(target_boxes, dtype=torch.float32),
            "target_classes": torch.tensor(target_classes, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.float32),
            "text": [
                self._normalize_prompt(sample.metadata.get("prompt", self.prompt))
                for sample in samples
            ],
            "images": batch_size,
            "image_text_pairs": batch_size,
            "patch_tokens": int(batch_size * (self.image_size // 16) ** 2),
            "tokens": int(batch_size * max(1, len(self.prompt.split()))),
            "num_images": batch_size,
            "num_pairs": batch_size,
        }
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch.update(dict(encoded))
            batch["tokens"] = int(batch.get("attention_mask").sum().item())
        if self.loss_type == "official":
            batch["labels"] = [
                {
                    "class_labels": torch.tensor(
                        [
                            int(row.get("cls", 0))
                            for row in (sample.target or [])
                        ],
                        dtype=torch.long,
                    ),
                    "boxes": torch.tensor(
                        [row["bbox"] for row in (sample.target or [])],
                        dtype=torch.float32,
                    ).reshape(-1, 4),
                }
                for sample in samples
            ]
        return batch

    @staticmethod
    def _require_torch() -> Any:
        try:
            import torch

            return torch
        except Exception as exc:
            raise ImportError("GroundDinoBatchAdapter requires torch.") from exc

    def _normalize_prompt(self, prompt: str) -> str:
        text = str(prompt).strip()
        if self.loss_type != "official" or not text:
            return text
        if text.endswith((".", "。")):
            return text
        return f"{text}."


class GroundDinoPhraseTargetAdapter:
    """Parse phrase-grounding JSON annotations for HF GroundingDINO loss."""

    def cache_paths(self, sample: VisionSample) -> Sequence[Path | str]:
        if isinstance(sample.annotation, (str, Path)):
            return [sample.annotation]
        return []

    def cache_extra(self, transform: VisionTransformConfig) -> Mapping[str, Any]:
        return {"target_format": "ground_dino_phrase_v1"}

    def build_target(self, sample: VisionSample) -> List[Dict[str, Any]]:
        if sample.annotation is None:
            return []
        payload = (
            dict(sample.annotation)
            if isinstance(sample.annotation, Mapping)
            else json.loads(Path(sample.annotation).read_text(encoding="utf-8"))
        )
        prompt = str(payload.get("prompt", sample.text or "")).strip()
        if prompt:
            sample.metadata["prompt"] = prompt
        rows = payload.get("objects", payload.get("labels", []))
        targets: List[Dict[str, Any]] = []
        for row in rows:
            bbox = row.get("bbox", row.get("box"))
            if bbox is None:
                continue
            class_label = row.get("class_label", row.get("label_index", row.get("cls", 0)))
            targets.append(
                {
                    "cls": int(class_label),
                    "bbox": [float(value) for value in bbox],
                    "phrase": str(row.get("phrase", "")),
                }
            )
        return targets


def collect_detection_samples(
    *,
    data_dir: str | None,
    image_dir: str | None,
    label_dir: str | None,
    limit: int,
) -> List[Dict[str, Path]]:
    """Collect image/YOLO-label pairs from a cached detection dataset."""
    resolved_image_dir = Path(image_dir or Path(str(data_dir)) / "images")
    resolved_label_dir = Path(label_dir or Path(str(data_dir)) / "labels")
    if not resolved_image_dir.exists():
        raise ValueError(f"Detection image directory does not exist: {resolved_image_dir}")
    if not resolved_label_dir.exists():
        raise ValueError(f"Detection label directory does not exist: {resolved_label_dir}")

    image_paths = sorted(
        path
        for suffix in ("*.jpg", "*.jpeg", "*.png")
        for path in resolved_image_dir.glob(suffix)
    )
    samples: List[Dict[str, Path]] = []
    for image_path in image_paths:
        label_path = resolved_label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            samples.append({"image": image_path, "label": label_path})
    if not samples:
        raise ValueError(
            "Detection dataset has no image/label pairs under "
            f"{resolved_image_dir} and {resolved_label_dir}."
        )
    return samples[: max(1, int(limit))]


def collect_phrase_grounding_samples(
    *,
    data_dir: str | None,
    image_dir: str | None,
    annotation_dir: str | None,
    limit: int,
) -> List[Dict[str, Path]]:
    """Collect image/JSON-annotation pairs for phrase grounding."""
    resolved_image_dir = Path(image_dir or Path(str(data_dir)) / "images")
    resolved_annotation_dir = Path(annotation_dir or Path(str(data_dir)) / "annotations")
    if not resolved_image_dir.exists():
        raise ValueError(
            f"Phrase grounding image directory does not exist: {resolved_image_dir}"
        )
    if not resolved_annotation_dir.exists():
        raise ValueError(
            "Phrase grounding annotation directory does not exist: "
            f"{resolved_annotation_dir}"
        )
    image_paths = sorted(
        path
        for suffix in ("*.jpg", "*.jpeg", "*.png")
        for path in resolved_image_dir.glob(suffix)
    )
    samples: List[Dict[str, Path]] = []
    for image_path in image_paths:
        annotation_path = resolved_annotation_dir / f"{image_path.stem}.json"
        if annotation_path.exists():
            samples.append({"image": image_path, "label": annotation_path})
    if not samples:
        raise ValueError(
            "Phrase grounding dataset has no image/annotation pairs under "
            f"{resolved_image_dir} and {resolved_annotation_dir}."
        )
    return samples[: max(1, int(limit))]
