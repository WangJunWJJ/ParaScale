# -*- coding: utf-8 -*-
# @Time : 2026/6/25 下午4:11
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Inference workload adapters for vision and multimodal smoke."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from parascale.runtime.inference.postprocess import (
    DetectionPostprocessConfig,
    DetectionPostprocessor,
)


def build_inference_components(
    config_data: Dict[str, Any],
) -> Tuple[Any, List[Any], str]:
    inference = _section(config_data, "inference")
    workload = str(inference.get("workload", "clip_synthetic"))
    batch_size = int(inference.get("batch_size", 2) or 2)
    num_batches = int(inference.get("num_batches", 1) or 1)
    if workload in {"clip_synthetic", "clip", "clip_embedding"}:
        return (
            SyntheticClipInferenceModel(),
            _build_clip_batches(batch_size, num_batches, image_mode="numbers"),
            "multimodal_embedding",
        )
    if workload in {"clip_real", "clip_hf", "hf_clip"}:
        return (
            HFClipInferenceModel(_model_path(config_data)),
            _build_clip_batches(batch_size, num_batches, image_mode="pil"),
            "multimodal_embedding",
        )
    if workload in {"yolo_world_synthetic", "yolo_world", "yoloworld"}:
        return (
            SyntheticYoloWorldInferenceModel(),
            _build_yolo_batches(batch_size, num_batches, image_mode="numbers"),
            "vision_detection",
        )
    if workload in {"yolo_world_real", "yoloworld_real", "ultralytics_yolo_world"}:
        return (
            UltralyticsYoloWorldInferenceModel(
                _model_path(config_data),
                postprocess_config=_detection_postprocess_config(inference),
                use_ultralytics_postprocess=(
                    str(inference.get("postprocess_mode", "device")).lower()
                    == "ultralytics"
                ),
            ),
            _build_yolo_batches(batch_size, num_batches, image_mode="pil"),
            "vision_detection",
        )
    raise ValueError(f"unsupported inference workload: {workload}")


class SyntheticClipInferenceModel:
    def __init__(self) -> None:
        self.device = "cpu"

    def to(self, device: str):
        self.device = str(device)
        return self

    def eval(self):
        return self

    def embed(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        count = int(batch.get("num_pairs", len(batch.get("texts", []))) or 0)
        torch = _optional_torch()
        if torch is not None:
            base = torch.arange(count, dtype=torch.float32, device=self.device)
            embeddings_tensor = torch.stack([base, torch.ones_like(base)], dim=1)
            similarity = embeddings_tensor @ embeddings_tensor.T
            return {
                "image_embeddings": embeddings_tensor.detach().cpu().tolist(),
                "text_embeddings": embeddings_tensor.detach().cpu().tolist(),
                "similarity": similarity.detach().cpu().tolist(),
            }
        embeddings = [[float(index), 1.0] for index in range(count)]
        return {
            "image_embeddings": embeddings,
            "text_embeddings": embeddings,
            "similarity": [
                [1.0 if i == j else 0.0 for j in range(count)] for i in range(count)
            ],
        }


class SyntheticYoloWorldInferenceModel:
    def __init__(self) -> None:
        self.device = "cpu"

    def to(self, device: str):
        self.device = str(device)
        return self

    def eval(self):
        return self

    def detect(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        count = int(batch.get("num_images", len(batch.get("images", []))) or 0)
        torch = _optional_torch()
        if torch is not None:
            boxes = torch.tensor(
                [[0.1, 0.1, 0.9, 0.9] for _ in range(count)],
                dtype=torch.float32,
                device=self.device,
            )
            scores = torch.full((count,), 0.5, dtype=torch.float32, device=self.device)
            return {
                "boxes": [
                    {
                        "xyxy": boxes[index].detach().cpu().tolist(),
                        "score": float(scores[index].detach().cpu().item()),
                        "label": "object",
                        "image_index": index,
                    }
                    for index in range(count)
                ]
            }
        return {
            "boxes": [
                {
                    "xyxy": [0.1, 0.1, 0.9, 0.9],
                    "score": 0.5,
                    "label": "object",
                    "image_index": index,
                }
                for index in range(count)
            ]
        }


class HFClipInferenceModel:
    """Thin adapter for Hugging Face CLIP embedding inference."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.device = "cpu"
        self._model = None
        self._processor = None

    def to(self, device: str):
        self.device = str(device)
        self._load()
        self._model.to(self.device)
        return self

    def eval(self):
        self._load()
        self._model.eval()
        return self

    def embed(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self._load()
        torch = _require_torch()
        inputs = self._processor(
            text=batch.get("texts", []),
            images=batch.get("images", []),
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        logits = outputs.logits_per_image
        return {
            "image_embeddings_shape": list(outputs.image_embeds.shape),
            "text_embeddings_shape": list(outputs.text_embeds.shape),
            "logits_per_image_shape": list(logits.shape),
            "logits_per_image_mean": float(logits.detach().float().cpu().mean().item()),
        }

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:
            raise ImportError(
                "clip_real inference requires transformers with CLIPModel and "
                "CLIPProcessor installed."
            ) from exc
        self._processor = CLIPProcessor.from_pretrained(self.model_path)
        self._model = CLIPModel.from_pretrained(self.model_path)


class UltralyticsYoloWorldInferenceModel:
    """Thin adapter for Ultralytics YOLO-World detection inference."""

    def __init__(
        self,
        model_path: str,
        *,
        postprocess_config: DetectionPostprocessConfig | None = None,
        use_ultralytics_postprocess: bool = False,
    ) -> None:
        self.model_path = model_path
        self.device = "cpu"
        self._model = None
        self.postprocess_config = postprocess_config or DetectionPostprocessConfig()
        self.use_ultralytics_postprocess = bool(use_ultralytics_postprocess)

    def to(self, device: str):
        self.device = str(device)
        self._load()
        model = getattr(self._model, "model", None)
        to_device = getattr(model, "to", None)
        if callable(to_device):
            to_device(self.device)
        return self

    def eval(self):
        self._load()
        model = getattr(self._model, "model", None)
        eval_fn = getattr(model, "eval", None)
        if callable(eval_fn):
            eval_fn()
        return self

    def detect(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self._load()
        if not self.use_ultralytics_postprocess:
            return self._detect_with_parascale_postprocess(batch)
        results = self._model.predict(
            source=batch.get("images", []),
            device=self.device,
            verbose=False,
        )
        boxes = []
        for image_index, result in enumerate(results):
            result_boxes = getattr(result, "boxes", None)
            if result_boxes is None:
                continue
            xyxy = getattr(result_boxes, "xyxy", [])
            conf = getattr(result_boxes, "conf", [])
            cls = getattr(result_boxes, "cls", [])
            for box_index in range(len(xyxy)):
                boxes.append(
                    {
                        "image_index": image_index,
                        "xyxy": _tensor_row_to_list(xyxy, box_index),
                        "score": _tensor_item_to_float(conf, box_index),
                        "class_id": _tensor_item_to_float(cls, box_index),
                    }
                )
        return {"boxes": boxes, "num_boxes": len(boxes)}

    def _detect_with_parascale_postprocess(
        self, batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        torch = _require_torch()
        images = _images_to_tensor(batch.get("images", []), device=self.device)
        model = getattr(self._model, "model", None)
        if model is None:
            raise RuntimeError("Ultralytics model does not expose a raw model module.")
        with torch.no_grad():
            prediction = model(images)
        result = DetectionPostprocessor(self.postprocess_config).from_yolo_prediction(
            prediction
        )
        result["postprocess"]["path"] = "parascale"
        return result

    def _load(self) -> None:
        if self._model is not None:
            return
        _ensure_ultralytics_config_dir()
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise ImportError(
                "yolo_world_real inference requires ultralytics installed."
            ) from exc
        self._model = YOLO(self.model_path)


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def _model_path(config_data: Dict[str, Any]) -> str:
    inference = _section(config_data, "inference")
    model = _section(config_data, "model")
    value = (
        inference.get("model_path")
        or model.get("path")
        or model.get("model_path")
        or model.get("model_id")
    )
    if not value:
        raise ValueError(
            "real inference workloads require model.path or inference.model_path."
        )
    path = Path(str(value)).expanduser()
    return str(path)


def _detection_postprocess_config(
    inference: Dict[str, Any],
) -> DetectionPostprocessConfig:
    mode = str(inference.get("postprocess_mode", "device") or "device").lower()
    if mode == "ultralytics":
        mode = "device"
    return DetectionPostprocessConfig(
        mode=mode,
        confidence_threshold=float(inference.get("confidence_threshold", 0.25) or 0.25),
        iou_threshold=float(inference.get("iou_threshold", 0.45) or 0.45),
        max_detections=int(inference.get("max_detections", 300) or 300),
    )


def _ensure_ultralytics_config_dir() -> str:
    value = os.environ.get("YOLO_CONFIG_DIR")
    if value:
        return value
    path = Path(tempfile.gettempdir()) / "parascale_ultralytics"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(path)
    return str(path)


def _build_clip_batches(
    batch_size: int, num_batches: int, *, image_mode: str
) -> List[Dict[str, Any]]:
    return [
        {
            "images": _build_images(batch_size, image_mode=image_mode),
            "texts": [f"a photo of object {index}" for index in range(batch_size)],
            "num_images": batch_size,
            "num_pairs": batch_size,
        }
        for _ in range(num_batches)
    ]


def _build_yolo_batches(
    batch_size: int, num_batches: int, *, image_mode: str
) -> List[Dict[str, Any]]:
    return [
        {
            "images": _build_images(batch_size, image_mode=image_mode),
            "prompts": ["person", "car", "chair"][: max(1, min(batch_size, 3))],
            "num_images": batch_size,
        }
        for _ in range(num_batches)
    ]


def _build_images(batch_size: int, *, image_mode: str) -> List[Any]:
    if image_mode == "numbers":
        return [[float(index)] for index in range(batch_size)]
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise ImportError(
            "real inference image batches require Pillow installed."
        ) from exc
    images = []
    for index in range(batch_size):
        image = Image.new(
            "RGB",
            (224, 224),
            color=((index * 47) % 255, (index * 83) % 255, (index * 19) % 255),
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle((48, 48, 176, 176), outline=(255, 255, 255), width=4)
        images.append(image)
    return images


def _images_to_tensor(images: List[Any], *, device: str) -> Any:
    torch = _require_torch()
    tensors = []
    for image in images:
        if hasattr(image, "resize"):
            image = image.convert("RGB").resize((640, 640))
            data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            tensor = data.view(640, 640, 3).permute(2, 0, 1).float() / 255.0
        else:
            tensor = torch.as_tensor(image, dtype=torch.float32)
            if tensor.ndim == 1:
                tensor = tensor.view(1, 1, -1).expand(3, 640, 640)
        tensors.append(tensor)
    if not tensors:
        return torch.empty((0, 3, 640, 640), dtype=torch.float32, device=device)
    return torch.stack(tensors, dim=0).to(device)


def _optional_torch() -> Any:
    try:
        import torch
    except Exception:
        return None
    return torch


def _require_torch() -> Any:
    torch = _optional_torch()
    if torch is None:
        raise ImportError("real inference workloads require torch installed.")
    return torch


def _tensor_row_to_list(value: Any, index: int) -> List[float]:
    row = value[index]
    detach = getattr(row, "detach", None)
    if callable(detach):
        row = row.detach().cpu()
    tolist = getattr(row, "tolist", None)
    if callable(tolist):
        return [float(item) for item in tolist()]
    return [float(item) for item in row]


def _tensor_item_to_float(value: Any, index: int) -> float:
    item = value[index]
    detach = getattr(item, "detach", None)
    if callable(detach):
        item = item.detach().cpu()
    item_fn = getattr(item, "item", None)
    if callable(item_fn):
        return float(item_fn())
    return float(item)


__all__ = [
    "HFClipInferenceModel",
    "SyntheticClipInferenceModel",
    "SyntheticYoloWorldInferenceModel",
    "UltralyticsYoloWorldInferenceModel",
    "build_inference_components",
]
