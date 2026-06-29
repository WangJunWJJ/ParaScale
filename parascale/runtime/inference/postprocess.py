# -*- coding: utf-8 -*-
# @Time : 2026/6/25 下午5:04
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Detection postprocessing utilities for inference runtimes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class DetectionPostprocessConfig:
    mode: str = "device"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300


class DetectionPostprocessor:
    """Generic detection postprocessor with device and async CPU paths."""

    def __init__(self, config: DetectionPostprocessConfig | None = None) -> None:
        self.config = config or DetectionPostprocessConfig()

    def from_boxes_scores(
        self,
        boxes: Any,
        scores: Any,
        class_ids: Any,
        *,
        image_index: int = 0,
    ) -> Dict[str, Any]:
        if self.config.mode == "async_cpu":
            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(
                    self._from_boxes_scores_python,
                    _to_python_rows(boxes),
                    _to_python_values(scores),
                    _to_python_values(class_ids),
                    image_index,
                    "async_cpu",
                ).result()
        if _looks_like_torch_tensor(boxes):
            return self._from_boxes_scores_torch(boxes, scores, class_ids, image_index)
        return self._from_boxes_scores_python(
            _to_python_rows(boxes),
            _to_python_values(scores),
            _to_python_values(class_ids),
            image_index,
            "python",
        )

    def from_yolo_prediction(self, prediction: Any) -> Dict[str, Any]:
        tensor = _first_prediction_tensor(prediction)
        if not _looks_like_torch_tensor(tensor):
            raise TypeError("YOLO prediction must be a torch-like tensor.")
        if len(tensor.shape) != 3:
            raise ValueError(
                f"expected YOLO prediction with 3 dimensions, got {tensor.shape}"
            )
        if int(tensor.shape[1]) < int(tensor.shape[2]):
            tensor = tensor.permute(0, 2, 1)
        merged: List[Dict[str, Any]] = []
        for image_index in range(int(tensor.shape[0])):
            per_image = tensor[image_index]
            boxes_xywh = per_image[:, :4]
            class_scores = per_image[:, 4:]
            scores, class_ids = class_scores.max(dim=1)
            boxes = _xywh_to_xyxy(boxes_xywh)
            result = self.from_boxes_scores(
                boxes,
                scores,
                class_ids,
                image_index=image_index,
            )
            merged.extend(result["boxes"])
        return {
            "boxes": merged,
            "num_boxes": len(merged),
            "postprocess": self._metadata(
                (
                    "torch"
                    if _looks_like_torch_tensor(tensor)
                    and self.config.mode != "async_cpu"
                    else self.config.mode
                ),
                (
                    "torch"
                    if _looks_like_torch_tensor(tensor)
                    and self.config.mode != "async_cpu"
                    else "python"
                ),
            ),
        }

    def _from_boxes_scores_torch(
        self,
        boxes: Any,
        scores: Any,
        class_ids: Any,
        image_index: int,
    ) -> Dict[str, Any]:
        keep = scores >= float(self.config.confidence_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        if int(scores.numel()) == 0:
            return {
                "boxes": [],
                "num_boxes": 0,
                "postprocess": self._metadata("torch", "torch"),
            }
        keep_indices = _class_aware_torch_nms(
            boxes,
            scores,
            class_ids,
            float(self.config.iou_threshold),
            int(self.config.max_detections),
        )
        boxes = boxes[keep_indices].detach().float().cpu()
        scores = scores[keep_indices].detach().float().cpu()
        class_ids = class_ids[keep_indices].detach().float().cpu()
        return self._from_boxes_scores_python(
            boxes.tolist(),
            scores.tolist(),
            class_ids.tolist(),
            image_index,
            "torch",
            nms_backend="torch",
        )

    def _from_boxes_scores_python(
        self,
        boxes: Sequence[Sequence[float]],
        scores: Sequence[float],
        class_ids: Sequence[float],
        image_index: int,
        backend: str,
        *,
        nms_backend: str = "python",
    ) -> Dict[str, Any]:
        filtered = [
            (box, float(score), int(class_id))
            for box, score, class_id in zip(boxes, scores, class_ids)
            if float(score) >= float(self.config.confidence_threshold)
        ]
        keep = _class_aware_python_nms(
            filtered,
            iou_threshold=float(self.config.iou_threshold),
            max_detections=int(self.config.max_detections),
        )
        output = [
            {
                "image_index": int(image_index),
                "xyxy": [float(value) for value in box],
                "score": float(score),
                "class_id": int(class_id),
            }
            for box, score, class_id in keep
        ]
        return {
            "boxes": output,
            "num_boxes": len(output),
            "postprocess": self._metadata(backend, nms_backend),
        }

    def _metadata(self, backend: str, nms_backend: str) -> Dict[str, Any]:
        return {
            "backend": backend,
            "nms_backend": nms_backend,
            "confidence_threshold": float(self.config.confidence_threshold),
            "iou_threshold": float(self.config.iou_threshold),
            "max_detections": int(self.config.max_detections),
        }


def _class_aware_torch_nms(
    boxes: Any,
    scores: Any,
    class_ids: Any,
    iou_threshold: float,
    max_detections: int,
) -> Any:
    keep_all = []
    for class_id in class_ids.unique():
        indices = (class_ids == class_id).nonzero(as_tuple=False).flatten()
        selected = _torch_nms(boxes[indices], scores[indices], iou_threshold)
        keep_all.append(indices[selected])
    if not keep_all:
        return scores.new_empty((0,), dtype=_torch_long_dtype(scores))
    keep = _torch_cat(keep_all)
    order = scores[keep].argsort(descending=True)
    return keep[order[:max_detections]]


def _torch_nms(boxes: Any, scores: Any, iou_threshold: float) -> Any:
    order = scores.argsort(descending=True)
    keep = []
    while int(order.numel()) > 0:
        current = order[0]
        keep.append(current)
        if int(order.numel()) == 1:
            break
        rest = order[1:]
        iou = _torch_iou_one_to_many(boxes[current], boxes[rest])
        order = rest[iou <= iou_threshold]
    if not keep:
        return scores.new_empty((0,), dtype=_torch_long_dtype(scores))
    return _torch_stack(keep).to(dtype=_torch_long_dtype(scores))


def _torch_iou_one_to_many(box: Any, boxes: Any) -> Any:
    x1 = _torch_max(box[0], boxes[:, 0])
    y1 = _torch_max(box[1], boxes[:, 1])
    x2 = _torch_min(box[2], boxes[:, 2])
    y2 = _torch_min(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    box_area = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)
    return inter / (box_area + boxes_area - inter).clamp(min=1e-7)


def _class_aware_python_nms(
    detections: Sequence[tuple[Sequence[float], float, int]],
    *,
    iou_threshold: float,
    max_detections: int,
) -> List[tuple[Sequence[float], float, int]]:
    selected: List[tuple[Sequence[float], float, int]] = []
    by_class: Dict[int, List[tuple[Sequence[float], float, int]]] = {}
    for detection in detections:
        by_class.setdefault(int(detection[2]), []).append(detection)
    for class_detections in by_class.values():
        remaining = sorted(class_detections, key=lambda item: item[1], reverse=True)
        while remaining:
            current = remaining.pop(0)
            selected.append(current)
            remaining = [
                item
                for item in remaining
                if _python_iou(current[0], item[0]) <= iou_threshold
            ]
    return sorted(selected, key=lambda item: item[1], reverse=True)[:max_detections]


def _python_iou(box: Sequence[float], other: Sequence[float]) -> float:
    x1 = max(float(box[0]), float(other[0]))
    y1 = max(float(box[1]), float(other[1]))
    x2 = min(float(box[2]), float(other[2]))
    y2 = min(float(box[3]), float(other[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )
    other_area = max(0.0, float(other[2]) - float(other[0])) * max(
        0.0, float(other[3]) - float(other[1])
    )
    return inter / max(area + other_area - inter, 1e-7)


def _xywh_to_xyxy(boxes: Any) -> Any:
    half = boxes[:, 2:4] / 2.0
    top_left = boxes[:, 0:2] - half
    bottom_right = boxes[:, 0:2] + half
    return _torch_cat([top_left, bottom_right], dim=1)


def _first_prediction_tensor(prediction: Any) -> Any:
    if isinstance(prediction, (list, tuple)):
        return prediction[0]
    return prediction


def _looks_like_torch_tensor(value: Any) -> bool:
    return (
        hasattr(value, "detach")
        and hasattr(value, "shape")
        and hasattr(value, "device")
    )


def _to_python_rows(value: Any) -> List[List[float]]:
    if _looks_like_torch_tensor(value):
        value = value.detach().float().cpu().tolist()
    return [[float(item) for item in row] for row in value]


def _to_python_values(value: Any) -> List[float]:
    if _looks_like_torch_tensor(value):
        value = value.detach().float().cpu().tolist()
    return [float(item) for item in value]


def _torch_cat(values: Iterable[Any], dim: int = 0) -> Any:
    items = list(values)
    first = items[0]
    torch = _torch_from_tensor(first)
    return torch.cat(items, dim=dim)


def _torch_stack(values: Sequence[Any]) -> Any:
    return _torch_from_tensor(values[0]).stack(list(values))


def _torch_max(left: Any, right: Any) -> Any:
    return _torch_from_tensor(right).maximum(left, right)


def _torch_min(left: Any, right: Any) -> Any:
    return _torch_from_tensor(right).minimum(left, right)


def _torch_long_dtype(tensor: Any) -> Any:
    return _torch_from_tensor(tensor).long


def _torch_from_tensor(tensor: Any) -> Any:
    module = type(tensor).__module__.split(".", 1)[0]
    if module != "torch":
        import torch

        return torch
    import torch

    return torch


__all__ = ["DetectionPostprocessConfig", "DetectionPostprocessor"]
