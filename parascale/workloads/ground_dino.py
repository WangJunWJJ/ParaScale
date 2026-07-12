# -*- coding: utf-8 -*-
# @Time : 2026/7/11 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""GroundingDINO workload builder."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from parascale.data.vision.batch import VisionBatchCollator
from parascale.data.vision.preprocessor import (
    VisionPreprocessor,
    VisionSample,
    VisionTransformConfig,
)
from parascale.runtime.specs import GroundDinoSpec
from parascale.workloads.adapters.ground_dino import (
    GroundDinoBatchAdapter,
    GroundDinoPhraseTargetAdapter,
    collect_detection_samples,
    collect_phrase_grounding_samples,
)
from parascale.workloads.adapters.yolo import YoloDetectionTargetAdapter

from .common import _require_torch


def build_ground_dino_components(spec: GroundDinoSpec):
    """Build a GroundingDINO training workload from cached detection data."""
    torch = _require_torch()
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(spec.seed)
    resolved_model_path = _resolve_model_path(spec.model_path)
    hf_model = _load_ground_dino_model(resolved_model_path)
    tokenizer = _load_ground_dino_tokenizer(resolved_model_path)
    model = _ground_dino_module_proxy(nn)(hf_model)
    optimizer = optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=spec.lr,
    )
    if spec.data_type in {"phrase_grounding", "grounding_phrase", "ground_dino_json"}:
        samples = collect_phrase_grounding_samples(
            data_dir=spec.data_dir,
            image_dir=spec.image_dir,
            annotation_dir=spec.annotation_dir,
            limit=spec.num_samples,
        )
        target_adapter = GroundDinoPhraseTargetAdapter()
        cache_format = "vision_tensor_ground_dino_phrase_v1"
    else:
        samples = collect_detection_samples(
            data_dir=spec.data_dir,
            image_dir=spec.image_dir,
            label_dir=spec.label_dir,
            limit=spec.num_samples,
        )
        target_adapter = YoloDetectionTargetAdapter()
        cache_format = "vision_tensor_ground_dino_proxy_v1"
    dataset = _GroundDinoDataset(samples, prompt=spec.prompt)
    collator = VisionBatchCollator(
        preprocessor=VisionPreprocessor(
            transform=VisionTransformConfig(
                image_size=spec.image_size,
                cache_format=cache_format,
            ),
            target_adapter=target_adapter,
            tensor_cache_dir=spec.tensor_cache_dir,
            tensor_cache=spec.tensor_cache,
        ),
        batch_adapter=GroundDinoBatchAdapter(
            prompt=spec.prompt,
            image_size=spec.image_size,
            loss_type=spec.loss_type,
            tokenizer=tokenizer,
        ),
    )

    def dataloader() -> Iterable[Dict[str, Any]]:
        yield from _iter_ground_dino_batches(
            dataset=dataset,
            samples=samples,
            collator=collator,
            spec=spec,
        )

    def loss_fn(output, batch):
        return output["loss"] if isinstance(output, dict) and "loss" in output else output

    return model, optimizer, dataloader(), loss_fn


class GroundDinoTrainingProxy:
    """Wrap Hugging Face GroundingDINO with a compact trainable loss."""

    def __init__(self, wrapped_model: Any) -> None:
        self.wrapped_model = wrapped_model

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped_model, name)

    def to(self, *args: Any, **kwargs: Any) -> "GroundDinoTrainingProxy":
        self.wrapped_model.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True) -> "GroundDinoTrainingProxy":
        self.wrapped_model.train(mode)
        return self

    def eval(self) -> "GroundDinoTrainingProxy":
        self.wrapped_model.eval()
        return self

    def parameters(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_model.parameters(*args, **kwargs)

    def named_parameters(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_model.named_parameters(*args, **kwargs)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_model.load_state_dict(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(
        self,
        pixel_values=None,
        target_boxes=None,
        target_mask=None,
        text=None,
        labels=None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        torch = _require_torch()
        if pixel_values is None:
            pixel_values = kwargs["pixel_values"]
        model_kwargs = {"pixel_values": pixel_values}
        for key in ("input_ids", "attention_mask", "token_type_ids", "pixel_mask"):
            if key in kwargs and kwargs[key] is not None:
                model_kwargs[key] = kwargs[key]
        if labels is not None:
            model_kwargs["labels"] = GroundDinoTrainingProxy._move_labels(
                labels, pixel_values.device
            )
        outputs = self.wrapped_model(**model_kwargs)
        loss = getattr(outputs, "loss", None)
        loss_dict = getattr(outputs, "loss_dict", None)
        if loss is None and isinstance(outputs, dict):
            loss = outputs.get("loss")
            loss_dict = outputs.get("loss_dict")
        if loss is not None:
            payload = {"loss": loss}
            if loss_dict is not None:
                payload["loss_dict"] = loss_dict
            return payload
        logits = getattr(outputs, "logits", None)
        pred_boxes = getattr(outputs, "pred_boxes", None)
        if logits is None and isinstance(outputs, dict):
            logits = outputs.get("logits")
            pred_boxes = outputs.get("pred_boxes")
        if logits is None or pred_boxes is None:
            loss = self._fallback_tensor_loss(torch, outputs)
            return {"loss": loss}

        objectness = logits.float().sigmoid().amax(dim=-1)
        best_index = objectness.argmax(dim=1)
        batch_index = torch.arange(pred_boxes.shape[0], device=pred_boxes.device)
        selected_boxes = pred_boxes[batch_index, best_index].float()
        selected_scores = objectness[batch_index, best_index].float()
        target_boxes = target_boxes.to(
            device=selected_boxes.device,
            dtype=selected_boxes.dtype,
            non_blocking=True,
        )
        target_mask = target_mask.to(
            device=selected_scores.device,
            dtype=selected_scores.dtype,
            non_blocking=True,
        )
        bbox_loss = torch.nn.functional.l1_loss(
            selected_boxes,
            target_boxes,
            reduction="none",
        ).mean(dim=1)
        objectness_loss = torch.nn.functional.binary_cross_entropy(
            selected_scores.clamp(1e-4, 1 - 1e-4),
            target_mask.clamp(0, 1),
            reduction="none",
        )
        loss = (bbox_loss * target_mask + objectness_loss).mean()
        return {
            "loss": loss,
            "logits": logits,
            "pred_boxes": pred_boxes,
        }

    @staticmethod
    def _move_labels(labels: Any, device: Any) -> Any:
        moved = []
        for label in labels:
            moved.append(
                {
                    key: value.to(device=device, non_blocking=True)
                    if hasattr(value, "to")
                    else value
                    for key, value in dict(label).items()
                }
            )
        return moved

    @staticmethod
    def _fallback_tensor_loss(torch: Any, outputs: Any) -> Any:
        terms = []

        def collect(value: Any) -> None:
            if torch.is_tensor(value) and value.is_floating_point():
                terms.append(value.float().square().mean())
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        collect(outputs)
        if not terms:
            raise RuntimeError("GroundingDINO output did not contain trainable tensors.")
        return sum(terms) / float(len(terms))


def _ground_dino_module_proxy(nn: Any) -> Any:
    class GroundDinoTrainingModule(nn.Module):
        def __init__(self, wrapped_model: Any) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return GroundDinoTrainingProxy.forward(self, *args, **kwargs)

    return GroundDinoTrainingModule


class _GroundDinoDataset:
    def __init__(self, samples: List[Dict[str, Path]], *, prompt: str) -> None:
        self.samples = list(samples)
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> VisionSample:
        sample = self.samples[index % len(self.samples)]
        image_path = Path(sample["image"])
        return VisionSample(
            image=image_path,
            annotation=sample["label"],
            sample_id=image_path.stem,
            text=self.prompt,
        )


def _iter_ground_dino_batches(
    *,
    dataset: Any,
    samples: Sequence[Any],
    collator: Any,
    spec: GroundDinoSpec,
    dataloader_cls: Any | None = None,
) -> Iterable[Dict[str, Any]]:
    if dataloader_cls is None:
        try:
            from torch.utils.data import DataLoader
        except (ImportError, ModuleNotFoundError):
            warnings.warn(
                "torch DataLoader is unavailable; falling back to synchronous "
                "GroundingDINO batching.",
                RuntimeWarning,
                stacklevel=2,
            )
            yield from _iter_ground_dino_sync_batches(samples, collator, spec)
            return
        dataloader_cls = DataLoader

    kwargs: Dict[str, Any] = {
        "batch_size": spec.batch_size,
        "num_workers": max(0, spec.num_workers),
        "pin_memory": bool(spec.pin_memory),
        "collate_fn": collator,
    }
    if spec.num_workers > 0:
        kwargs["prefetch_factor"] = max(1, spec.prefetch_factor)
        kwargs["persistent_workers"] = bool(spec.persistent_workers)

    loader = dataloader_cls(dataset, **kwargs)
    yielded = 0
    while yielded < spec.num_batches:
        for batch in loader:
            yield batch
            yielded += 1
            if yielded >= spec.num_batches:
                return


def _iter_ground_dino_sync_batches(
    samples: Sequence[Any],
    collator: Any,
    spec: GroundDinoSpec,
) -> Iterable[Dict[str, Any]]:
    yielded = 0
    index = 0
    while yielded < spec.num_batches:
        batch_samples = [
            _sample_to_vision_sample(
                samples[(index + offset) % len(samples)],
                prompt=spec.prompt,
            )
            for offset in range(spec.batch_size)
        ]
        index += spec.batch_size
        yield collator(batch_samples)
        yielded += 1


def _sample_to_vision_sample(sample: Dict[str, Path], *, prompt: str) -> VisionSample:
    image_path = Path(sample["image"])
    return VisionSample(
        image=image_path,
        annotation=sample["label"],
        sample_id=image_path.stem,
        text=prompt,
    )


def _load_ground_dino_model(model_path: str) -> Any:
    try:
        from transformers import AutoModelForZeroShotObjectDetection
    except Exception as exc:
        raise ImportError(
            "GroundingDINO workloads require transformers. Install "
            "parascale[grounding-dino] or provide an environment with "
            "AutoModelForZeroShotObjectDetection."
        ) from exc

    return AutoModelForZeroShotObjectDetection.from_pretrained(model_path)


def _load_ground_dino_tokenizer(model_path: str) -> Any:
    try:
        from transformers import AutoProcessor
    except Exception:
        return None
    try:
        processor = AutoProcessor.from_pretrained(model_path)
    except Exception:
        return None
    return getattr(processor, "tokenizer", None)


def _resolve_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if path.exists():
        return str(path)
    if "/" in model_path and not path.exists():
        return model_path

    search_dirs: List[Path] = []
    env_dirs = os.environ.get("PARASCALE_MODEL_DIRS", "")
    for item in env_dirs.split(os.pathsep):
        if item.strip():
            search_dirs.append(Path(item.strip()).expanduser())
    search_dirs.extend(
        [Path("/models"), Path("/ground_dino_models"), Path("/workspace/models")]
    )
    for directory in search_dirs:
        candidate = directory / path.name
        if candidate.exists():
            return str(candidate)
    searched = ", ".join(str(directory) for directory in search_dirs)
    raise FileNotFoundError(
        f"GroundingDINO model '{model_path}' was not found. Set model.path to a "
        f"Hugging Face id or put the model under PARASCALE_MODEL_DIRS. "
        f"Searched: {searched}"
    )
