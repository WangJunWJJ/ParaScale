# -*- coding: utf-8 -*-
# @Time : 2026/6/18 下午4:09
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""YOLO-World workload builders and official detection-loss adapter."""

from __future__ import annotations

import io
import os
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

from parascale.data.vision.batch import VisionBatchCollator
from parascale.data.vision.preprocessor import (
    VisionPreprocessor,
    VisionSample,
    VisionTransformConfig,
)
from parascale.workloads.adapters.yolo import (
    YoloDetectionTargetAdapter,
    YoloOfficialBatchAdapter,
)
from parascale.workloads.specs.yolo import YoloWorldSpec

from .common import _require_torch, _select_torch_device


def build_yolo_world_components(spec: YoloWorldSpec):
    torch = _require_torch()
    import torch.nn as nn
    import torch.optim as optim

    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
    try:
        import numpy as np
        from ultralytics import YOLOWorld
    except Exception as exc:
        raise ImportError(
            "YOLO-World workloads require ultralytics, Pillow and numpy."
        ) from exc

    torch.manual_seed(spec.seed)
    device = _select_torch_device(torch, spec.device)
    model_path = _resolve_model_path(spec.model_path)
    yolo = YOLOWorld(model_path)
    base_model = yolo.model.train()
    if spec.loss_type == "official":
        _prepare_yolo_world_official_loss_args(base_model)
    for parameter in base_model.parameters():
        parameter.requires_grad_(True)

    class YoloWorldTrainingProxy(nn.Module):
        def __init__(self, wrapped_model) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(self, pixel_values=None, **kwargs):
            if kwargs.get("official_loss"):
                return self._forward_official_loss(pixel_values=pixel_values, **kwargs)
            images = pixel_values if pixel_values is not None else kwargs["images"]
            try:
                parameter = next(self.wrapped_model.parameters())
                images = images.to(dtype=parameter.dtype)
            except StopIteration:
                pass
            return self.wrapped_model.predict(images)

        def _forward_official_loss(self, pixel_values=None, **kwargs):
            images = kwargs.get("img", pixel_values)
            if images is None:
                raise RuntimeError("YOLO official loss batches require an img tensor.")
            try:
                parameter = next(self.wrapped_model.parameters())
                target_dtype = parameter.dtype
                target_device = parameter.device
            except StopIteration:
                target_dtype = images.dtype
                target_device = images.device
            model_batch = {
                "img": images.to(
                    device=target_device, dtype=target_dtype, non_blocking=True
                ),
                "cls": kwargs["cls"].to(
                    device=target_device, dtype=torch.float32, non_blocking=True
                ),
                "bboxes": kwargs["bboxes"].to(
                    device=target_device, dtype=torch.float32, non_blocking=True
                ),
                "batch_idx": kwargs["batch_idx"].to(
                    device=target_device, dtype=torch.float32, non_blocking=True
                ),
            }
            text_features = getattr(self.wrapped_model, "txt_feats", None)
            if text_features is not None:
                model_batch["txt_feats"] = text_features.to(
                    device=model_batch["img"].device,
                    dtype=target_dtype,
                )
            return self.wrapped_model(model_batch)

    model = YoloWorldTrainingProxy(base_model)
    optimizer = optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=spec.lr,
    )

    if spec.loss_type == "official":
        samples = _yolo_cached_detection_samples(spec)
        dataset = _YoloOfficialDataset(samples)
        collator = _YoloOfficialCollator(
            image_size=spec.image_size,
            tensor_cache_dir=spec.tensor_cache_dir,
            tensor_cache=spec.tensor_cache,
        )

        def dataloader() -> Iterable[Dict[str, Any]]:
            yield from _iter_yolo_official_batches(
                dataset=dataset,
                samples=samples,
                collator=collator,
                spec=spec,
                device=device,
            )

    else:
        if spec.data_zip is None:
            raise ValueError("YOLO proxy loss requires data.zip_path.")
        image_names = _yolo_zip_image_names(spec.data_zip, spec.num_samples)

        def dataloader() -> Iterable[Dict[str, Any]]:
            yielded = 0
            index = 0
            with zipfile.ZipFile(spec.data_zip) as archive:
                while yielded < spec.num_batches:
                    names = [
                        image_names[(index + offset) % len(image_names)]
                        for offset in range(spec.batch_size)
                    ]
                    index += spec.batch_size
                    images = [
                        _read_yolo_zip_image(
                            archive,
                            name,
                            image_size=spec.image_size,
                            np=np,
                            torch=torch,
                        )
                        for name in names
                    ]
                    batch = torch.stack(images, dim=0)
                    yield {
                        "pixel_values": batch,
                        "images": batch.shape[0],
                        "image_text_pairs": batch.shape[0],
                        "patch_tokens": int(
                            batch.shape[0] * (spec.image_size // 32) ** 2
                        ),
                        "tokens": int(batch.shape[0]),
                        "num_images": batch.shape[0],
                        "num_pairs": batch.shape[0],
                    }
                    yielded += 1

    def loss_fn(output, batch):
        if batch.get("official_loss"):
            loss = output[0] if isinstance(output, (list, tuple)) else output
            if torch.is_tensor(loss):
                return loss.sum() if loss.ndim > 0 else loss
            if isinstance(loss, (list, tuple)):
                terms = [item.sum() for item in loss if torch.is_tensor(item)]
                if terms:
                    return sum(terms)
            raise RuntimeError("YOLO official loss did not return a tensor loss.")
        terms = []

        def collect(value):
            if torch.is_tensor(value) and value.is_floating_point():
                terms.append(value.float().square().mean())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        collect(output)
        if not terms:
            raise RuntimeError("YOLO-World output did not contain floating tensors.")
        return sum(terms) / float(len(terms))

    return model, optimizer, dataloader(), loss_fn


class _YoloOfficialDataset:
    def __init__(self, samples: List[Dict[str, Path]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> VisionSample:
        sample = self.samples[index % len(self.samples)]
        return _to_vision_sample(sample)


class _YoloOfficialCollator:
    def __init__(
        self,
        *,
        image_size: int,
        tensor_cache_dir: str | None = None,
        tensor_cache: bool = False,
    ) -> None:
        self.image_size = int(image_size)
        self.tensor_cache_dir = tensor_cache_dir
        self.tensor_cache = bool(tensor_cache)

    def __call__(self, samples: List[Dict[str, Path] | VisionSample]) -> Dict[str, Any]:
        vision_samples = [
            sample if isinstance(sample, VisionSample) else _to_vision_sample(sample)
            for sample in samples
        ]
        collator = VisionBatchCollator(
            preprocessor=VisionPreprocessor(
                transform=VisionTransformConfig(
                    image_size=self.image_size,
                    cache_format="vision_tensor_yolo_official_v1",
                ),
                target_adapter=YoloDetectionTargetAdapter(),
                tensor_cache_dir=self.tensor_cache_dir,
                tensor_cache=self.tensor_cache,
            ),
            batch_adapter=YoloOfficialBatchAdapter(image_size=self.image_size),
        )
        return collator(vision_samples)


def _iter_yolo_official_batches(
    *,
    dataset: Any,
    samples: Sequence[Any],
    collator: Any,
    spec: Any,
    device: Any,
    dataloader_cls: Any | None = None,
) -> Iterable[Dict[str, Any]]:
    if dataloader_cls is None:
        try:
            from torch.utils.data import DataLoader
        except (ImportError, ModuleNotFoundError):
            warnings.warn(
                "torch DataLoader is unavailable; falling back to synchronous "
                "YOLO batching.",
                RuntimeWarning,
                stacklevel=2,
            )
            yield from _iter_yolo_sync_batches(samples, collator, spec)
            return
        dataloader_cls = DataLoader

    kwargs: Dict[str, Any] = {
        "batch_size": spec.batch_size,
        "num_workers": max(0, spec.num_workers),
        "pin_memory": bool(spec.pin_memory and device.type == "cuda"),
        "collate_fn": collator,
    }
    if spec.num_workers > 0:
        kwargs["prefetch_factor"] = max(1, spec.prefetch_factor)
        kwargs["persistent_workers"] = bool(spec.persistent_workers)

    try:
        loader = dataloader_cls(dataset, **kwargs)
        yielded = 0
        while yielded < spec.num_batches:
            for batch in loader:
                yield batch
                yielded += 1
                if yielded >= spec.num_batches:
                    return
    except Exception:
        warnings.warn(
            "YOLO DataLoader failed; refusing to silently fall back to synchronous "
            "batching because benchmark results would no longer describe the "
            "requested execution path.",
            RuntimeWarning,
            stacklevel=2,
        )
        raise


def _iter_yolo_sync_batches(
    samples: Sequence[Any],
    collator: Any,
    spec: Any,
) -> Iterable[Dict[str, Any]]:
    yielded = 0
    index = 0
    while yielded < spec.num_batches:
        batch_samples = [
            samples[(index + offset) % len(samples)]
            for offset in range(spec.batch_size)
        ]
        index += spec.batch_size
        yield collator(batch_samples)
        yielded += 1


def _to_vision_sample(sample: Dict[str, Path]) -> VisionSample:
    image_path = Path(sample["image"])
    return VisionSample(
        image=image_path,
        annotation=sample["label"],
        sample_id=image_path.stem,
    )


def _resolve_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if path.exists():
        return str(path)

    search_dirs: List[Path] = []
    env_dirs = os.environ.get("PARASCALE_MODEL_DIRS", "")
    for item in env_dirs.split(os.pathsep):
        if item.strip():
            search_dirs.append(Path(item.strip()).expanduser())
    search_dirs.extend(
        [Path("/models"), Path("/yolo_models"), Path("/workspace/models")]
    )

    for directory in search_dirs:
        candidate = directory / path.name
        if candidate.exists():
            return str(candidate)

    searched = ", ".join(str(directory) for directory in search_dirs)
    raise FileNotFoundError(
        f"Model file '{model_path}' was not found. Put '{path.name}' in one of "
        f"the configured offline model directories or set PARASCALE_MODEL_DIRS. "
        f"Searched: {searched}"
    )


def _prepare_yolo_world_official_loss_args(base_model: Any) -> None:
    args = getattr(base_model, "args", None)
    if isinstance(args, dict):
        values = dict(args)
    elif args is None:
        values = {}
    else:
        values = dict(getattr(args, "__dict__", {}) or {})
    values.setdefault("box", 7.5)
    values.setdefault("cls", 0.5)
    values.setdefault("dfl", 1.5)
    base_model.args = SimpleNamespace(**values)
    if hasattr(base_model, "init_criterion"):
        base_model.criterion = base_model.init_criterion()


def _yolo_cached_detection_samples(spec: YoloWorldSpec) -> List[Dict[str, Path]]:
    image_dir = Path(spec.image_dir or Path(str(spec.data_dir)) / "images")
    label_dir = Path(spec.label_dir or Path(str(spec.data_dir)) / "labels")
    if not image_dir.exists():
        raise ValueError(f"YOLO cached image directory does not exist: {image_dir}")
    if not label_dir.exists():
        raise ValueError(f"YOLO cached label directory does not exist: {label_dir}")
    image_paths = sorted(
        path
        for suffix in ("*.jpg", "*.jpeg", "*.png")
        for path in image_dir.glob(suffix)
    )
    samples: List[Dict[str, Path]] = []
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            samples.append({"image": image_path, "label": label_path})
    rank = int(os.environ.get("RANK", "0") or 0)
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1") or 1))
    if world_size > 1 and len(samples) >= world_size:
        samples = [
            sample
            for sample_index, sample in enumerate(samples)
            if sample_index % world_size == rank
        ]
    samples = samples[: spec.num_samples]
    if not samples:
        raise ValueError(
            f"YOLO cached dataset has no image/label pairs under {image_dir} and {label_dir}."
        )
    return samples


def _yolo_zip_image_names(zip_path: str, limit: int) -> List[str]:
    with zipfile.ZipFile(zip_path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    if not names:
        raise ValueError(f"YOLO-World data zip contains no images: {zip_path}")
    return names[: max(1, min(int(limit), len(names)))]


def _read_yolo_zip_image(archive, name: str, *, image_size: int, np: Any, torch: Any):
    from PIL import Image

    image = Image.open(io.BytesIO(archive.read(name))).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype="uint8").copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)
