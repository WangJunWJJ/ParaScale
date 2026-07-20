# -*- coding: utf-8 -*-
# @Time : 2026/6/18 下午4:09
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Vision synthetic workload builders."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from parascale.data.vision import (
    PatchTokenBatchSampler,
    VisionCollator,
    estimate_patch_tokens,
)
from parascale.workloads.specs.vision import VisionSyntheticSpec

from .common import _patchify, _require_torch


def build_vision_synthetic_components(spec: VisionSyntheticSpec):
    torch = _require_torch()
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(spec.seed)

    class TinyPatchClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            patch_dim = spec.channels * spec.patch_size * spec.patch_size
            self.patch_embed = nn.Linear(patch_dim, spec.hidden_dim)
            self.norm = nn.LayerNorm(spec.hidden_dim)
            self.head = nn.Linear(spec.hidden_dim, spec.num_classes)

        def forward(self, pixel_values=None, **kwargs):
            x = pixel_values if pixel_values is not None else kwargs["x"]
            patches = _patchify(torch, x.float(), spec.patch_size)
            features = self.patch_embed(patches)
            pooled = self.norm(features).mean(dim=1)
            return self.head(pooled)

    model = TinyPatchClassifier()
    optimizer = optim.AdamW(model.parameters(), lr=spec.lr)
    dataset = _build_synthetic_vision_dataset(torch, spec)

    def dataloader() -> Iterable[Dict[str, Any]]:
        batch_indices = _vision_batch_indices(spec, dataset)
        yielded = 0
        for indices in batch_indices:
            samples = [dataset[index] for index in indices]
            batch = VisionCollator(patch_size=spec.patch_size)(samples)
            yield batch
            yielded += 1
            if yielded >= spec.num_batches:
                break

    def loss_fn(output, batch):
        return nn.functional.cross_entropy(output, batch["labels"])

    return model, optimizer, dataloader(), loss_fn


def _build_synthetic_vision_dataset(
    torch: Any, spec: VisionSyntheticSpec
) -> List[Dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(spec.seed)
    dataset: List[Dict[str, Any]] = []
    for index in range(spec.num_samples):
        image = torch.randn(
            spec.channels, spec.image_size, spec.image_size, generator=generator
        )
        label = int(index % spec.num_classes)
        dataset.append(
            {
                "pixel_values": image,
                "label": label,
                "height": spec.image_size,
                "width": spec.image_size,
                "patch_tokens": estimate_patch_tokens(
                    spec.image_size, spec.image_size, spec.patch_size
                ),
            }
        )
    return dataset


def _vision_batch_indices(spec: VisionSyntheticSpec, dataset: List[Dict[str, Any]]):
    if spec.max_patch_tokens_per_batch:
        return iter(
            PatchTokenBatchSampler(
                dataset,
                max_patch_tokens=spec.max_patch_tokens_per_batch,
                patch_size=spec.patch_size,
                max_samples=spec.batch_size,
            )
        )
    return (
        list(range(start, min(start + spec.batch_size, len(dataset))))
        for start in range(0, len(dataset), spec.batch_size)
    )
