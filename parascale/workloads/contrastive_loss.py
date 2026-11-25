# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午7:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""CLIP and SigLIP objectives over explicit data-parallel feature groups."""

from __future__ import annotations

from typing import Any


class ContrastiveLoss:
    """Compute a local-query mean against all replicas' differentiable keys.

    The runtime binds the data-parallel group and weights local means by sample
    count before gradient averaging. An unbound instance is strictly local.
    Gradient accumulation does not share negatives between microbatches.
    """

    def __init__(self, family: str) -> None:
        if family not in {"clip", "siglip"}:
            raise ValueError("contrastive family must be clip or siglip")
        self.family = family
        self.group = None
        self.objective_metadata = {
            "family": family,
            "version": 1,
            "negative_pool": "data_parallel_microbatch",
            "remote_feature_gradients": True,
            "normalization": "local_query_mean",
        }

    def bind_data_parallel(self, group: Any) -> None:
        """Bind a validated DP process group, or None for local execution."""
        if group is not None:
            import torch.distributed as dist

            if not dist.is_initialized() or dist.get_world_size(group) < 1:
                raise ValueError(
                    "contrastive loss requires a valid data-parallel group"
                )
        self.group = group

    def _gather(self, images: Any, texts: Any) -> tuple[Any, Any, int]:
        import torch
        import torch.distributed as dist

        shape = [
            images.shape[0] if images.ndim == 2 else -1,
            texts.shape[0] if texts.ndim == 2 else -1,
            images.shape[1] if images.ndim == 2 else -1,
            texts.shape[1] if texts.ndim == 2 else -1,
        ]
        world_size = dist.get_world_size(self.group) if self.group is not None else 1
        if world_size > 1:
            metadata = torch.tensor(shape, device=images.device, dtype=torch.int64)
            gathered = [torch.empty_like(metadata) for _ in range(world_size)]
            dist.all_gather(gathered, metadata, group=self.group)
            shapes = torch.stack(gathered).cpu().tolist()
        else:
            shapes = [shape]
        if any(
            count <= 0
            or count != text_count
            or width <= 0
            or width != text_width
            or width != shapes[0][2]
            for count, text_count, width, text_width in shapes
        ):
            raise ValueError(
                "contrastive features require nonempty paired [batch, width] tensors on every rank"
            )
        if world_size == 1:
            return images, texts, 0

        from torch.distributed.nn.functional import all_gather

        sizes = [item[0] for item in shapes]
        # One gather carries both modalities; padding permits unequal tails.
        paired = torch.cat((images, texts), dim=1)
        padding = paired.new_zeros((max(sizes) - paired.shape[0], paired.shape[1]))
        chunks = all_gather(torch.cat((paired, padding), dim=0), group=self.group)
        global_pairs = torch.cat([chunk[:size] for chunk, size in zip(chunks, sizes)])
        all_images, all_texts = global_pairs.chunk(2, dim=1)
        offset = sum(sizes[: dist.get_rank(self.group)])
        return all_images, all_texts, offset

    def __call__(self, output: dict[str, Any], batch: Any) -> Any:
        """Return the CLIP or SigLIP mean; reject invalid paired features."""
        import torch
        import torch.nn.functional as functional

        images, texts = output["image_features"], output["text_features"]
        all_images, all_texts, offset = self._gather(images, texts)
        labels = torch.arange(images.shape[0], device=images.device) + offset
        # Match HF's operand order: CPU BF16 scalar-left multiplication can
        # round the scale first, unlike a tensor multiplied by an FP32 scalar.
        text_logits = None
        if all_images is images:
            # Reuse the same graph for both local directions, as HF does. Two
            # separate BF16 graphs round intermediate gradients independently.
            text_logits = (texts @ images.T) * output["logit_scale"]
            image_logits = text_logits.T
        else:
            image_logits = (images @ all_texts.T) * output["logit_scale"]
        if self.family == "siglip":
            image_logits = image_logits + output["logit_bias"]
            if image_logits.dtype in {torch.float16, torch.bfloat16}:
                image_logits = image_logits.float()
            signs = -torch.ones_like(image_logits)
            signs.scatter_(1, labels[:, None], 1)
            return -functional.logsigmoid(signs * image_logits).sum() / images.shape[0]
        if text_logits is None:
            text_logits = (texts @ all_images.T) * output["logit_scale"]
        return (
            functional.cross_entropy(image_logits, labels)
            + functional.cross_entropy(text_logits, labels)
        ) * 0.5
