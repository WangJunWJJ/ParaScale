# -*- coding: utf-8 -*-
# @Time : 2026/6/18 下午4:09
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""CLIP-style contrastive workload builders."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List

from parascale.data import MultiModalCollator
from parascale.data.vision import estimate_patch_tokens
from parascale.workloads.specs.clip import ClipContrastiveSpec

from .common import (
    _patchify,
    _require_torch,
    _suppress_activation_checkpointing_future_warning,
)
from .datacomp import (
    _build_datacomp_metadata_dataset,
    _build_datacomp_wds_dataset,
    _stream_datacomp_wds_batches,
)


def build_clip_contrastive_components(spec: ClipContrastiveSpec):
    torch = _require_torch()
    import torch.nn as nn
    import torch.optim as optim

    if spec.activation_checkpointing:
        _suppress_activation_checkpointing_future_warning()

    torch.manual_seed(spec.seed)

    if spec.model_type in {"hf_clip", "openai_clip", "siglip", "hf_siglip"}:
        return _build_hf_clip_contrastive_components(torch, spec)

    class TinyClipContrastive(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            patch_dim = spec.channels * spec.patch_size * spec.patch_size
            self.image_patch = nn.Linear(patch_dim, spec.embed_dim)
            self.text_embed = nn.Embedding(spec.vocab_size, spec.embed_dim)
            self.image_proj = nn.Linear(spec.embed_dim, spec.embed_dim)
            self.text_proj = nn.Linear(spec.embed_dim, spec.embed_dim)
            self.logit_scale = nn.Parameter(torch.tensor(1.0 / spec.temperature))

        def forward(
            self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs
        ):
            images = pixel_values if pixel_values is not None else kwargs["images"]
            tokens = input_ids if input_ids is not None else kwargs["input_ids"]
            patches = _patchify(
                torch, images.to(dtype=self.image_patch.weight.dtype), spec.patch_size
            )
            image_features = self.image_patch(patches).mean(dim=1)
            token_features = self.text_embed(tokens.long())
            if attention_mask is not None:
                mask = attention_mask.float().unsqueeze(-1)
                token_features = token_features * mask
                denom = mask.sum(dim=1).clamp_min(1.0)
                text_features = token_features.sum(dim=1) / denom
            else:
                text_features = token_features.mean(dim=1)
            image_features = nn.functional.normalize(
                self._project(image_features, self.image_proj), dim=-1
            )
            text_features = nn.functional.normalize(
                self._project(text_features, self.text_proj), dim=-1
            )
            return self.logit_scale.exp().clamp(max=100.0) * (
                image_features @ text_features.T
            )

        @staticmethod
        def _project(features, projection):
            return projection(features.to(dtype=projection.weight.dtype))

    class MediumClipContrastive(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            grid = spec.image_size // spec.patch_size
            sequence_length = grid * grid + 1
            self.patch_embed = nn.Conv2d(
                spec.channels,
                spec.embed_dim,
                kernel_size=spec.patch_size,
                stride=spec.patch_size,
            )
            self.image_cls = nn.Parameter(torch.zeros(1, 1, spec.embed_dim))
            self.image_pos = nn.Parameter(
                torch.zeros(1, sequence_length, spec.embed_dim)
            )
            self.text_embed = nn.Embedding(spec.vocab_size, spec.embed_dim)
            self.text_pos = nn.Parameter(
                torch.zeros(1, spec.text_length, spec.embed_dim)
            )
            vision_layer = nn.TransformerEncoderLayer(
                d_model=spec.embed_dim,
                nhead=spec.num_heads,
                dim_feedforward=int(spec.embed_dim * spec.mlp_ratio),
                batch_first=True,
                activation="gelu",
            )
            text_layer = nn.TransformerEncoderLayer(
                d_model=spec.embed_dim,
                nhead=spec.num_heads,
                dim_feedforward=int(spec.embed_dim * spec.mlp_ratio),
                batch_first=True,
                activation="gelu",
            )
            self.vision_encoder = nn.TransformerEncoder(
                vision_layer, num_layers=max(1, spec.vision_layers)
            )
            self.text_encoder = nn.TransformerEncoder(
                text_layer, num_layers=max(1, spec.text_layers or spec.vision_layers)
            )
            self.image_norm = nn.LayerNorm(spec.embed_dim)
            self.text_norm = nn.LayerNorm(spec.embed_dim)
            self.image_proj = nn.Linear(spec.embed_dim, spec.embed_dim)
            self.text_proj = nn.Linear(spec.embed_dim, spec.embed_dim)
            self.logit_scale = nn.Parameter(torch.tensor(1.0 / spec.temperature))
            self._init_parameters()

        def _init_parameters(self) -> None:
            nn.init.normal_(self.image_cls, std=0.02)
            nn.init.normal_(self.image_pos, std=0.02)
            nn.init.normal_(self.text_pos, std=0.02)

        def forward(
            self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs
        ):
            images = pixel_values if pixel_values is not None else kwargs["images"]
            tokens = input_ids if input_ids is not None else kwargs["input_ids"]
            batch_size = images.shape[0]
            image_tokens = (
                self.patch_embed(images.to(dtype=self.patch_embed.weight.dtype))
                .flatten(2)
                .transpose(1, 2)
            )
            cls = self.image_cls.expand(batch_size, -1, -1)
            image_tokens = torch.cat([cls, image_tokens], dim=1)
            image_tokens = image_tokens + self.image_pos[:, : image_tokens.shape[1], :]
            image_features = self._encode(
                self.vision_encoder, image_tokens, key_padding_mask=None
            )[:, 0]

            text_tokens = self.text_embed(tokens.long())
            text_tokens = text_tokens + self.text_pos[:, : text_tokens.shape[1], :]
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = attention_mask.to(dtype=torch.bool).logical_not()
            text_encoded = self._encode(
                self.text_encoder, text_tokens, key_padding_mask=key_padding_mask
            )
            if attention_mask is not None:
                mask = attention_mask.float().unsqueeze(-1)
                text_features = (text_encoded * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp_min(1.0)
            else:
                text_features = text_encoded.mean(dim=1)

            image_features = nn.functional.normalize(
                self._project_with_norm(
                    image_features, self.image_norm, self.image_proj
                ),
                dim=-1,
            )
            text_features = nn.functional.normalize(
                self._project_with_norm(text_features, self.text_norm, self.text_proj),
                dim=-1,
            )
            return self.logit_scale.exp().clamp(max=100.0) * (
                image_features @ text_features.T
            )

        @staticmethod
        def _project_with_norm(features, norm, projection):
            normed = norm(features.to(dtype=norm.weight.dtype))
            return projection(normed.to(dtype=projection.weight.dtype))

        def _encode(self, encoder, tokens, *, key_padding_mask=None):
            if not spec.activation_checkpointing or not self.training:
                if key_padding_mask is None:
                    return encoder(tokens)
                return encoder(tokens, src_key_padding_mask=key_padding_mask)
            try:
                from torch.utils.checkpoint import checkpoint
            except Exception:
                if key_padding_mask is None:
                    return encoder(tokens)
                return encoder(tokens, src_key_padding_mask=key_padding_mask)
            hidden = tokens
            for layer in encoder.layers:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="`torch.cpu.amp.autocast\\(args\\.\\.\\.\\)` is deprecated.*",
                        category=FutureWarning,
                    )
                    if key_padding_mask is None:
                        hidden = checkpoint(layer, hidden, use_reentrant=False)
                    else:
                        hidden = checkpoint(
                            lambda value, mask, module=layer: module(
                                value, src_key_padding_mask=mask
                            ),
                            hidden,
                            key_padding_mask,
                            use_reentrant=False,
                        )
            if encoder.norm is not None:
                hidden = encoder.norm(hidden)
            return hidden

    model_cls = (
        MediumClipContrastive
        if spec.model_type in {"clip_medium", "clip_b", "vit_b_clip", "clip_vit_b"}
        else TinyClipContrastive
    )
    model = model_cls()
    optimizer = optim.AdamW(model.parameters(), lr=spec.lr)
    collator = MultiModalCollator(max_length=spec.text_length, return_tensors="pt")

    def dataloader() -> Iterable[Dict[str, Any]]:
        yielded = 0
        for samples in _clip_sample_batches(torch, spec):
            batch = collator(samples)
            batch["labels"] = torch.arange(
                batch["input_ids"].shape[0], dtype=torch.long
            )
            yield batch
            yielded += 1
            if yielded >= spec.num_batches:
                break

    def loss_fn(output, batch):
        labels = batch["labels"]
        image_to_text = nn.functional.cross_entropy(output, labels)
        text_to_image = nn.functional.cross_entropy(output.T, labels)
        return (image_to_text + text_to_image) * 0.5

    return model, optimizer, dataloader(), loss_fn


def _build_hf_clip_contrastive_components(torch: Any, spec: ClipContrastiveSpec):
    try:
        from transformers import AutoModel
    except Exception as exc:
        raise ImportError(
            "Pretrained CLIP/SigLIP workloads require transformers. "
            "Install transformers and ensure weights are available locally or online."
        ) from exc
    if not spec.pretrained_model_name_or_path:
        raise ValueError(
            "model.pretrained_model_name_or_path is required for hf_clip/hf_siglip workloads."
        )
    import torch.nn as nn
    import torch.optim as optim

    hf_model = AutoModel.from_pretrained(spec.pretrained_model_name_or_path)
    if spec.activation_checkpointing and hasattr(
        hf_model, "gradient_checkpointing_enable"
    ):
        hf_model.gradient_checkpointing_enable()

    class HFContrastiveWrapper(nn.Module):
        def __init__(self, wrapped_model) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(
            self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs
        ):
            return self.wrapped_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

    model = HFContrastiveWrapper(hf_model)
    optimizer = optim.AdamW(model.parameters(), lr=spec.lr)
    collator = MultiModalCollator(max_length=spec.text_length, return_tensors="pt")

    def dataloader() -> Iterable[Dict[str, Any]]:
        yielded = 0
        for samples in _clip_sample_batches(torch, spec):
            batch = collator(samples)
            batch["labels"] = torch.arange(
                batch["input_ids"].shape[0], dtype=torch.long
            )
            yield batch
            yielded += 1
            if yielded >= spec.num_batches:
                break

    def loss_fn(output, batch):
        logits = getattr(output, "logits_per_image", None)
        if logits is None and isinstance(output, dict):
            logits = output.get("logits_per_image")
        if logits is None:
            image_embeds = getattr(output, "image_embeds", None)
            text_embeds = getattr(output, "text_embeds", None)
            if image_embeds is None or text_embeds is None:
                raise RuntimeError(
                    "HuggingFace CLIP/SigLIP output must expose logits or image/text embeddings."
                )
            image_embeds = nn.functional.normalize(image_embeds, dim=-1)
            text_embeds = nn.functional.normalize(text_embeds, dim=-1)
            logits = image_embeds @ text_embeds.T
        labels = batch["labels"]
        image_to_text = nn.functional.cross_entropy(logits, labels)
        text_to_image = nn.functional.cross_entropy(logits.T, labels)
        return (image_to_text + text_to_image) * 0.5

    return model, optimizer, dataloader(), loss_fn


def _clip_sample_batches(
    torch: Any, spec: ClipContrastiveSpec
) -> Iterable[List[Dict[str, Any]]]:
    if spec.streaming and spec.data_type in {"datacomp_wds", "webdataset", "wds"}:
        yield from _stream_datacomp_wds_batches(torch, spec)
        return
    dataset = _build_clip_dataset(torch, spec)
    for start in range(0, len(dataset), spec.batch_size):
        yield dataset[start : start + spec.batch_size]


def _build_clip_dataset(torch: Any, spec: ClipContrastiveSpec) -> List[Dict[str, Any]]:
    if spec.data_type in {"datacomp_wds", "webdataset", "wds"}:
        return _build_datacomp_wds_dataset(torch, spec)
    if spec.data_type in {"datacomp_metadata", "metadata_parquet"}:
        return _build_datacomp_metadata_dataset(torch, spec)
    return _build_synthetic_clip_dataset(torch, spec)


def _build_synthetic_clip_dataset(
    torch: Any, spec: ClipContrastiveSpec
) -> List[Dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(spec.seed)
    dataset: List[Dict[str, Any]] = []
    patch_tokens = estimate_patch_tokens(
        spec.image_size, spec.image_size, spec.patch_size
    )
    for index in range(spec.num_samples):
        image = torch.randn(
            spec.channels, spec.image_size, spec.image_size, generator=generator
        )
        length = max(2, spec.text_length - (index % 4))
        input_ids = torch.randint(
            low=1,
            high=spec.vocab_size,
            size=(length,),
            generator=generator,
        ).tolist()
        dataset.append(
            {
                "pixel_values": image,
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "text": f"synthetic caption {index}",
                "height": spec.image_size,
                "width": spec.image_size,
                "patch_tokens": patch_tokens,
                "metadata": {"sample_id": f"clip-{index:04d}"},
            }
        )
    return dataset
