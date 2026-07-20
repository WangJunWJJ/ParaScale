# -*- coding: utf-8 -*-
# @Time : 2026/6/18 下午4:09
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""VLM LoRA workload builders and processor adapters."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

from parascale.data import MultiModalCollator
from parascale.workloads.specs.clip import ClipContrastiveSpec
from parascale.workloads.specs.vlm_lora import VlmLoraSpec

from .clip import _clip_sample_batches
from .common import _patchify, _require_torch
from .datacomp import _DataCompWdsIterableDataset
from .optimizer import build_adamw_optimizer_for_model
from .vlm_cache import (
    _load_vlm_processor_cache,
    _normalize_pipeline_profile,
    _pipeline_profile_from_sample_metadata,
    _sample_to_pil_image,
    _save_vlm_processor_cache,
    _timed_vlm_processor_call,
    _vlm_processor_cache_key,
    _vlm_prompt,
)


def build_vlm_lora_components(spec: VlmLoraSpec):
    torch = _require_torch()
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(spec.seed)

    class LoRALinear(nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            rank: int,
            alpha: float,
            dropout: float,
        ) -> None:
            super().__init__()
            self.base = nn.Linear(in_features, out_features)
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.lora_a = nn.Linear(in_features, rank, bias=False)
            self.lora_b = nn.Linear(rank, out_features, bias=False)
            self.scaling = float(alpha) / float(max(rank, 1))
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)

        def forward(self, value):
            value = value.to(dtype=self.base.weight.dtype)
            adapter = self.lora_b(self.lora_a(self.dropout(value))) * self.scaling
            return self.base(value) + adapter

    if spec.model_type in {
        "hf_vlm_lora",
        "qwen2_vl_lora",
        "llava_onevision_lora",
        "internvl_lora",
        "real_vlm_lora",
    }:
        return _build_hf_vlm_lora_components(torch, nn, optim, spec)

    if spec.model_type in {"hf_clip_lora", "openai_clip_lora"}:
        return _build_hf_clip_lora_components(torch, nn, optim, LoRALinear, spec)

    class TinyVlmLora(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            patch_dim = spec.channels * spec.patch_size * spec.patch_size
            self.image_adapter = LoRALinear(
                patch_dim,
                spec.embed_dim,
                rank=spec.lora_rank,
                alpha=spec.lora_alpha,
                dropout=spec.lora_dropout,
            )
            self.text_embed = nn.Embedding(spec.vocab_size, spec.embed_dim)
            for parameter in self.text_embed.parameters():
                parameter.requires_grad_(False)
            self.fusion_adapter = LoRALinear(
                spec.embed_dim * 2,
                spec.embed_dim,
                rank=spec.lora_rank,
                alpha=spec.lora_alpha,
                dropout=spec.lora_dropout,
            )
            self.norm = nn.LayerNorm(spec.embed_dim)
            self.lm_head = nn.Linear(spec.embed_dim, spec.vocab_size)
            if not spec.train_lm_head:
                for parameter in self.lm_head.parameters():
                    parameter.requires_grad_(False)
            self.total_parameters = sum(
                parameter.numel() for parameter in self.parameters()
            )
            self.trainable_parameters = sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
            self.adapter_parameters = sum(
                parameter.numel()
                for name, parameter in self.named_parameters()
                if parameter.requires_grad and "lora_" in name
            )
            self.trainable_ratio = self.trainable_parameters / max(
                float(self.total_parameters), 1.0
            )
            self.lora_rank = int(spec.lora_rank)

        def forward(
            self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs
        ):
            images = pixel_values if pixel_values is not None else kwargs["images"]
            tokens = input_ids if input_ids is not None else kwargs["input_ids"]
            patches = _patchify(
                torch,
                images.to(dtype=self.image_adapter.base.weight.dtype),
                spec.patch_size,
            )
            image_features = self.image_adapter(patches).mean(dim=1)
            token_features = self.text_embed(tokens.long().clamp(min=0))
            if attention_mask is not None:
                mask = attention_mask.float().unsqueeze(-1)
                text_features = (token_features * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp_min(1.0)
            else:
                text_features = token_features.mean(dim=1)
            fused = torch.cat([image_features, text_features], dim=-1)
            fused = self.norm(nn.functional.gelu(self.fusion_adapter(fused)))
            return self.lm_head(fused.to(dtype=self.lm_head.weight.dtype))

        def adapter_state_dict(self) -> Dict[str, Any]:
            return {
                name: value.detach().cpu()
                for name, value in self.state_dict().items()
                if "lora_" in name or (spec.train_lm_head and "lm_head" in name)
            }

        def load_adapter_state_dict(self, state: Mapping[str, Any]) -> None:
            current = self.state_dict()
            current.update(dict(state))
            self.load_state_dict(current)

    model = TinyVlmLora()
    optimizer = build_adamw_optimizer_for_model(optim, model, lr=spec.lr)
    collator = MultiModalCollator(max_length=spec.text_length, return_tensors="pt")
    clip_spec = ClipContrastiveSpec(
        model_type="tiny_clip",
        data_type=spec.data_type,
        data_dir=spec.data_dir,
        metadata_path=spec.metadata_path,
        streaming=spec.streaming,
        num_workers=spec.num_workers,
        prefetch_factor=spec.prefetch_factor,
        persistent_workers=spec.persistent_workers,
        dataset_local_cache_dir=spec.dataset_local_cache_dir,
        wds_image_mode="bytes",
        image_size=spec.image_size,
        channels=spec.channels,
        patch_size=spec.patch_size,
        vocab_size=spec.vocab_size,
        text_length=spec.text_length,
        embed_dim=spec.embed_dim,
        num_samples=spec.num_samples,
        batch_size=spec.batch_size,
        num_batches=spec.num_batches,
        lr=spec.lr,
        seed=spec.seed,
        device=spec.device,
    )

    def dataloader() -> Iterable[Dict[str, Any]]:
        yielded = 0
        for samples in _clip_sample_batches(torch, clip_spec):
            batch = collator(samples)
            batch["labels"] = _vlm_lora_targets(torch, batch["input_ids"], spec)
            _attach_vlm_lora_batch_metrics(batch, model)
            yield batch
            yielded += 1
            if yielded >= spec.num_batches:
                break

    def loss_fn(output, batch):
        return nn.functional.cross_entropy(output, batch["labels"])

    return model, optimizer, dataloader(), loss_fn


def _vlm_lora_targets(torch: Any, input_ids: Any, spec: VlmLoraSpec):
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        return torch.zeros(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )
    targets = input_ids[:, 0].long().remainder(max(2, spec.vocab_size))
    return targets.to(device=input_ids.device)


def _attach_vlm_lora_batch_metrics(batch: Dict[str, Any], model: Any) -> None:
    batch["adapter_params"] = int(getattr(model, "adapter_parameters", 0) or 0)
    batch["trainable_params"] = int(getattr(model, "trainable_parameters", 0) or 0)
    batch["total_params"] = int(getattr(model, "total_parameters", 0) or 0)
    batch["trainable_ratio"] = float(getattr(model, "trainable_ratio", 0.0) or 0.0)
    batch["lora_rank"] = int(getattr(model, "lora_rank", 0) or 0)


def _build_hf_vlm_lora_components(torch: Any, nn: Any, optim: Any, spec: VlmLoraSpec):
    try:
        from peft import (
            LoraConfig,
            get_peft_model,
            get_peft_model_state_dict,
            set_peft_model_state_dict,
        )
    except Exception as exc:
        raise ImportError(
            "Real VLM LoRA workloads require peft. Install peft in the runtime image."
        ) from exc
    try:
        from transformers import AutoConfig, AutoProcessor
    except Exception as exc:
        raise ImportError(
            "Real VLM LoRA workloads require transformers and local VLM weights."
        ) from exc
    if not spec.pretrained_model_name_or_path:
        raise ValueError(
            "model.pretrained_model_name_or_path is required for real VLM LoRA."
        )

    model_path = spec.pretrained_model_name_or_path
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    base_model = _load_hf_vlm_model(model_path, spec, AutoConfig)
    if spec.activation_checkpointing and hasattr(
        base_model, "gradient_checkpointing_enable"
    ):
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "config"):
            base_model.config.use_cache = False
    lora_config = LoraConfig(
        r=spec.lora_rank,
        lora_alpha=spec.lora_alpha,
        target_modules=list(spec.lora_target_modules),
        lora_dropout=spec.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.train()

    class HfVlmLoraWrapper(nn.Module):
        def __init__(self, wrapped_model: Any) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model
            self.lora_rank = int(spec.lora_rank)
            self.total_parameters = sum(
                parameter.numel() for parameter in self.parameters()
            )
            self.trainable_parameters = sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
            self.adapter_parameters = self.trainable_parameters
            self.trainable_ratio = self.trainable_parameters / max(
                float(self.total_parameters), 1.0
            )

        def forward(self, **batch: Any):
            accepted = {
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
                "image_sizes",
                "cross_attention_mask",
                "labels",
            }
            model_inputs = {
                key: value for key, value in batch.items() if key in accepted
            }
            model_inputs["use_cache"] = False
            return self.wrapped_model(**model_inputs)

        def adapter_state_dict(self) -> Dict[str, Any]:
            return get_peft_model_state_dict(self.wrapped_model)

        def load_adapter_state_dict(self, state: Mapping[str, Any]) -> None:
            set_peft_model_state_dict(self.wrapped_model, state)

    model = HfVlmLoraWrapper(peft_model)
    optimizer = build_adamw_optimizer_for_model(optim, model, lr=spec.lr)
    clip_spec = ClipContrastiveSpec(
        model_type="tiny_clip",
        data_type=spec.data_type,
        data_dir=spec.data_dir,
        metadata_path=spec.metadata_path,
        streaming=spec.streaming,
        num_workers=spec.num_workers,
        prefetch_factor=spec.prefetch_factor,
        persistent_workers=spec.persistent_workers,
        dataset_local_cache_dir=spec.dataset_local_cache_dir,
        wds_image_mode="bytes",
        image_size=spec.image_size,
        channels=spec.channels,
        patch_size=spec.patch_size,
        vocab_size=spec.vocab_size,
        text_length=spec.text_length,
        embed_dim=spec.embed_dim,
        num_samples=spec.num_samples,
        batch_size=spec.batch_size,
        num_batches=spec.num_batches,
        lr=spec.lr,
        seed=spec.seed,
        device=spec.device,
    )

    def dataloader() -> Iterable[Dict[str, Any]]:
        yielded = 0
        if (
            spec.preprocess_in_workers
            and spec.streaming
            and spec.data_type in {"datacomp_wds", "webdataset", "wds"}
            and spec.num_workers > 0
        ):
            batches = _stream_vlm_processor_batches(torch, processor, clip_spec, spec)
        else:
            batches = (
                _vlm_processor_batch(torch, processor, samples, spec)
                for samples in _clip_sample_batches(torch, clip_spec)
            )
        for batch in batches:
            _attach_vlm_lora_batch_metrics(batch, model)
            yield batch
            yielded += 1
            if yielded >= spec.num_batches:
                break

    def loss_fn(output: Any, batch: Dict[str, Any]):
        loss = getattr(output, "loss", None)
        if loss is not None:
            return loss
        logits = getattr(output, "logits", None)
        if logits is None:
            raise RuntimeError("VLM model output must expose loss or logits.")
        labels = batch["labels"]
        return nn.functional.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )

    return model, optimizer, dataloader(), loss_fn


def _load_hf_vlm_model(model_path: str, spec: VlmLoraSpec, auto_config: Any):
    config = auto_config.from_pretrained(model_path, trust_remote_code=True)
    model_type = str(getattr(config, "model_type", "") or "").lower()
    loaders = []
    try:
        from transformers import AutoModelForImageTextToText

        loaders.append(AutoModelForImageTextToText)
    except Exception:
        pass
    try:
        from transformers import AutoModelForVision2Seq

        loaders.append(AutoModelForVision2Seq)
    except Exception:
        pass
    try:
        from transformers import AutoModelForCausalLM

        loaders.append(AutoModelForCausalLM)
    except Exception:
        pass
    torch_dtype = _hf_torch_dtype(spec)
    last_error: Exception | None = None
    for loader in loaders:
        try:
            return loader.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Unable to load VLM model {model_path} with model_type={model_type}."
    ) from last_error


def _hf_torch_dtype(spec: VlmLoraSpec):
    try:
        import torch
    except Exception:
        return None
    if spec.model_type and "bf16" in spec.model_type:
        return torch.bfloat16
    return None


def _stream_vlm_processor_batches(
    torch: Any,
    processor: Any,
    clip_spec: ClipContrastiveSpec,
    spec: VlmLoraSpec,
) -> Iterable[Dict[str, Any]]:
    try:
        from torch.utils.data import DataLoader, IterableDataset
    except Exception:
        for samples in _clip_sample_batches(torch, clip_spec):
            yield _vlm_processor_batch(torch, processor, samples, spec)
        return

    class TorchDataCompWdsIterableDataset(_DataCompWdsIterableDataset, IterableDataset):
        pass

    dataset = TorchDataCompWdsIterableDataset(torch, clip_spec)

    def collate(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return _vlm_processor_batch(torch, processor, samples, spec)

    kwargs: Dict[str, Any] = {
        "batch_size": spec.batch_size,
        "num_workers": max(0, spec.num_workers),
        "collate_fn": collate,
        "pin_memory": bool(spec.pin_memory),
    }
    if spec.num_workers > 0:
        kwargs["prefetch_factor"] = max(1, spec.prefetch_factor)
        kwargs["persistent_workers"] = bool(spec.persistent_workers)
    loader = DataLoader(dataset, **kwargs)
    yielded = 0
    for batch in loader:
        if batch:
            yield batch
            yielded += 1
        if yielded >= spec.num_batches:
            break


def _vlm_processor_batch(
    torch: Any,
    processor: Any,
    samples: Sequence[Mapping[str, Any]],
    spec: VlmLoraSpec,
) -> Dict[str, Any]:
    profile: Dict[str, float] = _pipeline_profile_from_sample_metadata(samples)
    start = time.perf_counter()
    prompts = [_vlm_prompt(processor, sample, spec) for sample in samples]
    profile["prompt_template_ms"] = (time.perf_counter() - start) * 1000.0

    cache_key = _vlm_processor_cache_key(processor, samples, prompts, spec)
    encoded = _load_vlm_processor_cache(torch, spec, cache_key)
    if encoded is None:
        start = time.perf_counter()
        images = [_sample_to_pil_image(sample, spec) for sample in samples]
        profile["image_decode_ms"] = (time.perf_counter() - start) * 1000.0
        processor_start = time.perf_counter()
        encoded, nested_profile = _timed_vlm_processor_call(
            processor,
            text=prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=spec.text_length,
            return_tensors="pt",
        )
        profile["processor_ms"] = (time.perf_counter() - processor_start) * 1000.0
        profile.update(nested_profile)
        _save_vlm_processor_cache(torch, spec, cache_key, encoded)
        profile["cache_hit"] = 0.0
    else:
        profile["processor_ms"] = 0.0
        profile["tokenizer_ms"] = 0.0
        profile["image_processor_ms"] = 0.0
        profile["image_decode_ms"] = 0.0
        profile["cache_hit"] = 1.0

    batch = dict(encoded)

    label_start = time.perf_counter()
    input_ids = batch.get("input_ids")
    if input_ids is None:
        raise RuntimeError("VLM processor did not return input_ids.")
    labels = input_ids.clone()
    attention_mask = batch.get("attention_mask")
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels = labels.masked_fill(input_ids == int(pad_token_id), -100)
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)
    profile["label_build_ms"] = (time.perf_counter() - label_start) * 1000.0
    batch["labels"] = labels
    batch["tokens"] = (
        int(attention_mask.sum().item())
        if attention_mask is not None
        else int(input_ids.numel())
    )
    batch["num_images"] = len(samples)
    batch["num_pairs"] = len(samples)
    batch["image_text_pairs"] = len(samples)
    batch["pipeline_profile"] = _normalize_pipeline_profile(profile)
    return batch


def _build_hf_clip_lora_components(
    torch: Any,
    nn: Any,
    optim: Any,
    lora_linear_cls: Any,
    spec: VlmLoraSpec,
):
    try:
        from transformers import AutoModel
    except Exception as exc:
        raise ImportError(
            "HF CLIP LoRA workloads require transformers and local model weights."
        ) from exc
    if not spec.pretrained_model_name_or_path:
        raise ValueError(
            "model.pretrained_model_name_or_path is required for hf_clip_lora."
        )

    base_model = AutoModel.from_pretrained(spec.pretrained_model_name_or_path)
    if spec.activation_checkpointing and hasattr(
        base_model, "gradient_checkpointing_enable"
    ):
        base_model.gradient_checkpointing_enable()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    feature_dim = int(
        getattr(
            getattr(base_model, "config", None),
            "projection_dim",
            spec.embed_dim,
        )
        or spec.embed_dim
    )

    class HFClipLoraAdapter(nn.Module):
        def __init__(self, wrapped_model) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model
            self.image_adapter = lora_linear_cls(
                feature_dim,
                spec.embed_dim,
                rank=spec.lora_rank,
                alpha=spec.lora_alpha,
                dropout=spec.lora_dropout,
            )
            self.text_adapter = lora_linear_cls(
                feature_dim,
                spec.embed_dim,
                rank=spec.lora_rank,
                alpha=spec.lora_alpha,
                dropout=spec.lora_dropout,
            )
            self.fusion_adapter = lora_linear_cls(
                spec.embed_dim * 2,
                spec.embed_dim,
                rank=spec.lora_rank,
                alpha=spec.lora_alpha,
                dropout=spec.lora_dropout,
            )
            self.norm = nn.LayerNorm(spec.embed_dim)
            self.lm_head = nn.Linear(spec.embed_dim, spec.vocab_size)
            if not spec.train_lm_head:
                for parameter in self.lm_head.parameters():
                    parameter.requires_grad_(False)
            self.total_parameters = sum(
                parameter.numel() for parameter in self.parameters()
            )
            self.trainable_parameters = sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )

        def forward(
            self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs
        ):
            with torch.no_grad():
                output = self.wrapped_model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                image_embeds = getattr(output, "image_embeds", None)
                text_embeds = getattr(output, "text_embeds", None)
                if image_embeds is None and isinstance(output, dict):
                    image_embeds = output.get("image_embeds")
                    text_embeds = output.get("text_embeds")
                if image_embeds is None or text_embeds is None:
                    raise RuntimeError(
                        "hf_clip_lora requires a CLIP-like model exposing "
                        "image_embeds and text_embeds."
                    )
            image_features = self.image_adapter(image_embeds.detach())
            text_features = self.text_adapter(text_embeds.detach())
            fused = torch.cat([image_features, text_features], dim=-1)
            fused = self.norm(nn.functional.gelu(self.fusion_adapter(fused)))
            return self.lm_head(fused.to(dtype=self.lm_head.weight.dtype))

    model = HFClipLoraAdapter(base_model)
    optimizer = build_adamw_optimizer_for_model(optim, model, lr=spec.lr)
    collator = MultiModalCollator(max_length=spec.text_length, return_tensors="pt")
    clip_spec = ClipContrastiveSpec(
        model_type="hf_clip",
        data_type=spec.data_type,
        data_dir=spec.data_dir,
        metadata_path=spec.metadata_path,
        streaming=spec.streaming,
        num_workers=spec.num_workers,
        prefetch_factor=spec.prefetch_factor,
        persistent_workers=spec.persistent_workers,
        dataset_local_cache_dir=spec.dataset_local_cache_dir,
        image_size=spec.image_size,
        channels=spec.channels,
        patch_size=spec.patch_size,
        vocab_size=spec.vocab_size,
        text_length=spec.text_length,
        embed_dim=spec.embed_dim,
        pretrained_model_name_or_path=spec.pretrained_model_name_or_path,
        num_samples=spec.num_samples,
        batch_size=spec.batch_size,
        num_batches=spec.num_batches,
        lr=spec.lr,
        seed=spec.seed,
        device=spec.device,
    )

    def dataloader() -> Iterable[Dict[str, Any]]:
        yielded = 0
        for samples in _clip_sample_batches(torch, clip_spec):
            batch = collator(samples)
            batch["labels"] = _vlm_lora_targets(torch, batch["input_ids"], spec)
            _attach_vlm_lora_batch_metrics(batch, model)
            yield batch
            yielded += 1
            if yielded >= spec.num_batches:
                break

    def loss_fn(output, batch):
        return nn.functional.cross_entropy(output, batch["labels"])

    return model, optimizer, dataloader(), loss_fn
