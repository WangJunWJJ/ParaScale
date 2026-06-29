# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:57
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Multimodal task and batch specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class MultiModalTaskSpec:
    name: str
    objective: str
    modalities: tuple[str, ...]
    adapter_policy: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "objective": self.objective,
            "modalities": list(self.modalities),
            "adapter_policy": self.adapter_policy,
        }


@dataclass(frozen=True)
class VlmLoraSpec(MultiModalTaskSpec):
    lora_rank: int = 16
    lora_alpha: int = 32
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    def __init__(
        self,
        name: str = "vlm_lora",
        objective: str = "supervised_finetune",
        modalities: tuple[str, ...] = ("text", "image"),
        adapter_policy: str = "lora",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "modalities", modalities)
        object.__setattr__(self, "adapter_policy", adapter_policy)
        object.__setattr__(self, "lora_rank", lora_rank)
        object.__setattr__(self, "lora_alpha", lora_alpha)
        object.__setattr__(self, "target_modules", target_modules)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "target_modules": list(self.target_modules),
            }
        )
        return data


@dataclass(frozen=True)
class ContrastivePairSpec(MultiModalTaskSpec):
    temperature: float = 0.07
    symmetric_loss: bool = True

    def __init__(
        self,
        name: str = "clip_contrastive",
        objective: str = "image_text_contrastive",
        modalities: tuple[str, ...] = ("text", "image"),
        adapter_policy: str = "projection_head",
        temperature: float = 0.07,
        symmetric_loss: bool = True,
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "modalities", modalities)
        object.__setattr__(self, "adapter_policy", adapter_policy)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "symmetric_loss", symmetric_loss)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {"temperature": self.temperature, "symmetric_loss": self.symmetric_loss}
        )
        return data


__all__ = ["ContrastivePairSpec", "MultiModalTaskSpec", "VlmLoraSpec"]
