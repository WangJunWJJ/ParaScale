# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:55
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""DDP communication hook recommendation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class DdpHookPlan:
    hook: str = "none"
    reason: str = ""
    trainable_ratio: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hook": self.hook,
            "reason": self.reason,
            "trainable_ratio": self.trainable_ratio,
        }


def recommend_ddp_hook(
    *,
    precision: str,
    task_type: str,
    model_family: str = "",
    trainable_ratio: float | None = None,
) -> DdpHookPlan:
    precision = str(precision or "").lower()
    task_type = str(task_type or "").lower()
    model_family = str(model_family or "").lower()
    if trainable_ratio is not None and trainable_ratio < 0.05:
        return DdpHookPlan(
            hook="none",
            reason="LoRA/adapter-style training has a small trainable ratio; prefer adapter-only synchronization before gradient compression.",
            trainable_ratio=trainable_ratio,
        )
    if task_type == "multimodal" or model_family in {"clip", "siglip", "vlm"}:
        if precision == "bf16":
            return DdpHookPlan(
                hook="bf16_compress",
                reason="Verified CLIP/DataComp native-DDP path benefits from bf16 gradient compression.",
                trainable_ratio=trainable_ratio,
            )
        if precision == "fp16":
            return DdpHookPlan(
                hook="fp16_compress",
                reason="Use fp16 gradient compression for multimodal native-DDP when fp16 precision is selected.",
                trainable_ratio=trainable_ratio,
            )
    return DdpHookPlan(
        hook="none",
        reason="No communication hook selected without scenario-specific benchmark evidence.",
        trainable_ratio=trainable_ratio,
    )
