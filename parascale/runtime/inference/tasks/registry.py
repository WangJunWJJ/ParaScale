# -*- coding: utf-8 -*-
# @Time : 2026/6/29 下午4:06
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Setup-time inference task adapter registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

from .base import InferenceTaskAdapter
from .multimodal import MultimodalInferenceTaskAdapter
from .text import TextInferenceTaskAdapter
from .vision import VisionInferenceTaskAdapter


@dataclass
class InferenceTaskRegistry:
    adapters: Dict[str, InferenceTaskAdapter] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)

    def register(
        self, adapter: InferenceTaskAdapter, *, aliases: Iterable[str] = ()
    ) -> None:
        name = self._normalize(adapter.name)
        if name in self.adapters:
            raise ValueError(f"inference task adapter already registered: {name}")
        self.adapters[name] = adapter
        self.aliases[name] = name
        for alias in aliases:
            normalized_alias = self._normalize(alias)
            if normalized_alias in self.aliases:
                raise ValueError(
                    f"inference task alias already registered: {normalized_alias}"
                )
            self.aliases[normalized_alias] = name

    def resolve(self, name: str) -> InferenceTaskAdapter:
        normalized = self._normalize(name)
        try:
            return self.adapters[self.aliases[normalized]]
        except KeyError as exc:
            supported = ", ".join(sorted(self.adapters))
            raise ValueError(
                f"unknown inference task: {name}; supported tasks: {supported}"
            ) from exc

    @staticmethod
    def _normalize(name: str) -> str:
        return str(name or "").strip().lower()


def default_inference_task_registry() -> InferenceTaskRegistry:
    registry = InferenceTaskRegistry()
    registry.register(VisionInferenceTaskAdapter(), aliases=("vision_detection",))
    registry.register(TextInferenceTaskAdapter(), aliases=("text_generation",))
    registry.register(
        MultimodalInferenceTaskAdapter(), aliases=("multimodal_embedding", "vlm")
    )
    return registry


__all__ = ["InferenceTaskRegistry", "default_inference_task_registry"]
