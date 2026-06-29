# -*- coding: utf-8 -*-
# @Time : 2026/6/29 下午4:02
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Setup-time registry for workload adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from parascale.contracts import WorkloadAdapter


@dataclass
class WorkloadAdapterRegistry:
    """Resolve adapters before training or inference enters its hot path."""

    adapters: Dict[str, WorkloadAdapter] = field(default_factory=dict)

    def register(self, adapter: WorkloadAdapter) -> None:
        name = self._normalize(adapter.name)
        if name in self.adapters:
            raise ValueError(f"workload adapter already registered: {name}")
        self.adapters[name] = adapter

    def resolve(self, name: str) -> WorkloadAdapter:
        normalized = self._normalize(name)
        try:
            return self.adapters[normalized]
        except KeyError as exc:
            supported = ", ".join(self.names())
            raise ValueError(
                f"unknown workload adapter: {name}; supported adapters: {supported}"
            ) from exc

    def create(self, name: str, config_data: Mapping[str, Any]) -> Any:
        return self.resolve(name).build(config_data)

    def names(self) -> list[str]:
        return sorted(self.adapters)

    @staticmethod
    def _normalize(name: str) -> str:
        normalized = str(name or "").strip().lower()
        if not normalized:
            raise ValueError("workload adapter name must not be empty")
        return normalized


__all__ = ["WorkloadAdapterRegistry"]
