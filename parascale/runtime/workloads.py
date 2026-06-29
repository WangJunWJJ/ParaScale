# -*- coding: utf-8 -*-
# @Time : 2026/6/17 下午8:54
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Workload registry for training component factories.

The registry is intentionally small in phase A: it centralizes workload name
resolution while keeping the existing factory implementations in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable

TrainingComponentBuilder = Callable[[Dict[str, Any]], Any]


@dataclass
class WorkloadRegistry:
    builders: Dict[str, TrainingComponentBuilder] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)

    def register(
        self,
        name: str,
        builder: TrainingComponentBuilder,
        *,
        aliases: Iterable[str] = (),
    ) -> None:
        canonical = self._normalize(name)
        self.builders[canonical] = builder
        self.aliases[canonical] = canonical
        for alias in aliases:
            self.aliases[self._normalize(alias)] = canonical

    def resolve(self, name: str) -> str:
        normalized = self._normalize(name)
        try:
            return self.aliases[normalized]
        except KeyError as exc:
            supported = ", ".join(sorted(self.builders))
            raise ValueError(
                f"unsupported factory workload: {name}; supported workloads: {supported}"
            ) from exc

    def create(self, name: str, config_data: Dict[str, Any]) -> Any:
        canonical = self.resolve(name)
        return self.builders[canonical](config_data)

    def names(self) -> list[str]:
        return sorted(self.builders)

    @staticmethod
    def _normalize(name: str) -> str:
        return str(name or "").strip().lower()
