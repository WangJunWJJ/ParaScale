# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:54
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Checkpoint contract shared by training backends and serving adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class CheckpointFile:
    path: str
    role: str
    format: str = "torch"
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "format": self.format,
            "required": self.required,
        }


@dataclass(frozen=True)
class CheckpointContract:
    step: int
    backend: str
    files: tuple[CheckpointFile, ...] = ()
    adapter_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "backend": self.backend,
            "adapter_only": self.adapter_only,
            "files": [file.to_dict() for file in self.files],
            "metadata": dict(self.metadata),
        }
