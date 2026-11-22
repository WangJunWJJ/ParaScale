# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:54
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Checkpoint contract shared by training backends and serving adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol


class CheckpointRuntime(Protocol):
    """Local-only participant in an all-rank checkpoint transaction.

    Methods must not perform collectives. Identity/capture/validate must not
    mutate training or RNG state. Restore may mutate state; application errors
    require job teardown, not an attempt to continue partially restored training.
    """

    def identity(self, optimizer: Any) -> dict[str, Any]:
        """Check optimizer binding and return rank-independent JSON identity."""
        ...

    def capture(self) -> dict[str, Any]:
        """Snapshot committed, not merely prefetched, local progress."""
        ...

    def validate(
        self, state: Any, *, step: int, optimizer_state: dict[str, Any]
    ) -> None:
        """Reject invalid state without applying it or changing global RNG."""
        ...

    def restore(self, state: dict[str, Any]) -> None:
        """Apply validated runtime state before the executor restores Torch RNG."""
        ...


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
