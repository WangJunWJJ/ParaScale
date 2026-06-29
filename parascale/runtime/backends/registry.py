# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:59
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Training backend registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from .ascend_native import AscendNativeTrainingBackend
from .base import TrainingBackend
from .deepspeed import DeepSpeedTrainingBackend
from .fsdp import FSDPTrainingBackend
from .native import NativeDdpTrainingBackend, NativeTrainingBackend


@dataclass
class TrainingBackendRegistry:
    factories: Dict[str, Callable[..., TrainingBackend]] = field(default_factory=dict)

    def register(self, name: str, factory: Callable[..., TrainingBackend]) -> None:
        self.factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> TrainingBackend:
        if name not in self.factories:
            raise ValueError(f"unknown training backend: {name}")
        return self.factories[name](**kwargs)


def default_training_backend_registry() -> TrainingBackendRegistry:
    registry = TrainingBackendRegistry()
    registry.register("native", lambda **kwargs: NativeTrainingBackend(**kwargs))
    registry.register("native_ddp", lambda **kwargs: NativeDdpTrainingBackend(**kwargs))
    registry.register("fsdp", lambda **kwargs: FSDPTrainingBackend(**kwargs))
    registry.register("deepspeed", lambda **kwargs: DeepSpeedTrainingBackend(**kwargs))
    registry.register(
        "ascend_native", lambda **kwargs: AscendNativeTrainingBackend(**kwargs)
    )
    return registry


def create_runtime_training_backend(
    model: Any = None,
    optimizer: Any = None,
    config: Any = None,
    local_rank: int = 0,
) -> TrainingBackend:
    backend_name = (
        getattr(config, "training_backend", "native")
        if config is not None
        else "native"
    )
    if backend_name == "auto":
        zero_stage = int(getattr(config, "zero_stage", 0) or 0)
        zero_offload = bool(getattr(config, "zero_offload", False))
        backend_name = "deepspeed" if zero_offload or zero_stage >= 3 else "fsdp"
        if config is not None:
            config.training_backend = backend_name
    return default_training_backend_registry().create(
        backend_name,
        model=model,
        optimizer=optimizer,
        config=config,
        local_rank=local_rank,
    )


__all__ = [
    "TrainingBackendRegistry",
    "create_runtime_training_backend",
    "default_training_backend_registry",
]
