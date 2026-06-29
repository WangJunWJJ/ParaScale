# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Inference runtime independent from third-party training engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from parascale.core import (
    CollectiveBackend,
    CpuDeviceBackend,
    DeviceBackend,
    MockCollectiveBackend,
)


@dataclass
class ServeState:
    initialized: bool = False
    requests: int = 0
    last_latency_ms: float = 0.0


@dataclass
class ServeEngine:
    config: Any = field(default_factory=dict)
    device_backend: DeviceBackend = field(default_factory=CpuDeviceBackend)
    collective: CollectiveBackend = field(default_factory=MockCollectiveBackend)
    state: ServeState = field(default_factory=ServeState)
    model: Any = None
    mock_mode: bool = False

    def initialize(self, world_size: int = 1) -> "ServeEngine":
        self.collective.init_process_group(world_size=max(1, int(world_size)), rank=0)
        self.state.initialized = True
        return self

    def load_model(
        self, model: Any = None, checkpoint: Any = None, *, mock: bool = False
    ) -> "ServeEngine":
        self.model = model if model is not None else checkpoint
        self.mock_mode = bool(mock or self.model == "mock")
        return self

    def generate(self, requests: Any) -> Dict[str, Any]:
        self.record_request()
        if self.mock_mode:
            return {
                "requests": requests,
                "outputs": self._mock_outputs(requests, "generated"),
                "mode": "mock",
            }
        self._require_model("generate")
        generate = getattr(self.model, "generate", None)
        if callable(generate):
            return {
                "requests": requests,
                "outputs": generate(requests),
                "mode": "model",
            }
        if callable(self.model):
            return {
                "requests": requests,
                "outputs": self.model(requests),
                "mode": "callable",
            }
        raise RuntimeError(
            "Loaded model does not provide generate(requests) or __call__(requests)."
        )

    def embed(self, requests: Any) -> Dict[str, Any]:
        self.record_request()
        if self.mock_mode:
            return {
                "requests": requests,
                "embeddings": self._mock_outputs(requests, []),
                "mode": "mock",
            }
        self._require_model("embed")
        embed = getattr(self.model, "embed", None)
        if callable(embed):
            return {
                "requests": requests,
                "embeddings": embed(requests),
                "mode": "model",
            }
        raise RuntimeError("Loaded model does not provide embed(requests).")

    def prefill(self, batch: Any) -> Dict[str, Any]:
        if self.mock_mode or self.model is None:
            return {"batch": batch, "state": "prefilled", "mode": "mock"}
        prefill = getattr(self.model, "prefill", None)
        if callable(prefill):
            return {"batch": batch, "state": prefill(batch), "mode": "model"}
        return {"batch": batch, "state": "prefilled", "mode": "fallback"}

    def decode(self, batch: Any) -> Dict[str, Any]:
        if self.mock_mode or self.model is None:
            return {"batch": batch, "state": "decoded", "mode": "mock"}
        decode = getattr(self.model, "decode", None)
        if callable(decode):
            return {"batch": batch, "state": decode(batch), "mode": "model"}
        return {"batch": batch, "state": "decoded", "mode": "fallback"}

    def record_request(self, latency_ms: float = 0.0) -> ServeState:
        self.state.requests += 1
        self.state.last_latency_ms = max(0.0, float(latency_ms))
        return self.state

    def shutdown(self) -> None:
        self.collective.shutdown()
        self.state.initialized = False

    def _require_model(self, operation: str) -> None:
        if self.model is None:
            raise RuntimeError(
                f"ServeEngine.{operation} requires load_model(...), or load_model(..., mock=True)."
            )

    @staticmethod
    def _mock_outputs(requests: Any, value: Any) -> list[Any]:
        if isinstance(requests, (list, tuple)):
            return [value for _ in requests]
        return [value]


class InferenceEngine(ServeEngine):
    """Production-facing inference runtime entrypoint."""


__all__ = ["InferenceEngine", "ServeEngine", "ServeState"]
