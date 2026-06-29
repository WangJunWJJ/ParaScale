# -*- coding: utf-8 -*-
# @Time : 2026/6/29 下午4:06
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Inference task adapter contract and shared helpers."""

from __future__ import annotations

from typing import Any, Dict, Protocol, Sequence


class InferenceTaskAdapter(Protocol):
    """Customize task behavior without coupling it to the runtime loop."""

    name: str

    def prepare_batch(self, batch: Any) -> Any: ...

    def predict(self, model: Any, batch: Any) -> Any: ...

    def postprocess(self, output: Any) -> Any: ...

    def metric_counts(self, batch: Any) -> Dict[str, int]: ...

    def execution_hints(self) -> Dict[str, Any]: ...


def invoke_model(model: Any, batch: Any, methods: Sequence[str]) -> Any:
    for method_name in methods:
        method = getattr(model, method_name, None)
        if callable(method):
            return method(batch)
    if callable(model):
        return model(batch)
    supported = ", ".join(methods)
    raise RuntimeError(
        f"inference model must implement one of [{supported}] or __call__"
    )


def count_batch_value(batch: Any, key: str) -> int:
    if isinstance(batch, dict) and key in batch:
        return int(batch.get(key) or 0)
    if isinstance(batch, list):
        return sum(count_batch_value(item, key) for item in batch)
    return 0


__all__ = ["InferenceTaskAdapter", "count_batch_value", "invoke_model"]
