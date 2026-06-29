# -*- coding: utf-8 -*-
# @Time : 2026/6/29 下午4:06
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Text inference task adapter."""

from __future__ import annotations

from typing import Any, Dict

from .base import count_batch_value, invoke_model


class TextInferenceTaskAdapter:
    name = "text"

    def prepare_batch(self, batch: Any) -> Any:
        return batch

    def predict(self, model: Any, batch: Any) -> Any:
        return invoke_model(model, batch, ("generate", "predict"))

    def postprocess(self, output: Any) -> Any:
        return output

    def metric_counts(self, batch: Any) -> Dict[str, int]:
        return {"tokens": count_batch_value(batch, "tokens")}

    def execution_hints(self) -> Dict[str, Any]:
        return {"task": self.name, "preferred_methods": ("generate", "predict")}


__all__ = ["TextInferenceTaskAdapter"]
