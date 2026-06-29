# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:26
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Multimodal inference task adapter."""

from __future__ import annotations

from typing import Any, Dict

from .base import count_batch_value, invoke_model


class MultimodalInferenceTaskAdapter:
    name = "multimodal"

    def prepare_batch(self, batch: Any) -> Any:
        return batch

    def predict(self, model: Any, batch: Any) -> Any:
        return invoke_model(model, batch, ("generate", "embed", "predict"))

    def postprocess(self, output: Any) -> Any:
        return output

    def metric_counts(self, batch: Any) -> Dict[str, int]:
        return {
            "images": count_batch_value(batch, "num_images"),
            "image_text_pairs": count_batch_value(batch, "num_pairs"),
            "tokens": count_batch_value(batch, "tokens"),
        }

    def execution_hints(self) -> Dict[str, Any]:
        return {
            "task": self.name,
            "preferred_methods": ("generate", "embed", "predict"),
        }


__all__ = ["MultimodalInferenceTaskAdapter"]
