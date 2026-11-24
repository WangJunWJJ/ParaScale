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


class CausalLMInferenceTaskAdapter(TextInferenceTaskAdapter):
    """Generation records carry actual request and post-EOS token counts."""

    name = "causal_lm_generation"

    def prepare_model_batch(self, model: Any, batch: Any) -> Any:
        prepare = getattr(model, "prepare_batch", None)
        return prepare(batch) if callable(prepare) else batch

    def output_metric_counts(self, output: Any) -> Dict[str, int]:
        counts = output.get("counts") if isinstance(output, dict) else None
        keys = {"requests", "prompt_tokens", "generated_tokens"}
        if (
            not isinstance(counts, dict)
            or set(counts) != keys
            or any(type(value) is not int or value < 0 for value in counts.values())
        ):
            raise ValueError("Invalid causal-LM output counts")
        records = output.get("records")
        if not isinstance(records, list) or len(records) != counts["requests"]:
            raise ValueError("Causal-LM output counts do not match records")
        return {**counts, "tokens": counts["generated_tokens"]}


__all__ = ["CausalLMInferenceTaskAdapter", "TextInferenceTaskAdapter"]
