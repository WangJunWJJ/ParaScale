# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午10:16
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Explicit, bounded single-device causal generation configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class CausalLmInferenceSpec:
    model_path: str
    input_path: str
    device: str = "cuda:0"
    execution: str = "float"
    precision: str = "bf16"
    artifact_path: str | None = None
    output_jsonl: str | None = None
    batch_size: int = 1
    max_line_bytes: int = 1048576
    max_input_tokens: int = 256
    max_new_tokens: int = 128
    max_requests: int | None = None
    max_outputs: int = 32
    warmup_steps: int = 1
    attention_policy: str = "auto"
    cache_implementation: str = "dynamic"
    compile_mode: str = "off"

    def __post_init__(self) -> None:
        for name in ("model_path", "input_path", "artifact_path", "output_jsonl"):
            value = getattr(self, name)
            if name in {"artifact_path", "output_jsonl"} and value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty local path")
        if not isinstance(self.device, str) or not re.fullmatch(
            r"cpu|cuda(?::\d+)?", self.device
        ):
            raise ValueError("device must be cpu or cuda[:index]")
        for name in (
            "batch_size",
            "max_line_bytes",
            "max_input_tokens",
            "max_new_tokens",
            "max_requests",
            "max_outputs",
            "warmup_steps",
        ):
            value = getattr(self, name)
            if name == "max_requests" and value is None:
                continue
            minimum = 0 if name in {"max_outputs", "warmup_steps"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, allowed in (
            ("execution", {"float", "reference", "gemlite_int8"}),
            ("precision", {"fp32", "bf16"}),
            ("attention_policy", {"auto", "math"}),
            ("cache_implementation", {"dynamic", "static"}),
            ("compile_mode", {"off", "max-autotune"}),
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"Unsupported {name}: {value!r}")
        if (self.execution == "float") != (self.artifact_path is None):
            raise ValueError(
                "artifact_path is required only for reference/gemlite_int8 execution"
            )
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("CPU generation requires precision=fp32")
        if self.execution == "gemlite_int8" and (
            self.device == "cpu" or self.precision != "bf16"
        ):
            raise ValueError("gemlite_int8 requires CUDA BF16 precision")
        if self.compile_mode != "off" and self.cache_implementation != "static":
            raise ValueError("compile_mode requires static cache")
        if self.cache_implementation == "static":
            if self.batch_size != 1:
                raise ValueError("static cache currently requires batch_size=1")
            if self.device == "cpu" or self.precision != "bf16":
                raise ValueError("static cache currently requires CUDA BF16 precision")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CausalLmInferenceSpec:
        """Parse workload-owned fields, rejecting unknown generation settings."""
        sections = {}
        for key in ("model", "data", "inference", "runtime"):
            value = config.get(key, {})
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object")
            sections[key] = value
        model, data = sections["model"], sections["data"]
        inference, runtime = sections["inference"], sections["runtime"]
        if data.get("type") != "prompts_jsonl":
            raise ValueError("causal_lm inference requires data.type=prompts_jsonl")
        excluded = {"model_path", "input_path", "device"}
        allowed = {field.name for field in fields(cls)} - excluded
        unknown = set(inference) - allowed - {"workload"}
        if unknown:
            raise ValueError(f"Unknown causal_lm inference options: {sorted(unknown)}")
        accelerator = runtime.get("accelerator", "cuda")
        if not isinstance(accelerator, str) or accelerator not in {"cpu", "cuda"}:
            raise ValueError("causal_lm runtime.accelerator must be cpu or cuda")
        device = runtime.get("device", "cpu" if accelerator == "cpu" else "cuda:0")
        if "accelerator" in runtime and not str(device).startswith(accelerator):
            raise ValueError("runtime device and accelerator disagree")
        return cls(
            model_path=model.get("pretrained_model_name_or_path", ""),
            input_path=data.get("path", ""),
            device=device,
            **{key: value for key, value in inference.items() if key in allowed},
        )
