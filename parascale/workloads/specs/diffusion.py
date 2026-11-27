# -*- coding: utf-8 -*-
# @Time : 2026/11/24 下午9:18
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Explicit local Stable Diffusion execution and sampling policy."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class DiffusionInferenceSpec:
    model_path: str
    input_path: str
    output_dir: str
    device: str = "cuda:0"
    precision: str = "fp16"
    batch_size: int = 1
    height: int = 512
    width: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    seed: int = 42
    warmup_steps: int = 1
    max_requests: int | None = None
    max_line_bytes: int = 1048576
    compile_mode: str = "off"

    def __post_init__(self) -> None:
        for name in ("model_path", "input_path", "output_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty local path")
        if not isinstance(self.device, str) or not re.fullmatch(
            r"cpu|cuda(?::\d+)?", self.device
        ):
            raise ValueError("device must be cpu or cuda[:index]")
        for name, allowed in (
            ("precision", {"fp32", "fp16"}),
            ("compile_mode", {"off", "max-autotune"}),
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"Unsupported {name}: {value!r}")
        for name in (
            "batch_size",
            "height",
            "width",
            "num_inference_steps",
            "seed",
            "warmup_steps",
            "max_requests",
            "max_line_bytes",
        ):
            value = getattr(self, name)
            if name == "max_requests" and value is None:
                continue
            minimum = 0 if name in {"seed", "warmup_steps"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("height", "width"):
            if getattr(self, name) % 8:
                raise ValueError(f"{name} must be a multiple of 8")
        if self.seed >= 2**63:
            raise ValueError("seed must be less than 2**63")
        if (
            type(self.guidance_scale) not in {int, float}
            or not math.isfinite(self.guidance_scale)
            or self.guidance_scale < 1
        ):
            raise ValueError("guidance_scale must be finite and >= 1")
        if not isinstance(self.negative_prompt, str):
            raise ValueError("negative_prompt must be a string")
        if self.device == "cpu" and (
            self.precision != "fp32" or self.compile_mode != "off"
        ):
            raise ValueError(
                "CPU diffusion requires precision=fp32 and compile_mode=off"
            )
        if self.compile_mode != "off" and self.warmup_steps < 1:
            raise ValueError("compile_mode requires warmup_steps >= 1")

    @classmethod
    def from_config(cls, config: dict) -> DiffusionInferenceSpec:
        sections = {}
        for key in ("model", "data", "inference", "runtime"):
            value = config.get(key, {})
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object")
            sections[key] = value
        model, data, inference, runtime = (
            sections[key] for key in ("model", "data", "inference", "runtime")
        )
        if data.get("type") != "prompts_jsonl":
            raise ValueError("diffusion requires data.type=prompts_jsonl")
        allowed = {field.name for field in fields(cls)} - {
            "model_path",
            "input_path",
            "device",
        }
        unknown = set(inference) - allowed - {"workload"}
        if unknown:
            raise ValueError(f"Unknown diffusion options: {sorted(unknown)}")
        if "output_dir" not in inference:
            raise ValueError("diffusion requires inference.output_dir")
        accelerator = runtime.get("accelerator", "cuda")
        if not isinstance(accelerator, str) or accelerator not in {"cpu", "cuda"}:
            raise ValueError("diffusion runtime.accelerator must be cpu or cuda")
        device = runtime.get("device", "cpu" if accelerator == "cpu" else "cuda:0")
        if "accelerator" in runtime and not str(device).startswith(accelerator):
            raise ValueError("runtime device and accelerator disagree")
        return cls(
            model_path=model.get("pretrained_model_name_or_path", ""),
            input_path=data.get("path", ""),
            device=device,
            **{key: value for key, value in inference.items() if key in allowed},
        )
