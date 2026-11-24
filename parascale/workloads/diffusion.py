# -*- coding: utf-8 -*-
# @Time : 2026/11/24 下午9:18
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Local Stable Diffusion adapter with explicit sampling and compilation."""

import copy
import json
import os
from pathlib import Path

from parascale.data.text.prompts import iter_prompt_batches
from parascale.workloads.causal_lm_assets import file_sha256
from parascale.workloads.specs.diffusion import DiffusionInferenceSpec


def load_diffusion_pipeline(spec):
    """Load trusted local safetensors assets without replacing sampling policy."""
    import diffusers
    import torch
    import transformers

    if diffusers.__version__ != "0.35.1" or transformers.__version__ != "4.51.3":
        raise RuntimeError(
            "Diffusion inference requires Diffusers 0.35.1 and Transformers 4.51.3; install parascale[diffusion]"
        )
    root = Path(spec.model_path)
    index = root / "model_index.json"
    definition = (
        json.loads(index.read_text(encoding="utf-8")) if index.is_file() else None
    )
    if (
        not isinstance(definition, dict)
        or definition.get("_class_name") != "StableDiffusionPipeline"
    ):
        raise ValueError("diffusion requires a local StableDiffusionPipeline directory")
    for name, component in definition.items():
        if name.startswith("_") or not isinstance(component, list):
            continue
        if component == [None, None]:
            continue
        if len(component) != 2 or not all(
            isinstance(value, str) for value in component
        ):
            raise ValueError(f"Invalid diffusion component: {name}")
        library, _ = component
        folder = (root / name).resolve()
        if (
            library not in {"diffusers", "transformers", "stable_diffusion"}
            or not folder.is_relative_to(root.resolve())
            or (folder / f"{library}.py").is_file()
        ):
            raise ValueError(f"Unsupported custom diffusion component: {name}")
    if spec.compile_mode != "off" and torch.__version__.split("+")[0] != "2.9.1":
        raise RuntimeError(
            "Compiled diffusion requires Torch 2.9.1, Diffusers 0.35.1 and Transformers 4.51.3"
        )
    hashes = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".txt", ".safetensors"}
    }
    pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
        root,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype={"fp32": torch.float32, "fp16": torch.float16}[spec.precision],
    ).to(spec.device)
    pipeline.set_progress_bar_config(disable=True)
    for module in pipeline.components.values():
        if isinstance(module, torch.nn.Module):
            module.eval().requires_grad_(False)
    if spec.compile_mode != "off":
        pipeline.unet = torch.compile(
            pipeline.unet, mode=spec.compile_mode, fullgraph=True, dynamic=False
        )
    return pipeline, {
        "model_assets_sha256": hashes,
        "torch_version": str(torch.__version__),
        "diffusers_version": diffusers.__version__,
        "transformers_version": transformers.__version__,
        "scheduler": {
            "class": type(pipeline.scheduler).__name__,
            "config": dict(pipeline.scheduler.config),
        },
        "safety_checker_enabled": pipeline.safety_checker is not None,
    }


class DiffusionInferenceModel:
    """Single-owner pipeline; seeds belong to requests, not mutable RNG state."""

    def __init__(self, spec):
        self.spec = spec
        self._pipeline = None
        self._execution = {}
        self._closed = False

    def resolve_device(self):
        import torch

        device = torch.device(self.spec.device)
        if device.type == "cpu":
            return "cpu", lambda: None
        index = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
        torch.cuda.set_device(index)
        return f"cuda:{index}", lambda: torch.cuda

    def to(self, device):
        requested, _ = self.resolve_device()
        if str(device) not in {requested, self.spec.device}:
            raise ValueError("Diffusion adapter cannot change its configured device")
        return self.eval()

    def eval(self):
        if self._closed:
            raise RuntimeError("Diffusion model is closed")
        if self._pipeline is None:
            self.resolve_device()
            self._pipeline, metadata = load_diffusion_pipeline(self.spec)
            self._execution = {
                **metadata,
                "backend": "diffusers",
                "precision": self.spec.precision,
                "device": self.spec.device,
                "compile_mode": self.spec.compile_mode,
                "height": self.spec.height,
                "width": self.spec.width,
                "num_inference_steps": self.spec.num_inference_steps,
                "guidance_scale": self.spec.guidance_scale,
                "negative_prompt": self.spec.negative_prompt,
                "seed_policy": "(base seed + zero-based request ordinal) modulo 2**63; CPU generators",
                "input_sha256": file_sha256(self.spec.input_path),
            }
        return self

    @property
    def execution_metadata(self):
        return copy.deepcopy(self._execution)

    def prepare_batch(self, batch):
        self.eval()
        count = len(batch["prompts"])
        if (
            not 1 <= count <= self.spec.batch_size
            or len(batch["ids"]) != count
            or len(batch["seeds"]) != count
        ):
            raise ValueError("Invalid diffusion batch cardinality")
        tokenizer = self._pipeline.tokenizer
        encoded = tokenizer(
            batch["prompts"] + [self.spec.negative_prompt],
            truncation=False,
            padding=False,
        )
        if any(
            len(tokens) > tokenizer.model_max_length for tokens in encoded["input_ids"]
        ):
            raise ValueError(
                "Diffusion prompt exceeds tokenizer length; truncation is disabled"
            )
        return batch

    def generate(self, batch):
        import numpy as np
        import torch

        self.eval()
        with torch.inference_mode():
            output = self._pipeline(
                prompt=batch["prompts"],
                negative_prompt=[self.spec.negative_prompt] * len(batch["prompts"]),
                height=self.spec.height,
                width=self.spec.width,
                num_inference_steps=self.spec.num_inference_steps,
                guidance_scale=self.spec.guidance_scale,
                generator=[
                    torch.Generator(device="cpu").manual_seed(seed)
                    for seed in batch["seeds"]
                ],
                output_type="np",
                return_dict=True,
            )
        expected = (len(batch["prompts"]), self.spec.height, self.spec.width, 3)
        images = output.images
        if (
            not isinstance(images, np.ndarray)
            or images.shape != expected
            or not np.isfinite(images).all()
            or (images < 0).any()
            or (images > 1).any()
        ):
            raise ValueError(
                "Invalid diffusion image shape, cardinality or finite pixel range"
            )
        flags = output.nsfw_content_detected
        if flags is not None and len(flags) != len(images):
            raise ValueError("Diffusion image safety flags do not match images")
        return {
            "images": images,
            "ids": batch["ids"],
            "seeds": batch["seeds"],
            "safety_flags": flags,
        }

    def close(self):
        self._pipeline = None
        self._closed = True


def build_diffusion_inference_components(config):
    """Build a single-device job; dependencies and weights are loaded lazily."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "diffusion inference is single-device; distributed launch is unsupported"
        )
    spec = DiffusionInferenceSpec.from_config(config)

    def batches():
        ordinal = 0
        for batch in iter_prompt_batches(
            spec.input_path,
            batch_size=spec.batch_size,
            max_line_bytes=spec.max_line_bytes,
            max_requests=spec.max_requests,
        ):
            count = len(batch["prompts"])
            yield {
                **batch,
                "seeds": [
                    (spec.seed + ordinal + offset) % 2**63 for offset in range(count)
                ],
            }
            ordinal += count

    return DiffusionInferenceModel(spec), batches(), "diffusion_generation"
