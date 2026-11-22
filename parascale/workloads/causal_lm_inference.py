# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午10:28
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Local Llama inference with strict portable artifact loading."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from parascale.data.text.prompts import iter_prompt_batches
from parascale.runtime.inference.attention import attention_policy
from parascale.runtime.inference.generation import generate_causal_tokens
from parascale.workloads.specs.causal_lm_inference import CausalLmInferenceSpec


def generation_output(
    sequences: list[list[int]],
    *,
    input_width: int,
    prompt_lengths: list[int],
    ids: list[str],
    tokenizer: Any,
    eos_ids: set[int],
) -> dict:
    """Decode bounded continuations, counting first EOS but never batch fill."""
    if not (len(sequences) == len(prompt_lengths) == len(ids)):
        raise ValueError("Generation output cardinality does not match requests")
    records = []
    for identifier, sequence, prompt_length in zip(ids, sequences, prompt_lengths):
        continuation = sequence[input_width:]
        finish_reason = "length"
        for position, token in enumerate(continuation):
            if token in eos_ids:
                continuation = continuation[: position + 1]
                finish_reason = "eos"
                break
        records.append(
            {
                "id": identifier,
                "text": tokenizer.decode(continuation, skip_special_tokens=True),
                "generated_token_ids": continuation,
                "prompt_tokens": prompt_length,
                "generated_tokens": len(continuation),
                "finish_reason": finish_reason,
            }
        )
    return {
        "records": records,
        "counts": {
            "requests": len(records),
            "prompt_tokens": sum(prompt_lengths),
            "generated_tokens": sum(row["generated_tokens"] for row in records),
        },
    }


class CausalLmInferenceModel:
    """Single-owner adapter; use one isolated worker for each model instance.

    Loading/fusion is lazy and happens once. Prepared batches contain CPU token
    tensors for the runner to transfer outside prediction timing. Outputs never
    retain logits, GPU tensors or KV caches. No concurrent mutation is supported.
    """

    def __init__(self, spec: CausalLmInferenceSpec):
        self.spec = spec
        self._model = None
        self._tokenizer = None
        self._execution = {}
        self._closed = False

    @property
    def execution_metadata(self) -> dict:
        return copy.deepcopy(self._execution)

    def resolve_device(self):
        """Resolve exactly the requested device, failing instead of falling back."""
        import torch

        device = torch.device(self.spec.device)
        if device.type == "cuda":
            index = 0 if device.index is None else device.index
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
            torch.cuda.set_device(index)
            return f"cuda:{index}", lambda: torch.cuda
        return "cpu", lambda: None

    def to(self, device):
        requested, _ = self.resolve_device()
        if str(device) not in {requested, self.spec.device}:
            raise ValueError("Inference adapter cannot change its configured device")
        self._load()
        return self

    def eval(self):
        self._load()
        self._model.eval()
        return self

    def _load(self):
        if self._closed:
            raise RuntimeError("Inference model is closed")
        if self._model is not None:
            return
        import torch
        import transformers
        from transformers import AutoConfig, AutoTokenizer

        from parascale.data.text import tokenizer_fingerprint
        from parascale.workloads.causal_lm_assets import file_sha256, model_asset_hashes
        from parascale.workloads.causal_lm_model import load_causal_lm_model

        device, _ = self.resolve_device()
        root = Path(self.spec.model_path)
        if not root.is_dir():
            raise ValueError("causal_lm inference requires a local model directory")
        config = AutoConfig.from_pretrained(
            root, local_files_only=True, trust_remote_code=False
        )
        if config.model_type != "llama":
            raise ValueError("causal_lm inference currently requires a Llama model")
        if (
            self.spec.max_input_tokens + self.spec.max_new_tokens
            > config.max_position_embeddings
        ):
            raise ValueError(
                "Generation context budget exceeds max_position_embeddings"
            )
        if (
            self.spec.cache_implementation == "static"
            and transformers.__version__ != "4.51.3"
        ):
            raise RuntimeError(
                "Static-cache generation requires qualified Transformers 4.51.3"
            )
        assets = model_asset_hashes(
            root, include_weights=self.spec.artifact_path is None
        )
        model = load_causal_lm_model(
            root, from_config=self.spec.artifact_path is not None
        )
        artifact = None
        if self.spec.artifact_path is not None:
            from parascale.quantization import load_quantized_model

            digest = file_sha256(self.spec.artifact_path)
            model, quantization, _ = load_quantized_model(
                self.spec.artifact_path, model
            )
            if file_sha256(self.spec.artifact_path) != digest:
                raise ValueError("Quantized artifact changed during loading")
            artifact = {"sha256": digest, **quantization.to_dict()}
        tokenizer = AutoTokenizer.from_pretrained(
            root,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        identity = tokenizer_fingerprint(tokenizer)
        if identity != model.tokenizer_identity:
            raise ValueError("Tokenizer changed during model loading")
        vocabulary_size = model.model.get_input_embeddings().num_embeddings
        if max(tokenizer.get_vocab().values()) >= vocabulary_size:
            raise ValueError("Tokenizer exceeds model vocabulary")
        eos = model.config.eos_token_id
        eos = tokenizer.eos_token_id if eos is None else eos
        eos_ids = [eos] if type(eos) is int else eos
        if (
            not isinstance(eos_ids, list)
            or not eos_ids
            or any(
                type(value) is not int or not 0 <= value < vocabulary_size
                for value in eos_ids
            )
        ):
            raise ValueError("Model requires valid EOS token IDs")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = eos_ids[0]
        tokenizer.padding_side = "left"
        dtype = torch.float32 if self.spec.precision == "fp32" else torch.bfloat16
        model = model.eval().to(device=device, dtype=dtype)
        model.requires_grad_(False)
        fusion = None
        if self.spec.execution == "gemlite_int8":
            from parascale.quantization import prepare_quantized_inference_model

            model, fusion = prepare_quantized_inference_model(model, inplace=True)
        self._model, self._tokenizer = model, tokenizer
        self._eos_ids = set(eos_ids)
        self._execution = {
            "backend": self.spec.execution,
            "device": device,
            "precision": self.spec.precision,
            "tokenizer_identity": identity,
            "model_assets_sha256": assets,
            "artifact": artifact,
            "fusion": fusion,
            "torch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
            "effective_model_config": json.loads(
                model.config.to_json_string(use_diff=False)
            ),
            "input_sha256": file_sha256(self.spec.input_path),
            "generation_config": None,
        }

    def prepare_batch(self, batch: dict) -> dict:
        """Tokenize without truncation; enforce capacity before device transfer."""
        self._load()
        prompts, ids = batch["prompts"], batch["ids"]
        if (
            not prompts
            or len(prompts) != len(ids)
            or len(prompts) > self.spec.batch_size
        ):
            raise ValueError("Invalid prompt batch cardinality")
        static = self.spec.cache_implementation == "static"
        encoded = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length" if static else True,
            truncation=False,
            return_token_type_ids=False,
            **({"max_length": self.spec.max_input_tokens} if static else {}),
        )
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        if any(length < 1 or length > self.spec.max_input_tokens for length in lengths):
            raise ValueError("Prompt length exceeds max_input_tokens or is empty")
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "ids": ids,
            "prompt_lengths": lengths,
        }

    def generate(self, batch: dict) -> dict:
        """Generate a prepared batch with normal EOS stopping and bounded output."""
        import torch

        self._load()

        def observe_configuration(value):
            self._execution["generation_config"] = value

        observer = (
            observe_configuration
            if self._execution["generation_config"] is None
            else None
        )
        with (
            torch.inference_mode(),
            attention_policy(self.spec.attention_policy) as policy,
        ):
            sequences = generate_causal_tokens(
                self._model.model,
                batch["input_ids"],
                batch["attention_mask"],
                self.spec.max_new_tokens,
                self._tokenizer.pad_token_id,
                eos_token_id=sorted(self._eos_ids),
                static_cache=self.spec.cache_implementation == "static",
                compile_static=self.spec.compile_mode != "off",
                configuration_observer=observer,
            )
            self._execution["attention_policy"] = policy
            return generation_output(
                sequences.cpu().tolist(),
                input_width=batch["input_ids"].shape[1],
                prompt_lengths=batch["prompt_lengths"],
                ids=batch["ids"],
                tokenizer=self._tokenizer,
                eos_ids=self._eos_ids,
            )

    def close(self):
        """Release model-owned tensors and caches after an isolated inference job."""
        self._model = self._tokenizer = None
        self._closed = True


def build_causal_lm_inference_components(config: dict):
    """Build a lazy single-device model and bounded prompt iterable."""
    for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        if int(os.environ.get(name, "1")) != 1:
            raise ValueError(
                "causal_lm inference is single-device, not a distributed launch"
            )
    spec = CausalLmInferenceSpec.from_config(config)
    batches = iter_prompt_batches(
        spec.input_path,
        batch_size=spec.batch_size,
        max_line_bytes=spec.max_line_bytes,
        max_requests=spec.max_requests,
    )
    return CausalLmInferenceModel(spec), batches, "causal_lm_generation"
