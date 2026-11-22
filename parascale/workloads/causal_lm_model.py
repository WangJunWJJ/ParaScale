# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午10:16
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Stable causal-LM model identity shared by training and artifact reloads."""

from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from parascale.data.text import tokenizer_fingerprint


class CausalLmModel(nn.Module):
    """Preserve the HF child and training's cache-free, metadata-tolerant forward."""

    __constants__ = ("tokenizer_identity",)

    def __init__(self, model: nn.Module, *, tokenizer_identity: str):
        super().__init__()
        self._tokenizer_identity = tokenizer_identity
        self.model = model
        self.config = model.config
        self.config.use_cache = False

    @property
    def tokenizer_identity(self) -> str:
        """Tokenizer semantics included in the strict artifact manifest."""
        return self._tokenizer_identity

    def forward(self, input_ids, labels=None, attention_mask=None, **metadata):
        return self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            use_cache=False,
        )


def load_causal_lm_model(
    model_path: str | Path,
    *,
    attention_backend: str = "sdpa",
    from_config: bool = False,
) -> CausalLmModel:
    """Load a local CPU FP32 model, or construct a weight-free reload template.

    Config-only construction never calls a pretrained model loader. The strict
    quantization loader supplies all artifact state after checking this template.
    """
    path = Path(model_path)
    if not path.is_dir():
        raise ValueError("causal_lm requires a local model directory")
    tokenizer = AutoTokenizer.from_pretrained(
        path, use_fast=True, local_files_only=True, trust_remote_code=False
    )
    identity = tokenizer_fingerprint(tokenizer)
    with torch.device("cpu"):
        if from_config:
            config = AutoConfig.from_pretrained(
                path, local_files_only=True, trust_remote_code=False
            )
            config.use_cache = False
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=False,
                torch_dtype=torch.float32,
                attn_implementation=attention_backend,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.float32,
                attn_implementation=attention_backend,
            )
    return CausalLmModel(model, tokenizer_identity=identity)
