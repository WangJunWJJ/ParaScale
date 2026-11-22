# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Pretrained causal models wired into the standard training/runtime contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from parascale.data.text import TokenBatchCursor, TokenCorpus, tokenizer_fingerprint
from parascale.workloads.specs.causal_lm import CausalLmSpec


def build_causal_lm_components(config: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from parascale.optimizers import build_optimizer

    spec = CausalLmSpec.from_config(config)
    if not Path(spec.model_path).is_dir():
        raise ValueError("causal_lm requires a local model directory")
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_path, use_fast=True, local_files_only=True, trust_remote_code=False
    )
    corpus = TokenCorpus(
        spec.data_path, expected_tokenizer=tokenizer_fingerprint(tokenizer)
    )
    torch.manual_seed(spec.seed)
    hf_model = AutoModelForCausalLM.from_pretrained(
        spec.model_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        attn_implementation=spec.attention_backend,
    )
    if (
        corpus.manifest["max_token_id"]
        >= hf_model.get_input_embeddings().num_embeddings
    ):
        raise ValueError("Token corpus exceeds the model vocabulary")
    maximum = getattr(hf_model.config, "max_position_embeddings", None)
    if maximum is not None and spec.sequence_length > maximum:
        raise ValueError("sequence_length exceeds model max_position_embeddings")
    hf_model.config.use_cache = False
    if spec.activation_checkpointing:
        hf_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    class CausalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = hf_model
            self.config = hf_model.config

        def forward(self, input_ids, labels=None, attention_mask=None, **metadata):
            return self.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                use_cache=False,
            )

    model = CausalModel()
    model.train()
    batches = TokenBatchCursor(
        corpus,
        batch_size=spec.batch_size,
        sequence_length=spec.sequence_length,
        rank=int(os.environ.get("RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        seed=spec.data_seed,
        shuffle=spec.shuffle,
        repeat=spec.repeat,
        pin_memory=torch.cuda.is_available(),
        shuffle_window_blocks=spec.shuffle_window_blocks,
    )

    def loss_fn(output, batch):
        return output.loss

    loss_fn.normalization_weight = lambda batch: batch["num_tokens"]
    effective_config = hf_model.config.to_dict()
    effective_config.pop("_name_or_path", None)
    loss_fn.objective_metadata = {
        "type": "causal_next_token_v1",
        "model_config_sha256": hashlib.sha256(
            json.dumps(effective_config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "attention_backend": spec.attention_backend,
        "packing": corpus.manifest["document_policy"],
    }
    return model, build_optimizer(model, config), batches, loss_fn
