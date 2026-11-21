# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Explicit Llama stages backed by the pinned Hugging Face implementation."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


class LlamaStage(nn.Module):
    """Own a contiguous range of Llama layers for packed causal training.

    Args:
        source: A Transformers 4.51.3 LlamaForCausalLM with loaded FP32 weights.
            Modules are reused, not copied. After partitioning, release the source
            and do not use it as an independent reference or optimizer owner.
        stage_index: Zero-based pipeline stage index.
        stage_count: One or two pipeline stages.
        tp_size: One or two tensor-parallel ranks; the executor applies the plan.

    The first stage consumes int64 token blocks; later stages consume FP32 hidden
    states. The last stage returns full-vocabulary logits. All blocks start at
    position zero and use causal attention without padding, cache or custom masks.
    Cross-stage weight ties are rejected instead of silently untied.
    """

    def __init__(
        self,
        source: Any,
        stage_index: int = 0,
        stage_count: int = 1,
        *,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        import transformers
        from transformers import LlamaForCausalLM

        if transformers.__version__ != "4.51.3":
            raise RuntimeError("LlamaStage requires tested transformers==4.51.3")
        if type(source) is not LlamaForCausalLM:
            raise ValueError("source must be an unwrapped LlamaForCausalLM")
        if (
            type(stage_count) is not int
            or stage_count not in (1, 2)
            or type(stage_index) is not int
            or not 0 <= stage_index < stage_count
        ):
            raise ValueError("stage_index must select one of one or two stages")
        if type(tp_size) is not int or tp_size not in (1, 2):
            raise ValueError("tp_size must be one or two")
        config = source.config
        if len(source.model.layers) != config.num_hidden_layers or (
            config.num_hidden_layers < stage_count
        ):
            raise ValueError(
                "each stage must own at least one configured decoder layer"
            )
        if config.attention_dropout != 0:
            raise ValueError("attention_dropout must be zero for this stage contract")
        if getattr(config, "pretraining_tp", 1) != 1:
            raise ValueError("pretraining_tp must be one; use the explicit TP plan")
        if config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("LlamaStage supports eager or sdpa attention")
        if any(p.dtype != torch.float32 or p.is_meta for p in source.parameters()):
            raise ValueError("source parameters must be materialized float32 tensors")
        embedding, head = source.model.embed_tokens.weight, source.lm_head.weight
        if stage_count > 1 and (
            config.tie_word_embeddings or embedding.data_ptr() == head.data_ptr()
        ):
            raise ValueError(
                "tied embedding/head across pipeline stages is unsupported"
            )
        for field in (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
        ):
            if getattr(config, field) % tp_size:
                raise ValueError(f"{field} must be divisible by tp_size")
        self.stage_index = stage_index
        self.stage_count = stage_count
        self.tp_size = tp_size
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.attention_implementation = config._attn_implementation
        self.model = nn.Module()
        if stage_index == 0:
            self.model.embed_tokens = source.model.embed_tokens
        start = config.num_hidden_layers * stage_index // stage_count
        end = config.num_hidden_layers * (stage_index + 1) // stage_count
        self.model.layers = nn.ModuleDict(
            {str(index): source.model.layers[index] for index in range(start, end)}
        )
        self.model.rotary_emb = copy.deepcopy(source.model.rotary_emb)
        if stage_index == stage_count - 1:
            self.model.norm = source.model.norm
            self.lm_head = source.lm_head
        self.train(source.training)

    def validate_inputs(self, inputs: torch.Tensor) -> None:
        """Validate first-stage token indices during coordinated trainer preflight."""
        if self.stage_index != 0:
            return
        if inputs.ndim != 2 or inputs.dtype != torch.int64:
            raise ValueError("first stage requires int64 [batch, sequence] tokens")
        if ((inputs < 0) | (inputs >= self.vocab_size)).any():
            raise ValueError("input token is outside the configured vocabulary")

    def tensor_parallel_plan(self) -> dict[str, Any]:
        """Shard projections and share input-gradient reductions within each block."""
        from torch.distributed.tensor import Replicate
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            PrepareModuleInput,
            RowwiseParallel,
        )

        if self.tp_size == 1:
            return {}
        plan = {}
        for index in self.model.layers:
            prefix = f"model.layers.{index}"
            # Keep one DTensor input so branch gradients combine before reduction.
            plan[f"{prefix}.self_attn"] = PrepareModuleInput(
                input_kwarg_layouts={"hidden_states": Replicate()},
                desired_input_kwarg_layouts={"hidden_states": Replicate()},
            )
            plan[f"{prefix}.mlp"] = PrepareModuleInput(
                input_layouts=Replicate(), desired_input_layouts=Replicate()
            )
            for projection in ("q_proj", "k_proj", "v_proj"):
                plan[f"{prefix}.self_attn.{projection}"] = ColwiseParallel()
            plan[f"{prefix}.self_attn.o_proj"] = RowwiseParallel()
            for projection in ("gate_proj", "up_proj"):
                plan[f"{prefix}.mlp.{projection}"] = ColwiseParallel()
            plan[f"{prefix}.mlp.down_proj"] = RowwiseParallel()
        return plan

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return local hidden states or final logits for an unpadded causal block."""
        if self.stage_index == 0:
            if inputs.ndim != 2 or inputs.dtype != torch.int64:
                raise ValueError("first stage requires int64 [batch, sequence] tokens")
            hidden = self.model.embed_tokens(inputs)
        else:
            if (
                inputs.ndim != 3
                or inputs.dtype != torch.float32
                or inputs.shape[-1] != self.hidden_size
            ):
                raise ValueError(
                    "later stage requires float32 [batch, sequence, hidden]"
                )
            hidden = inputs
        length = hidden.shape[1]
        positions = torch.arange(length, device=hidden.device).unsqueeze(0)
        mask = None
        if self.attention_implementation == "eager":
            mask = torch.full(
                (length, length), torch.finfo(hidden.dtype).min, device=hidden.device
            ).triu(1)[None, None]
        rotary = self.model.rotary_emb(hidden, positions)
        for layer in self.model.layers.values():
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=positions,
                position_embeddings=rotary,
                use_cache=False,
            )[0]
        if self.stage_index == self.stage_count - 1:
            return self.lm_head(self.model.norm(hidden))
        return hidden
