# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Pretrained causal models wired into the standard training/runtime contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from parascale.data.text import TokenBatchCursor, TokenCorpus, tokenizer_fingerprint
from parascale.workloads.specs.causal_lm import CausalLmSpec

if TYPE_CHECKING:
    from parascale.parallel import CausalLMLoss, DistributedParallelConfig
    from parascale.parallel.llama import LlamaStage


def _load_source(
    spec: CausalLmSpec, *, data_rank: int, data_world_size: int
) -> tuple[Any, TokenBatchCursor]:
    import torch
    from transformers import AutoTokenizer

    from parascale.workloads.causal_lm_model import load_causal_lm_model

    if not Path(spec.model_path).is_dir():
        raise ValueError("causal_lm requires a local model directory")
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_path, use_fast=True, local_files_only=True, trust_remote_code=False
    )
    corpus = TokenCorpus(
        spec.data_path, expected_tokenizer=tokenizer_fingerprint(tokenizer)
    )
    batches = TokenBatchCursor(
        corpus,
        batch_size=spec.batch_size,
        sequence_length=spec.sequence_length,
        rank=data_rank,
        world_size=data_world_size,
        seed=spec.data_seed,
        shuffle=spec.shuffle,
        repeat=spec.repeat,
        pin_memory=torch.cuda.is_available(),
        shuffle_window_blocks=spec.shuffle_window_blocks,
    )
    torch.manual_seed(spec.seed)
    hf_model = load_causal_lm_model(
        spec.model_path, attention_backend=spec.attention_backend
    ).model
    if (
        corpus.manifest["max_token_id"]
        >= hf_model.get_input_embeddings().num_embeddings
    ):
        raise ValueError("Token corpus exceeds the model vocabulary")
    maximum = getattr(hf_model.config, "max_position_embeddings", None)
    if maximum is not None and spec.sequence_length > maximum:
        raise ValueError("sequence_length exceeds model max_position_embeddings")
    if spec.activation_checkpointing:
        hf_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return hf_model, batches


def build_causal_lm_stage_components(
    config: dict[str, Any],
    parallel: DistributedParallelConfig,
    *,
    rank: int,
    world_size: int,
) -> tuple[LlamaStage, TokenBatchCursor, CausalLMLoss]:
    """Build a local CPU Llama stage and data shared only within its DP replica.

    Args:
        config: Causal-LM model/data configuration, as used by ordinary training.
        parallel: Validated DistributedParallelConfig defining the global grid.
        rank: Explicit global rank; launcher environment is not read or changed.
        world_size: Actual launcher size, which must equal the configured grid.

    Returns:
        Local LlamaStage, DP-coordinate TokenBatchCursor and summed CausalLMLoss.
        No optimizer is constructed: initialize the distributed executor first,
        then bind the optimizer to its final local parameters. This builder does
        not implement device execution, checkpointing or the train CLI backend.

    Raises:
        ValueError: Invalid launcher, unsupported stage/model or inconsistent data.
        TypeError: The topology is not a DistributedParallelConfig.
        RuntimeError: The installed Transformers version is not stage-qualified.
    """
    from parascale.parallel import CausalLMLoss, DistributedParallelConfig

    if not isinstance(parallel, DistributedParallelConfig):
        raise TypeError("parallel must be a DistributedParallelConfig")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    parallel.validate_world_size(world_size)
    replica, pipeline, _ = parallel.coordinates(rank)
    spec = CausalLmSpec.from_config(config)
    if spec.activation_checkpointing:
        raise ValueError("Llama stages do not support activation_checkpointing")

    from parascale.parallel.llama import LlamaStage

    source, batches = _load_source(
        spec, data_rank=replica, data_world_size=parallel.dp_size
    )
    stage = LlamaStage(
        source.train(), pipeline, parallel.pp_size, tp_size=parallel.tp_size
    )
    effective_config = json.loads(source.config.to_json_string(use_diff=False))
    for key in ("_name_or_path", "_commit_hash"):
        effective_config.pop(key, None)
    # HF serialization omits the runtime-selected attention implementation.
    effective_config["_attn_implementation"] = source.config._attn_implementation
    stage.effective_model_config = json.loads(
        json.dumps(effective_config, allow_nan=False)
    )
    del source
    return stage, batches, CausalLMLoss(stage.vocab_size)


def build_causal_lm_components(config: dict[str, Any]):
    from parascale.optimizers import build_optimizer
    from parascale.workloads.causal_lm_model import (
        CausalLmModel,
        causal_lm_config_identity,
    )

    spec = CausalLmSpec.from_config(config)
    hf_model, batches = _load_source(
        spec,
        data_rank=int(os.environ.get("RANK", "0")),
        data_world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )

    model = CausalLmModel(
        hf_model, tokenizer_identity=batches.corpus.manifest["tokenizer"]
    )
    model.train()

    def loss_fn(output, batch):
        return output.loss

    loss_fn.normalization_weight = lambda batch: batch["tokens"]
    loss_fn.objective_metadata = {
        "type": "causal_next_token_v1",
        "model_config_sha256": causal_lm_config_identity(hf_model.config),
        "tokenizer_identity": model.tokenizer_identity,
        "attention_backend": spec.attention_backend,
        "packing": batches.corpus.manifest["document_policy"],
    }
    return model, build_optimizer(model, config), batches, loss_fn
