# -*- coding: utf-8 -*-
# @Time : 2026/11/26 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Verified held-out causal-LM data without model or collective ownership."""

from __future__ import annotations

from parascale.data.text import TokenBatchCursor, TokenCorpus
from parascale.workloads.specs.causal_lm import CausalLmEvaluationSpec


def build_causal_lm_evaluation_cursor(
    spec: CausalLmEvaluationSpec,
    training_cursor: TokenBatchCursor,
    *,
    microbatches: int,
    vocab_size: int,
) -> TokenBatchCursor:
    """Build a fresh finite cursor with capacity for the evaluation budget.

    The caller limits iteration to max_batches full executor windows. Identity
    checks reject identical payloads/sources, not semantic or pretraining overlap.
    No training cursor state, models or distributed collectives are touched.
    """
    for name, value in (("microbatches", microbatches), ("vocab_size", vocab_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    training_manifest = training_cursor.corpus.manifest
    corpus = TokenCorpus(
        spec.data_path, expected_tokenizer=training_manifest["tokenizer"]
    )
    for key in ("sha256", "source_sha256"):
        digest = training_manifest.get(key)
        if digest and digest == corpus.manifest.get(key):
            raise ValueError(f"Held-out corpus matches training corpus {key}")
    if corpus.manifest["max_token_id"] >= vocab_size:
        raise ValueError("Evaluation token corpus exceeds the model vocabulary")

    cursor = TokenBatchCursor(
        corpus,
        batch_size=training_cursor.batch_size,
        sequence_length=training_cursor.sequence_length,
        rank=training_cursor.rank,
        world_size=training_cursor.world_size,
        seed=training_cursor.seed,
        pin_memory=training_cursor.pin_memory,
        shuffle_window_blocks=training_cursor.identity["shuffle_window_blocks"],
        shuffle=False,
        repeat=False,
    )
    required = spec.max_batches * microbatches
    if cursor.batches_per_epoch < required:
        raise ValueError(
            "Evaluation corpus cannot provide max_batches full executor windows: "
            f"requires {required} microbatches per rank, "
            f"has {cursor.batches_per_epoch}"
        )
    return cursor
