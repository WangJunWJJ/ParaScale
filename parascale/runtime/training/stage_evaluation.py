# -*- coding: utf-8 -*-
# @Time : 2026/11/26 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded held-out block NLL, independent of training progress and timing."""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.distributed as dist

from .coordination import coordinated_call
from .random_state import preserve_rng_state


def evaluate_stage(
    trainer: Any, cursor: Any, *, max_batches: int, global_step: int
) -> dict[str, Any]:
    """Evaluate a deterministic prefix; restore the independent cursor on exit."""
    microbatches = trainer.config.microbatches

    def preflight():
        if type(max_batches) is not int or max_batches < 1:
            raise ValueError("evaluation max_batches must be a positive integer")
        if type(global_step) is not int or global_step < 0:
            raise ValueError("evaluation global_step must be a nonnegative integer")
        if cursor.shuffle or cursor.repeat:
            raise ValueError("evaluation cursor must be unshuffled and finite")
        if cursor.world_size != trainer.config.dp_size:
            raise ValueError("evaluation cursor must shard only across DP")
        if max_batches * microbatches > cursor.batches_per_epoch:
            raise ValueError("evaluation corpus cannot provide all requested windows")
        return {
            "corpus_identity": cursor.corpus.identity,
            "max_batches": max_batches,
            "global_step": global_step,
            "microbatches": microbatches,
            "dp_size": cursor.world_size,
            "batch_size": cursor.batch_size,
            "sequence_length": cursor.sequence_length,
        }

    identity = coordinated_call(
        preflight, phase="evaluation plan", identity=lambda x: x
    )
    snapshot = cursor.state_dict()
    total_loss, scored_tokens = 0.0, 0

    def sync():
        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)

    with preserve_rng_state():
        try:
            coordinated_call(
                lambda: cursor.load_state_dict({**snapshot, "position": 0}),
                phase="evaluation cursor reset",
            )
            iterator = iter(cursor)
            sync()
            started = time.perf_counter()
            for _ in range(max_batches):

                def window():
                    chunks = [next(iterator) for _ in range(microbatches)]
                    return (
                        torch.cat([chunk["input_ids"] for chunk in chunks]).to(
                            trainer.device
                        ),
                        torch.cat([chunk["labels"] for chunk in chunks]).to(
                            trainer.device
                        ),
                    )

                inputs, targets = coordinated_call(window, phase="evaluation window")
                result = trainer.evaluate(inputs, targets, return_outputs=False)

                def validate():
                    count = result.normalization_count
                    if type(count) is not int or count <= 0:
                        raise ValueError(
                            "evaluation requires positive integer scored tokens"
                        )
                    if not math.isfinite(result.loss):
                        raise FloatingPointError("Nonfinite evaluation NLL")
                    return result.loss * count, count

                loss, count = coordinated_call(validate, phase="evaluation metrics")
                total_loss += loss
                scored_tokens += count
            sync()
            elapsed = torch.tensor(
                time.perf_counter() - started,
                dtype=torch.float64,
                device=trainer.device,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        finally:
            cursor.load_state_dict(snapshot)

    def summarize():
        nll = total_loss / scored_tokens
        try:
            perplexity = math.exp(nll)
        except OverflowError as exc:
            raise FloatingPointError("Nonfinite evaluation perplexity") from exc
        if not math.isfinite(nll) or not math.isfinite(perplexity):
            raise FloatingPointError("Nonfinite evaluation NLL/perplexity")
        blocks = max_batches * microbatches * cursor.global_batch
        input_tokens = blocks * cursor.sequence_length
        return {
            **identity,
            "metric": "heldout_disjoint_block_next_token_nll",
            "packing": cursor.corpus.manifest["document_policy"],
            "nll": nll,
            "perplexity": perplexity,
            "loss_sum": total_loss,
            "scored_tokens": scored_tokens,
            "input_tokens": input_tokens,
            "block_count": blocks,
            "covered_block_range": [0, blocks],
            "excluded_tokens": len(cursor.corpus.tokens) - input_tokens,
            "seconds": elapsed.item(),
            "timing_scope": "evaluation windows including data/transfer/coordination; excludes setup and training",
        }

    return coordinated_call(summarize, phase="evaluation summary")
