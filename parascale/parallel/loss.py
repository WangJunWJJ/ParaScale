# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Explicit summed objectives for token-weighted model-parallel training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class SummedLoss(Protocol):
    """A differentiable scalar sum with a device-local int64 unit count.

    normalization_count validates the complete local target batch before forward.
    Zero local counts are valid; the distributed global count must be positive.
    __call__ must keep a backward graph even when every target is ignored.
    """

    def __call__(self, output: Any, targets: Any) -> Any: ...

    def normalization_count(self, targets: Any) -> Any: ...


@dataclass(frozen=True)
class CausalLMLoss:
    """Summed next-token cross entropy for fixed-length causal blocks.

    Args:
        vocab_size: Size of the unsharded output vocabulary.
        ignore_index: Label omitted from the objective, not an attention mask.

    Targets are int64 [batch, sequence], logits are [batch, sequence, vocabulary].
    The last logit and first label are excluded. normalization_count rejects
    invalid scored labels before distributed execution. Logits must already be
    replicated across TP ranks; vocabulary-parallel cross entropy is not implied.
    """

    vocab_size: int
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if type(self.vocab_size) is not int or self.vocab_size < 1:
            raise ValueError("vocab_size must be a positive integer")
        if type(self.ignore_index) is not int:
            raise ValueError("ignore_index must be an integer")

    @staticmethod
    def _validate_labels(labels: Any) -> None:
        import torch

        if (
            not isinstance(labels, torch.Tensor)
            or labels.dtype != torch.int64
            or labels.ndim != 2
            or labels.shape[0] < 1
            or labels.shape[1] < 2
        ):
            raise ValueError("labels must be nonempty int64 [batch, sequence>=2]")

    def normalization_count(self, targets: Any) -> Any:
        """Return valid shifted-token count, rejecting out-of-vocabulary labels."""
        self._validate_labels(targets)
        shifted = targets[:, 1:]
        valid = shifted != self.ignore_index
        if ((shifted < 0) | (shifted >= self.vocab_size)).logical_and(valid).any():
            raise ValueError("scored labels are outside the configured vocabulary")
        return valid.sum()

    def __call__(self, output: Any, targets: Any) -> Any:
        """Return a graph-connected scalar sum, including for all-ignored labels."""
        import torch
        import torch.nn.functional as functional

        self._validate_labels(targets)
        if (
            not isinstance(output, torch.Tensor)
            or not output.is_floating_point()
            or output.shape != (*targets.shape, self.vocab_size)
            or output.device != targets.device
        ):
            raise ValueError("logits must match labels and the configured vocabulary")
        return functional.cross_entropy(
            output[:, :-1].float().reshape(-1, self.vocab_size),
            targets[:, 1:].reshape(-1),
            reduction="sum",
            ignore_index=self.ignore_index,
        )
