# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午1:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Strict caller-declared training semantics for stage checkpoint restoration."""

from __future__ import annotations

import json
from typing import Any


def canonical_checkpoint_context(context: Any) -> str:
    """Return stable JSON without coercing keys, tuples or nonfinite numbers.

    The caller owns the descriptors and must derive them from the effective
    runtime configuration and assets, not just filenames. This validates identity
    declarations, not arbitrary model code or external data-loader state.
    """
    required = {"model", "loss", "data", "tokenizer", "precision"}
    if not isinstance(context, dict) or not required <= context.keys():
        raise ValueError(
            "Checkpoint context requires model, loss, data, tokenizer and precision"
        )
    if context["precision"] != "fp32":
        raise ValueError(
            "Checkpoint context precision must match the FP32 stage contract"
        )
    if any(
        not isinstance(context[key], (str, dict))
        or not context[key]
        or (isinstance(context[key], str) and not context[key].strip())
        for key in required - {"precision"}
    ):
        raise ValueError(
            "Checkpoint context descriptors must be nonempty strings or objects"
        )
    try:
        serialized = json.dumps(
            context, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("Checkpoint context must be finite, acyclic JSON") from error

    def validate(value: Any) -> None:
        if isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise ValueError("Checkpoint context object keys must be strings")
            for child in value.values():
                validate(child)
        elif isinstance(value, list):
            for child in value:
                validate(child)
        elif value is not None and type(value) not in (str, bool, int, float):
            raise ValueError("Checkpoint context must use JSON values without coercion")

    validate(context)
    return serialized
