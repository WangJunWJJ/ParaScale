# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Metadata-only checks for acceptance's native_parallel Llama layouts."""

import json
from pathlib import Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate config key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def model_layout_blockers(case, env):
    """Return blockers, not qualification; never import/load model libraries.

    Mirrors metadata checks in LlamaStage, not runtime tensor/weight validation.
    The product loader overrides stored dtype and attention with FP32 and SDPA.
    """
    if case.kind != "parallel":
        return []
    model = env["assets"].get("llama_model")
    if not model:
        return ["model preflight: missing local llama_model/config.json"]
    path = Path(model) / "config.json"
    prefix = f"model preflight {path}"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"),
                              object_pairs_hook=_unique_object,
                              parse_constant=_invalid_constant)
    except (OSError, ValueError, RecursionError) as exc:
        return [f"{prefix}: cannot read valid model config: {exc}"]
    if not isinstance(metadata, dict):
        return [f"{prefix}: model config must be a JSON object"]

    reasons = []
    if metadata.get("model_type") != "llama":
        reasons.append("model_type must be 'llama' for native_parallel LlamaStage")
    dimensions = {}
    for field in ("hidden_size", "intermediate_size", "num_attention_heads",
                  "num_key_value_heads", "num_hidden_layers", "vocab_size"):
        value = metadata.get(field)
        # LlamaConfig defaults missing/null KV heads to attention heads.
        if field == "num_key_value_heads" and value is None:
            value = metadata.get("num_attention_heads")
        if type(value) is not int or value <= 0:
            reasons.append(f"{field} must be a positive integer; got {value!r}")
        else:
            dimensions[field] = value

    _, tp, pp = case.layout
    for field in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads"):
        value = dimensions.get(field)
        if value is not None and value % tp:
            reasons.append(f"{field}={value} must be divisible by tp_size={tp}")
    layers = dimensions.get("num_hidden_layers")
    if layers is not None and layers < pp:
        reasons.append(f"num_hidden_layers={layers} must provide at least one layer per pipeline stage (pp_size={pp})")

    # Optional fields use the pinned LlamaConfig defaults; explicit bad types
    # are not coerced, so strings and bool-as-integer metadata fail closed.
    tied = metadata.get("tie_word_embeddings", False)
    if type(tied) is not bool:
        reasons.append("tie_word_embeddings must be a boolean")
    elif pp > 1 and tied:
        reasons.append(f"tie_word_embeddings=True is unsupported across pipeline stages (pp_size={pp})")
    dropout = metadata.get("attention_dropout", 0.0)
    if type(dropout) not in (int, float) or dropout != 0:
        reasons.append(f"attention_dropout must be zero; got {dropout!r}")
    pretraining_tp = metadata.get("pretraining_tp", 1)
    if type(pretraining_tp) is not int or pretraining_tp != 1:
        reasons.append(f"pretraining_tp must be one; got {pretraining_tp!r}; use the explicit TP layout")
    return [f"{prefix}: {reason}" for reason in reasons]
