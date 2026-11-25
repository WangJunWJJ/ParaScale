# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午7:25
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Canonical artifact loading and a matched, dequantized-once BF16 control."""

import torch
from torch import nn

from parascale.quantization import load_quantized_model
from parascale.quantization.execution_contracts import validate_artifact
from parascale.quantization.inference_artifact import _select_reference_layers


def load_reference_artifact(artifact, model_path):
    """Validate the product wrapper before exposing its HF generation child.

    Use the same tokenizer-bound template as product inference; no original
    float weights or legacy bare-HF artifact signatures are accepted.
    """
    from parascale.workloads.causal_lm_model import load_causal_lm_model

    template = load_causal_lm_model(model_path, from_config=True)
    model, quant_config, _ = load_quantized_model(artifact, template)
    _select_reference_layers(model)
    return model.model, quant_config.to_dict()


def materialize_reference_weights(model):
    """Use identical stored weights but pay dequantization once, outside timing."""
    validate_artifact(model)
    selected = _select_reference_layers(model)
    with torch.no_grad():
        for name, layer in selected.items():
            weight = layer.dequantized_weight(layer._weight_dtype.dtype)
            dense = nn.Linear(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                device=weight.device,
                dtype=weight.dtype,
            )
            dense.weight.copy_(weight)
            if layer.bias is not None:
                dense.bias.copy_(layer.bias)
            dense.requires_grad_(False).eval()
            if name:
                model.set_submodule(name, dense)
            else:
                model = dense
    return model
