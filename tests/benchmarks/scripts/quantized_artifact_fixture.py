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
    """Reconstruct from configuration and the artifact, not original float weights."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    template = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="sdpa"
    )
    model, quant_config, _ = load_quantized_model(artifact, template)
    _select_reference_layers(model)
    return model, quant_config.to_dict()


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
