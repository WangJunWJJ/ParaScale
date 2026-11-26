# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Physical Llama MLP channel removal with a magnitude-ranking baseline."""

import math
from collections import Counter
from copy import deepcopy
from numbers import Real
from typing import Any

import torch
from torch import nn

from parascale.core.model_transform import validate_artifact


@torch.inference_mode(False)
@torch.no_grad()
def prune_llama_mlp(
    model: nn.Module, *, keep_ratio: float = 0.75, alignment: int = 128
) -> tuple[nn.Module, dict[str, Any]]:
    """Return a separate, smaller HF Llama and an auditable channel report.

    Rank channels by FP32 L2 norm of down-projection columns, retaining the
    same indices in gate/up rows and down columns. Round retained width DOWN
    to alignment; the report states the actual ratio. Attention is unchanged.
    This heuristic needs independent quality evaluation and usually recovery
    training. No quality or speed improvement is guaranteed.

    Args:
        model: Ordinary eval-mode LlamaForCausalLM on one CPU/CUDA device.
            Quantization, adapters, distributed wrappers and hooks are rejected.
        keep_ratio: Fraction of MLP intermediate channels to retain, in (0, 1).
        alignment: Positive integer alignment of the resulting dense matrices.

    Returns:
        Copied model with standard Linear layers and matching HF configuration,
        and a JSON-compatible report. Copying does not create inference tensors,
        even inside inference_mode. Save using save_pretrained; save the
        original tokenizer alongside it. Construct a NEW optimizer afterward;
        old optimizer states and distributed checkpoints are incompatible.

    Raises:
        ValueError: Unsupported model/execution, invalid dimensions or weights.
        TypeError: Input is not the supported Hugging Face model type.
    """
    from transformers import LlamaForCausalLM
    from transformers.models.llama.modeling_llama import LlamaMLP

    if type(model) is not LlamaForCausalLM:
        raise TypeError("pruning requires an ordinary LlamaForCausalLM")
    if isinstance(keep_ratio, bool) or not isinstance(keep_ratio, Real) or not 0 < keep_ratio < 1:
        raise ValueError("keep_ratio must be finite and in (0, 1)")
    if type(alignment) is not int or alignment < 1:
        raise ValueError("alignment must be a positive integer")
    if any(module.training for module in model.modules()):
        raise ValueError("Call model.eval() before pruning")
    if getattr(model, "hf_device_map", None) or getattr(model.config, "pretraining_tp", 1) != 1:
        raise ValueError("pruning does not support dispatched or tensor-parallel models")
    validate_artifact(model)
    parameters = list(model.parameters())
    if len({p.device for p in parameters}) != 1 or any(
        type(p) is not nn.Parameter or p.device.type not in {"cpu", "cuda"}
        or p.dtype not in {torch.float32, torch.float16, torch.bfloat16}
        for p in parameters
    ):
        raise ValueError("pruning requires ordinary floating parameters on one CPU/CUDA device")
    width, hidden = model.config.intermediate_size, model.config.hidden_size
    kept = math.floor(width * keep_ratio / alignment) * alignment
    if not 0 < kept < width:
        raise ValueError("alignment and keep_ratio must retain nonzero, fewer channels")
    # Verify the architecture can be reconstructed from its public HF config.
    # Meta construction checks the entire model without allocating model weights.
    with torch.device("meta"):
        reference = LlamaForCausalLM(deepcopy(model.config))
    actual_modules = {name: type(module) for name, module in model.named_modules()}
    expected_modules = {name: type(module) for name, module in reference.named_modules()}
    actual_shapes = {name: tuple(p.shape) for name, p in model.named_parameters(remove_duplicate=False)}
    expected_shapes = {name: tuple(p.shape) for name, p in reference.named_parameters(remove_duplicate=False)}
    if actual_modules != expected_modules or actual_shapes != expected_shapes:
        raise ValueError("model architecture differs from its HF configuration; adapters/custom projections are unsupported")
    del reference
    aliases = Counter((p.device, p.untyped_storage().data_ptr())
                      for _, p in model.named_parameters(remove_duplicate=False))
    selections = []
    for layer in model.model.layers:
        mlp = layer.mlp
        if type(mlp) is not LlamaMLP:
            raise ValueError("custom or adapted MLP modules are unsupported")
        for name, shape in (("gate_proj", (width, hidden)), ("up_proj", (width, hidden)),
                            ("down_proj", (hidden, width))):
            linear = getattr(mlp, name)
            if type(linear) is not nn.Linear or tuple(linear.weight.shape) != shape:
                raise ValueError("MLP projections must be ordinary Linear layers with configured shapes")
            for parameter in linear.parameters():
                if aliases[(parameter.device, parameter.untyped_storage().data_ptr())] != 1:
                    raise ValueError("shared MLP parameters are unsupported")
                if not torch.isfinite(parameter).all():
                    raise ValueError("MLP parameters must be finite")
        scores = torch.linalg.vector_norm(mlp.down_proj.weight.detach().float(), dim=0)
        if not torch.isfinite(scores).all():
            raise ValueError("channel scores must be finite")
        selections.append(torch.argsort(scores, descending=True, stable=True)[:kept].sort().values)

    result = deepcopy(model)
    records = []
    with torch.no_grad():
        for index, (layer, selected) in enumerate(zip(result.model.layers, selections)):
            for name in ("gate_proj", "up_proj", "down_proj"):
                old = getattr(layer.mlp, name)
                axis = 1 if name == "down_proj" else 0
                weight = old.weight.index_select(axis, selected)
                new = nn.Linear(weight.shape[1], weight.shape[0], bias=old.bias is not None,
                                device="meta", dtype=old.weight.dtype)
                new.weight = nn.Parameter(weight.clone(), requires_grad=old.weight.requires_grad)
                if old.bias is not None:
                    bias = old.bias if axis == 1 else old.bias.index_select(0, selected)
                    new.bias = nn.Parameter(bias.clone(), requires_grad=old.bias.requires_grad)
                new.eval()
                setattr(layer.mlp, name, new)
            layer.mlp.intermediate_size = kept
            records.append({"layer": index, "kept_indices": selected.cpu().tolist()})
    result.config.intermediate_size = kept
    before = sum(p.numel() for p in model.parameters())
    after = sum(p.numel() for p in result.parameters())
    return result, {"method": "llama_mlp_down_column_l2", "version": 1,
                    "intermediate_size_before": width, "intermediate_size_after": kept,
                    "requested_keep_ratio": float(keep_ratio), "actual_keep_ratio": kept / width,
                    "alignment": alignment, "parameters_before": before, "parameters_after": after,
                    "parameter_reduction_fraction": 1 - after / before, "layers": records}
