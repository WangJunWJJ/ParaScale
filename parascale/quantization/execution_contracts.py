# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午6:44
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Preflight runtime callbacks that reference quantization cannot preserve."""

from torch import nn
from torch.nn.modules import module as module_runtime

_EXECUTION_HOOKS = (
    "_forward_pre_hooks",
    "_forward_hooks",
    "_backward_pre_hooks",
    "_backward_hooks",
)
_SERIALIZATION_HOOKS = (
    "_state_dict_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)


def validate_serialization_hooks(model: nn.Module) -> None:
    """Reject state callbacks before signature, export, or load can execute them."""
    for name, module in model.named_modules():
        for attribute in _SERIALIZATION_HOOKS:
            if getattr(module, attribute, None):
                raise ValueError(
                    f"unsupported serialization hooks: {name or '<root>'} "
                    f"({attribute}); remove callbacks before reference quantization"
                )


def validate_copy(model: nn.Module) -> None:
    """Reject callbacks or compiled calls that copying cannot safely preserve."""
    for attribute in _EXECUTION_HOOKS:
        if getattr(module_runtime, "_global" + attribute, None):
            raise ValueError(
                f"unsupported execution hooks: global {attribute}; "
                "remove hooks before reference quantization"
            )
    for name, module in model.named_modules():
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise ValueError(
                f"unsupported execution override: {name or '<root>'}; "
                "use an uncompiled model for reference quantization"
            )
    for name, parameter in model.named_parameters():
        if getattr(parameter, "_backward_hooks", None) or getattr(
            parameter, "_post_accumulate_grad_hooks", None
        ):
            raise ValueError(
                f"unsupported execution hooks: parameter {name}; "
                "deepcopy does not preserve gradient hooks"
            )
    validate_serialization_hooks(model)


def validate_replacement(model: nn.Module, prefix: str = "") -> None:
    """Reject executable module state lost when replacing this subtree."""
    for name, module in model.named_modules():
        path = ".".join(part for part in (prefix, name) if part) or "<root>"
        if any(getattr(module, attr, None) for attr in _EXECUTION_HOOKS):
            raise ValueError(
                f"unsupported execution hooks: {path}; "
                "remove hooks or adapt the model before reference quantization"
            )
        if "forward" in vars(module):
            raise ValueError(
                f"unsupported execution override: {path}; "
                "use an unmodified, uncompiled model for reference quantization"
            )


def validate_artifact(model: nn.Module) -> None:
    """Reject runtime callbacks omitted from the state-only artifact contract."""
    validate_copy(model)
    validate_replacement(model)
