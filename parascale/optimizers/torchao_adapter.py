# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Qualified entry points for the optional official TorchAO optimizer.

The upstream update and checkpoint representation remain unchanged. Importing
this module does not import TorchAO or initialize compiler/device state.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .spec import OptimizerSpec


def load_optimizer_class() -> Any:
    """Load the pinned implementation and register its weights-only state class."""
    import torch

    try:
        ao_version = version("torchao")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Install parascale[optimizer-4bit] in a separate CUDA environment."
        ) from exc
    if torch.__version__.split("+")[0] != "2.9.1" or ao_version != "0.15.0":
        raise RuntimeError(
            "torchao_adamw4bit requires torch 2.9.1 / torchao 0.15.0; "
            f"found {torch.__version__} / {ao_version}. "
            "Use a separate parascale[optimizer-4bit] environment."
        )
    from torchao.optim import AdamW4bit

    return AdamW4bit


def prepare_checkpoint(metadata: dict[str, Any]) -> None:
    """Import only the known state type after validating checkpoint identity."""
    if metadata.get("type") != "torchao_adamw4bit":
        return
    expected = OptimizerSpec(type="torchao_adamw4bit").to_metadata()
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("TorchAO checkpoint optimizer identity mismatch")
    load_optimizer_class()


def validate_execution(optimizer: Any, config: Any) -> None:
    """Check final placement before training; CPU construction is permitted."""
    import torch

    OptimizerSpec(type="torchao_adamw4bit").validate_backend(
        getattr(config, "training_backend", "native"),
        int(getattr(config, "zero_stage", 0) or 0),
        str(getattr(config, "precision", "fp32")),
    )
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.device.type != "cuda" or parameter.dtype != torch.float32:
                raise ValueError("torchao_adamw4bit requires CUDA FP32 parameters")


def state_inventory(optimizer: Any) -> dict[str, Any]:
    """Count physical state, dense fallbacks and group tensors by device.

    This is neither allocator peak memory nor an estimated AdamW saving.
    CPU scalar state is disclosed separately from CUDA storage.
    """
    from torchao.optim.subclass_4bit import OptimState4bit

    from .optimizers import _state_storage_bytes

    def physical(value):
        if isinstance(value, OptimState4bit):
            return (value.codes, value.scale, value.qmap)
        if isinstance(value, dict):
            return {key: physical(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [physical(item) for item in value]
        return value

    groups = [{key: value for key, value in group.items() if key != "params"}
              for group in optimizer.param_groups]
    by_device = _state_storage_bytes(physical([optimizer.state, groups]))
    quantized, dense = 0, 0
    for parameter, state in optimizer.state.items():
        if "exp_avg" not in state:
            continue
        if isinstance(state["exp_avg"], OptimState4bit):
            quantized += parameter.numel()
        else:
            dense += parameter.numel()
    return {"state_bytes_by_device": by_device, "state_bytes": sum(by_device.values()),
            "quantized_params": quantized, "dense_params": dense,
            "scope": "persistent optimizer state and group tensors; not peak memory"}
