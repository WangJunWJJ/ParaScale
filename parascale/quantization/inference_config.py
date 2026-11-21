# -*- coding: utf-8 -*-
# @Time : 2026/11/21 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Opt-in fused inference configuration, separate from reference QAT/PTQ."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceQuantizationConfig:
    """Groupwise INT4 weights with BF16 activations on CUDA SM80 or newer.

    Exclusions match a full module path, its descendants, or a path suffix.
    The default preserves the output head. This is not a training config.
    """

    group_size: int = 128
    exclude_modules: tuple[str, ...] = ("lm_head",)
    backend: str = "torchao_int4"

    def __post_init__(self):
        if self.backend not in ("torchao_int4", "gemlite_int8"):
            raise ValueError("backend must be torchao_int4 or gemlite_int8")
        if self.backend == "gemlite_int8" and self.group_size != 128:
            raise ValueError(
                "group_size applies only to torchao_int4; leave its default for INT8"
            )
        if type(self.group_size) is not int or self.group_size not in (
            32,
            64,
            128,
            256,
        ):
            raise ValueError("group_size must be one of 32, 64, 128, 256")
        if not isinstance(self.exclude_modules, (list, tuple)) or any(
            not isinstance(name, str) or not name.strip()
            for name in self.exclude_modules
        ):
            raise ValueError("exclude_modules must be a list/tuple of nonempty paths")
        object.__setattr__(self, "exclude_modules", tuple(self.exclude_modules))
