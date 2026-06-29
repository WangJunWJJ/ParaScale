# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:24
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""CPU device backend."""

from __future__ import annotations

from .base import DeviceBackend


class CpuDeviceBackend(DeviceBackend):
    def __init__(self) -> None:
        super().__init__(
            name="cpu", accelerator="cpu", communication="gloo", available=True
        )


__all__ = ["CpuDeviceBackend"]
