# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:56
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Distributed communication namespace."""

from .collective import (
    CollectiveBackend,
    MockCollectiveBackend,
    TorchDistributedCollectiveBackend,
)
from .process_group import ProcessGroupSpec
from .registry import create_collective_backend

__all__ = [
    "CollectiveBackend",
    "MockCollectiveBackend",
    "ProcessGroupSpec",
    "TorchDistributedCollectiveBackend",
    "create_collective_backend",
]
