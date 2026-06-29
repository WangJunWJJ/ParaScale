# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:56
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Collective backend registry."""

from __future__ import annotations

from .collective import MockCollectiveBackend, TorchDistributedCollectiveBackend


def create_collective_backend(kind: str = "mock"):
    normalized = (kind or "mock").lower()
    if normalized in {"mock", "none"}:
        return MockCollectiveBackend()
    if normalized in {"torch", "torch_distributed", "gloo", "nccl", "hccl", "auto"}:
        backend = "auto" if normalized in {"torch", "torch_distributed"} else normalized
        return TorchDistributedCollectiveBackend(backend=backend)
    raise ValueError(f"unknown collective backend: {kind}")


__all__ = ["create_collective_backend"]
