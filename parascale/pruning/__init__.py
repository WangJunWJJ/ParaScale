# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Structural pruning APIs with optional model dependencies loaded on demand."""

__all__ = ["prune_llama_mlp"]


def __getattr__(name):
    if name == "prune_llama_mlp":
        from .llama import prune_llama_mlp

        return prune_llama_mlp
    raise AttributeError(name)
