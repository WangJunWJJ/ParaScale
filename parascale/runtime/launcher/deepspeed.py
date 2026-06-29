# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:56
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""DeepSpeed launcher helpers."""

from __future__ import annotations


def deepspeed_command(
    entrypoint_args: list[str], config_path: str, *, nproc_per_node: int
) -> list[str]:
    return [
        "deepspeed",
        f"--num_gpus={int(nproc_per_node)}",
        *entrypoint_args,
        "--config",
        config_path,
    ]


__all__ = ["deepspeed_command"]
