# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:56
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Torchrun launcher helpers."""

from __future__ import annotations


def torchrun_command(
    entrypoint_args: list[str],
    config_path: str,
    *,
    world_size: int,
    nproc_per_node: int,
    nnodes: int | None = None,
    node_rank: int | None = None,
    master_addr: str | None = None,
    master_port: int | None = None,
) -> list[str]:
    resolved_nnodes = nnodes or max(1, int(world_size) // max(1, int(nproc_per_node)))
    rendezvous_args = (
        ["--standalone"]
        if int(resolved_nnodes) == 1
        else [
            f"--nnodes={int(resolved_nnodes)}",
            f"--node_rank={int(node_rank or 0)}",
            f"--master_addr={master_addr or '127.0.0.1'}",
            f"--master_port={int(master_port or 29500)}",
        ]
    )
    return [
        "torchrun",
        *rendezvous_args,
        f"--nproc_per_node={int(nproc_per_node)}",
        *entrypoint_args,
        "--config",
        config_path,
    ]


__all__ = ["torchrun_command"]
