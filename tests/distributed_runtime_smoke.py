# -*- coding: utf-8 -*-
# @Time : 2026/5/8 上午11:55
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Torchrun-driven distributed smoke checks for optional backends."""

from __future__ import annotations

import argparse
import importlib.util
import os


def main() -> int:
    parser = argparse.ArgumentParser(description="ParaScale distributed backend smoke")
    parser.add_argument("--backend", choices=["fsdp", "deepspeed"], required=True)
    args = parser.parse_args()

    if importlib.util.find_spec("torch") is None:
        return _skip("torch is not installed")

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim

    if not dist.is_available():
        return _skip("torch.distributed is not available")
    if (
        args.backend == "fsdp"
        and importlib.util.find_spec("torch.distributed.fsdp") is None
    ):
        return _skip("torch.distributed.fsdp is not available")
    if args.backend == "deepspeed" and importlib.util.find_spec("deepspeed") is None:
        return _skip("DeepSpeed is not installed")
    if not torch.cuda.is_available():
        return _skip("CUDA is not available; distributed training smoke requires CUDA")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    backend = "nccl"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    try:
        from parascale.config import ParaScaleConfig
        from parascale.runtime.backend import create_runtime_training_backend

        model = nn.Linear(4, 2).to(local_rank)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        config = ParaScaleConfig(training_backend=args.backend)
        runtime_backend = create_runtime_training_backend(
            model=model,
            optimizer=optimizer,
            config=config,
            local_rank=local_rank,
        )
        wrapped_model, wrapped_optimizer = runtime_backend.setup()
        batch = torch.randn(2, 4, device=local_rank)
        target = torch.randn(2, 2, device=local_rank)
        output = wrapped_model(batch)
        loss = ((output - target) ** 2).mean()
        runtime_backend.backward(loss)
        runtime_backend.step(wrapped_optimizer)
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                f"[ParaScale distributed smoke] backend={args.backend} "
                f"train_step=ok world_size={dist.get_world_size()}"
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


def _skip(reason: str) -> int:
    print(f"[ParaScale distributed smoke] skipped: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
