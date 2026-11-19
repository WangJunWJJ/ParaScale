# -*- coding: utf-8 -*-
# @Time : 2026/11/16 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""CUDA training contract and synthetic runtime throughput benchmark."""

import argparse
import json
import os
import time
from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist

from parascale.config import ParaScaleConfig
from parascale.runtime.training import TrainEngine


class Model(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
        )

    def forward(self, x, **kwargs):
        return self.layers(x)


def loss_fn(output, batch):
    return (output.float() - batch["labels"]).square().mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="native_ddp")
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    device = torch.device("cuda", local_rank)
    width = 1024 if args.performance else 8
    steps = args.steps if args.performance else 3
    torch.manual_seed(123)
    model = Model(width).to(device)
    reference = Model(width).to(device)
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    config = ParaScaleConfig(
        training_backend=args.backend,
        precision=args.precision,
        gradient_accumulation_steps=2,
        grad_clip_norm=None if args.performance else 0.5,
        batch_size=64,
    )
    engine = TrainEngine(config)

    def batch(worker, micro):
        count = 64 if args.performance else 1 + worker + micro % 2
        generator = torch.Generator(device=device).manual_seed(
            100 + worker * 1000 + micro
        )
        return {
            "x": torch.randn(count, width, generator=generator, device=device),
            "labels": torch.randn(count, width, generator=generator, device=device),
        }

    batches = [batch(rank, i) for i in range(2 * (steps + 20))]
    if args.performance:
        engine.fit(
            batches[:40],
            max_steps=20,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    profiler = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        if args.profile
        else nullcontext()
    )
    with profiler:
        engine.fit(
            batches[: steps * 2],
            max_steps=steps,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )
    torch.cuda.synchronize()
    elapsed = torch.tensor(time.perf_counter() - start, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    error = None
    if not args.performance:
        for step in range(steps):
            parts = [
                batch(worker, step * 2 + micro)
                for worker in range(dist.get_world_size())
                for micro in range(2)
            ]
            combined = {
                key: torch.cat([part[key] for part in parts]) for key in parts[0]
            }
            loss_fn(reference(**combined), combined).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5)
            reference_optimizer.step()
            reference_optimizer.zero_grad()
        actual = engine.training_backend.model
        if args.backend == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with FSDP.summon_full_params(actual):
                error = max(
                    (a - b).abs().max().item()
                    for a, b in zip(actual.parameters(), reference.parameters())
                )
        else:
            error = max(
                (a - b).abs().max().item()
                for a, b in zip(actual.parameters(), reference.parameters())
            )
        assert error < 2e-6, error
        assert engine.state.global_step == steps
        assert engine.state.consumed_micro_batches == steps * 2
    if rank == 0:
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "precision": args.precision,
                    "world_size": dist.get_world_size(),
                    "steps": steps,
                    "warmup_steps": 20 if args.performance else 0,
                    "seconds": elapsed.item(),
                    "max_parameter_error": error,
                    "samples_per_second": steps
                    * 128
                    * dist.get_world_size()
                    / elapsed.item()
                    if args.performance
                    else None,
                    "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
                    "status": "passed",
                    "synthetic": True,
                    "profile": {
                        event.key: {
                            "count": event.count,
                            "self_cpu_us": event.self_cpu_time_total,
                        }
                        for event in profiler.key_averages()
                        if any(
                            name in event.key
                            for name in (
                                "Synchronize",
                                "aten::add",
                                "aten::item",
                                "aten::div",
                            )
                        )
                    }
                    if args.profile
                    else None,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
