# -*- coding: utf-8 -*-

"""Run a direct PyTorch/DeepSpeed CLIP-style training baseline.

This script intentionally avoids ParaScale runtime imports. It is used to
compare ParaScale benchmark results against plain torch.distributed or
DeepSpeed training under the same synthetic CLIP-medium task shape.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


class ClipMedium(nn.Module):
    """CLIP-B style contrastive model matching the ParaScale benchmark shape."""

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        channels: int,
        vocab_size: int,
        text_length: int,
        embed_dim: int,
        vision_layers: int,
        text_layers: int,
        num_heads: int,
        mlp_ratio: float,
        temperature: float,
    ) -> None:
        super().__init__()
        grid = image_size // patch_size
        sequence_length = grid * grid + 1
        self.patch_embed = nn.Conv2d(
            channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.image_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.image_pos = nn.Parameter(torch.zeros(1, sequence_length, embed_dim))
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        self.text_pos = nn.Parameter(torch.zeros(1, text_length, embed_dim))
        vision_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            activation="gelu",
        )
        text_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            activation="gelu",
        )
        self.vision_encoder = nn.TransformerEncoder(
            vision_layer, num_layers=vision_layers
        )
        self.text_encoder = nn.TransformerEncoder(text_layer, num_layers=text_layers)
        self.image_norm = nn.LayerNorm(embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.image_proj = nn.Linear(embed_dim, embed_dim)
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temperature))
        nn.init.normal_(self.image_cls, std=0.02)
        nn.init.normal_(self.image_pos, std=0.02)
        nn.init.normal_(self.text_pos, std=0.02)

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        image_tokens = (
            self.patch_embed(pixel_values.to(dtype=self.patch_embed.weight.dtype))
            .flatten(2)
            .transpose(1, 2)
        )
        cls = self.image_cls.expand(batch_size, -1, -1)
        image_tokens = torch.cat([cls, image_tokens], dim=1)
        image_tokens = image_tokens + self.image_pos[:, : image_tokens.shape[1], :]
        image_features = self.vision_encoder(image_tokens)[:, 0]

        text_tokens = self.text_embed(input_ids.long())
        text_tokens = text_tokens + self.text_pos[:, : text_tokens.shape[1], :]
        text_features = self.text_encoder(text_tokens).mean(dim=1)

        image_features = F.normalize(
            self.image_proj(self.image_norm(image_features)), dim=-1
        )
        text_features = F.normalize(
            self.text_proj(self.text_norm(text_features)), dim=-1
        )
        return self.logit_scale.exp().clamp(max=100.0) * (
            image_features @ text_features.T
        )


def _distributed_context(
    local_rank_arg: int | None = None,
    *,
    init_process_group: bool = True,
) -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0") or 0)
    world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    local_rank = int(os.environ.get("LOCAL_RANK", local_rank_arg or 0) or 0)
    if init_process_group and world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def _destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _wrap_model(model: nn.Module, backend: str, device: torch.device) -> nn.Module:
    if backend == "ddp":
        return DistributedDataParallel(
            model.to(device),
            device_ids=[device.index],
            output_device=device.index,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    if backend == "fsdp":
        from torch.distributed.fsdp import (
            FullyShardedDataParallel,
            MixedPrecision,
            ShardingStrategy,
        )

        mixed_precision = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.float32,
        )
        return FullyShardedDataParallel(
            model.to(device),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_id=device,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _deepspeed_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 0,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 200000000,
            "overlap_comm": False,
            "reduce_scatter": True,
            "reduce_bucket_size": 200000000,
            "contiguous_gradients": True,
        },
    }


def _make_batch(
    *,
    batch_size: int,
    channels: int,
    image_size: int,
    text_length: int,
    vocab_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pixel_values = torch.randn(
        batch_size,
        channels,
        image_size,
        image_size,
        generator=generator,
    ).to(device, non_blocking=True)
    input_ids = torch.randint(
        low=1,
        high=vocab_size,
        size=(batch_size, text_length),
        generator=generator,
    ).to(device, non_blocking=True)
    labels = torch.arange(batch_size, dtype=torch.long, device=device)
    return {"pixel_values": pixel_values, "input_ids": input_ids, "labels": labels}


def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rank, world_size, local_rank = _distributed_context(
        args.local_rank,
        init_process_group=args.backend != "deepspeed",
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)

    model = ClipMedium(
        image_size=args.image_size,
        patch_size=args.patch_size,
        channels=args.channels,
        vocab_size=args.vocab_size,
        text_length=args.text_length,
        embed_dim=args.embed_dim,
        vision_layers=args.vision_layers,
        text_layers=args.text_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        temperature=args.temperature,
    )
    if args.backend == "deepspeed":
        import deepspeed

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        wrapped, optimizer, _, _ = deepspeed.initialize(
            model=model.to(device),
            model_parameters=model.parameters(),
            optimizer=optimizer,
            config=_deepspeed_config(args),
        )
    else:
        wrapped = _wrap_model(model, args.backend, device)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=args.lr)

    step_times = []
    end_to_end_times = []
    losses = []
    dataloader_waits = []
    torch.cuda.reset_peak_memory_stats(device)
    total_start = time.perf_counter()

    for step in range(args.steps):
        load_start = time.perf_counter()
        batch = _make_batch(
            batch_size=args.batch_size,
            channels=args.channels,
            image_size=args.image_size,
            text_length=args.text_length,
            vocab_size=args.vocab_size,
            seed=args.seed + rank * 100000 + step,
            device=device,
        )
        dataloader_wait = (time.perf_counter() - load_start) * 1000.0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_start = time.perf_counter()
        if args.backend == "deepspeed":
            wrapped.zero_grad()
            logits = wrapped(batch["pixel_values"], batch["input_ids"])
            loss = _loss(logits, batch["labels"])
            wrapped.backward(loss)
            wrapped.step()
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = wrapped(batch["pixel_values"], batch["input_ids"])
                loss = _loss(logits, batch["labels"])
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - step_start
        end_to_end = time.perf_counter() - load_start
        if step >= args.warmup_steps:
            step_times.append(elapsed)
            end_to_end_times.append(end_to_end)
            losses.append(float(loss.detach().cpu()))
            dataloader_waits.append(dataloader_wait)

    total_elapsed = time.perf_counter() - total_start
    local_images = args.batch_size
    global_images = local_images * world_size
    stable_step_time = _mean(step_times)
    stable_end_to_end = _mean(end_to_end_times)
    stable_images = global_images / stable_step_time if stable_step_time > 0 else 0.0
    stable_e2e_images = (
        global_images / stable_end_to_end if stable_end_to_end > 0 else 0.0
    )
    metrics = {
        "loss": _mean(losses),
        "batch_size": local_images,
        "global_batch_size": global_images,
        "step_time_seconds": stable_step_time,
        "images_per_second": stable_images,
        "image_text_pairs_per_second": stable_images,
        "end_to_end_images_per_second": stable_e2e_images,
        "end_to_end_image_text_pairs_per_second": stable_e2e_images,
        "stable_end_to_end_image_text_pairs_per_second": stable_e2e_images,
        "steps_per_second": 1.0 / stable_end_to_end if stable_end_to_end > 0 else 0.0,
        "dataloader_wait_ms": _mean(dataloader_waits),
        "peak_memory_bytes": float(torch.cuda.max_memory_allocated(device)),
    }
    if dist.is_initialized():
        tensor = torch.tensor(
            [
                metrics["loss"],
                metrics["images_per_second"],
                metrics["end_to_end_image_text_pairs_per_second"],
                metrics["peak_memory_bytes"],
                metrics["dataloader_wait_ms"],
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
        metrics["loss"] = float(tensor[0].item())
        metrics["images_per_second"] = float(tensor[1].item())
        metrics["image_text_pairs_per_second"] = float(tensor[1].item())
        metrics["end_to_end_images_per_second"] = float(tensor[2].item())
        metrics["end_to_end_image_text_pairs_per_second"] = float(tensor[2].item())
        metrics["stable_end_to_end_image_text_pairs_per_second"] = float(
            tensor[2].item()
        )
        metrics["peak_memory_bytes"] = float(tensor[3].item())
        metrics["dataloader_wait_ms"] = float(tensor[4].item())

    payload = {
        "mode": "direct_pytorch_baseline",
        "benchmark_type": "clip_contrastive_train",
        "runtime_status": "real_local",
        "capability_level": "direct_pytorch_clip_synthetic",
        "backend": f"direct_{args.backend}",
        "metrics": metrics,
        "train": {
            "mode": "train",
            "backend": f"direct_{args.backend}",
            "global_step": args.steps,
            "elapsed_seconds": total_elapsed,
            "last_metrics": metrics,
            "world_size": world_size,
            "precision": "bf16_autocast",
            "workload": "clip_contrastive",
            "data_type": "synthetic_clip",
        },
        "task": {
            "workload": "clip_contrastive",
            "objective": "image_text_contrastive",
        },
        "model": {
            "type": "clip_medium",
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "embed_dim": args.embed_dim,
            "vision_layers": args.vision_layers,
            "text_layers": args.text_layers,
        },
    }
    if rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ddp", "fsdp", "deepspeed"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=49408)
    parser.add_argument("--text-length", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--vision-layers", type=int, default=12)
    parser.add_argument("--text-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    try:
        run(args)
    finally:
        _destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
