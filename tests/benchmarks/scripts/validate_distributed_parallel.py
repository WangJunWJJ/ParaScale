# -*- coding: utf-8 -*-
# @Time : 2026/11/21 下午6:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Numerical and timing evidence for real CUDA model-parallel execution.

The deterministic residual MLP is a communication/correctness diagnostic,
not a pretrained-model accuracy or production throughput benchmark.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from parascale.parallel import DistributedParallelConfig


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.up = nn.Linear(width, width * 4, bias=False)
        self.down = nn.Linear(width * 4, width, bias=False)

    def forward(self, inputs):
        return inputs + self.down(torch.nn.functional.gelu(self.up(inputs)))


class LlamaDecoderBlock(nn.Module):
    """Real HF decoder operators on synthetic hidden states, not a full LM."""

    def __init__(self, width: int, layer_index: int):
        super().__init__()
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer,
            LlamaRotaryEmbedding,
        )

        config = LlamaConfig(
            hidden_size=width,
            intermediate_size=width * 4,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_hidden_layers=4,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        self.decoder = LlamaDecoderLayer(config, layer_index)
        self.rotary = LlamaRotaryEmbedding(config)

    def forward(self, inputs):
        length = inputs.shape[1]
        positions = torch.arange(length, device=inputs.device).unsqueeze(0)
        mask = torch.full(
            (length, length), torch.finfo(inputs.dtype).min, device=inputs.device
        ).triu(1)[None, None]
        output = self.decoder(
            inputs,
            attention_mask=mask,
            position_ids=positions,
            position_embeddings=self.rotary(inputs, positions),
            use_cache=False,
        )
        return output[0] if isinstance(output, tuple) else output


def full_tensor(value):
    return value.full_tensor() if hasattr(value, "full_tensor") else value


def maximum_error(actual, expected):
    error = (actual - expected).abs().max()
    return error.item() if torch.isfinite(error) else float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument(
        "--model", choices=["residual_mlp", "llama_decoder"], default="residual_mlp"
    )
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--global-batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    from parascale.parallel.execution import DistributedTrainer

    config = DistributedParallelConfig(args.dp, args.tp, args.pp, args.microbatches)
    if args.global_batch % (args.dp * args.microbatches):
        raise ValueError("global batch must divide DP * microbatches")
    if args.steps < 4 or args.steps <= args.warmup or args.warmup < 0:
        raise ValueError("steps must be at least four and exceed nonnegative warmup")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=120),
        device_id=torch.device("cuda", local_rank),
    )
    try:
        config.validate_world_size(dist.get_world_size())
        device = torch.device("cuda", local_rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(41)
        model = nn.Sequential(
            OrderedDict(
                (
                    f"block{i}",
                    ResidualBlock(args.width)
                    if args.model == "residual_mlp"
                    else LlamaDecoderBlock(args.width, i),
                )
                for i in range(4)
            )
        )
        reference = copy.deepcopy(model).to(device)
        replica, stage_id, _ = config.coordinates(dist.get_rank())
        names = list(model._modules)
        start, end = (
            len(names) * stage_id // args.pp,
            len(names) * (stage_id + 1) // args.pp,
        )
        stage = nn.Sequential(
            OrderedDict((name, model._modules[name]) for name in names[start:end])
        ).to(device)
        if dist.get_rank() != stage_id * args.tp:
            with torch.no_grad():
                for parameter in stage.parameters():
                    parameter.add_(0.01)
        plan = {}
        for name in names[start:end]:
            if args.model == "residual_mlp":
                plan[f"{name}.up"] = ColwiseParallel()
                plan[f"{name}.down"] = RowwiseParallel()
            else:
                for projection in ("q_proj", "k_proj", "v_proj"):
                    plan[f"{name}.decoder.self_attn.{projection}"] = ColwiseParallel()
                plan[f"{name}.decoder.self_attn.o_proj"] = RowwiseParallel()
                for projection in ("gate_proj", "up_proj"):
                    plan[f"{name}.decoder.mlp.{projection}"] = ColwiseParallel()
                plan[f"{name}.decoder.mlp.down_proj"] = RowwiseParallel()
        local_batch = args.global_batch // args.dp
        example = torch.zeros(
            local_batch // args.microbatches,
            args.sequence_length,
            args.width,
            device=device,
        )
        trainer = DistributedTrainer(
            stage,
            config,
            example,
            nn.MSELoss(),
            tensor_plan=plan if args.tp > 1 else None,
        )
        optimizer = torch.optim.SGD(
            trainer.parameters(), lr=0.01, momentum=0.9, foreach=False
        )
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.01, momentum=0.9, foreach=False
        )
        args.output.mkdir(parents=True, exist_ok=True)
        resume_step = (
            trainer.load_checkpoint(args.resume_from, optimizer)
            if args.resume_from
            else 0
        )
        if resume_step >= args.steps:
            raise ValueError("resume step must precede the requested final step")
        errors = {"loss": 0.0, "output": 0.0, "gradient": 0.0, "parameter": 0.0}
        durations = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.steps):
            generator = torch.Generator(device=device).manual_seed(2000 + step)
            inputs = torch.randn(
                args.global_batch,
                args.sequence_length,
                args.width,
                generator=generator,
                device=device,
            )
            targets = torch.randn(inputs.shape, generator=generator, device=device)
            local_inputs = inputs.chunk(args.dp)[replica]
            local_targets = targets.chunk(args.dp)[replica]
            reference_optimizer.zero_grad(set_to_none=True)
            expected = reference(inputs)
            expected_loss = nn.functional.mse_loss(expected, targets)
            expected_loss.backward()
            if step < resume_step:
                reference_optimizer.step()
                continue
            optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = trainer.forward_backward(local_inputs, local_targets)
            torch.cuda.synchronize()
            duration = time.perf_counter() - started
            errors["loss"] = max(
                errors["loss"], abs(result.loss - expected_loss.item())
            )
            if result.output is not None:
                errors["output"] = max(
                    errors["output"],
                    maximum_error(result.output, expected.chunk(args.dp)[replica]),
                )
            expected_parameters = dict(reference.named_parameters())
            for name, parameter in trainer.named_parameters():
                gradient = full_tensor(parameter.grad)
                errors["gradient"] = max(
                    errors["gradient"],
                    maximum_error(gradient, expected_parameters[name].grad),
                )
            optimizer.step()
            reference_optimizer.step()
            for name, parameter in trainer.named_parameters():
                actual = full_tensor(parameter.detach())
                errors["parameter"] = max(
                    errors["parameter"],
                    maximum_error(actual, expected_parameters[name]),
                )
            if step >= args.warmup:
                durations.append(duration)
            if step == 2:
                trainer.save_checkpoint(
                    args.output / "checkpoint", optimizer, step=step + 1
                )
                try:
                    trainer.save_checkpoint(
                        args.output / "checkpoint", optimizer, step=step + 1
                    )
                except RuntimeError as error:
                    if "already committed" not in str(error):
                        raise
                else:
                    raise AssertionError("Committed checkpoint was overwritten")
                shard = args.output / "checkpoint" / "rank-00000.pt"
                if dist.get_rank() == 0:
                    original_size = shard.stat().st_size
                    with shard.open("ab") as stream:
                        stream.write(b"corrupt")
                dist.barrier()
                try:
                    trainer.load_checkpoint(args.output / "checkpoint", optimizer)
                except RuntimeError as error:
                    if "checksum mismatch" not in str(error):
                        raise
                else:
                    raise AssertionError("Corrupt checkpoint was accepted")
                if dist.get_rank() == 0:
                    with shard.open("r+b") as stream:
                        stream.truncate(original_size)
                dist.barrier()
                with torch.no_grad():
                    for parameter in trainer.parameters():
                        parameter.add_(1)
                optimizer.state.clear()
                restored = trainer.load_checkpoint(
                    args.output / "checkpoint", optimizer
                )
                if restored != step + 1:
                    raise AssertionError("checkpoint step mismatch")
        maximum = torch.tensor(list(errors.values()), device=device)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        errors = dict(zip(errors, maximum.tolist()))
        if any(
            not torch.isfinite(torch.tensor(value)) or value > 3e-5
            for value in errors.values()
        ):
            raise AssertionError(f"reference mismatch: {errors}")
        elapsed = torch.tensor(sum(durations), device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        peak = torch.tensor(torch.cuda.max_memory_allocated(), device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if dist.get_rank() == 0:
            report = {
                "status": "passed",
                "dp": args.dp,
                "tp": args.tp,
                "pp": args.pp,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(),
                "model": args.model,
                "synthetic": True,
                "precision": "fp32",
                "tf32": False,
                "width": args.width,
                "sequence_length": args.sequence_length,
                "global_batch": args.global_batch,
                "microbatches": args.microbatches,
                "steps": args.steps,
                "warmup": args.warmup,
                "max_absolute_errors": errors,
                "tolerance": 3e-5,
                "resume": f"fresh process from step {resume_step}"
                if args.resume_from
                else "rank-local model and SGD momentum restored after step 3; subsequent reference parity",
                "checkpoint_guards": "tested during source run"
                if args.resume_from
                else "immutable manifest and corrupt-rank rejection on all ranks",
                "measured_steps": len(durations),
                "initialization_check": "non-source replicas deliberately perturbed before trainer setup",
                "optimizer": "SGD(momentum=0.9, foreach=False)",
                "forward_backward_samples_per_second": args.global_batch
                * len(durations)
                / elapsed.item(),
                "timing_scope": "forward/backward and loss reduction; excludes optimizer, input generation, reference, checkpoints",
                "peak_allocated_bytes_including_reference": int(peak.item()),
                "limitations": "Not real-data quality or end-to-end training performance; full 3D not validated",
            }
            (args.output / "result.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(report), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
