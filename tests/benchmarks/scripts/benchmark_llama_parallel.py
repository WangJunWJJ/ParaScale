# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Matched Llama TP throughput and validation NLL; not production acceptance."""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as functional
import transformers
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    parallelize_module,
)
from transformers import AutoTokenizer, LlamaForCausalLM


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summed_loss(prediction, labels):
    return functional.cross_entropy(
        prediction[:, :-1].float().reshape(-1, prediction.shape[-1]),
        labels[:, 1:].reshape(-1),
        reduction="sum",
    )


def native_tensor_plan(layers, *, shared=True):
    """Build the native HF reference plan independently of ParaScale adapters."""
    plan = {}
    for index in range(layers):
        prefix = f"model.layers.{index}"
        if shared:
            plan[f"{prefix}.self_attn"] = PrepareModuleInput(
                input_kwarg_layouts={"hidden_states": Replicate()},
                desired_input_kwarg_layouts={"hidden_states": Replicate()},
            )
            plan[f"{prefix}.mlp"] = PrepareModuleInput(
                input_layouts=Replicate(), desired_input_layouts=Replicate()
            )
        for projection in ("q_proj", "k_proj", "v_proj"):
            plan[f"{prefix}.self_attn.{projection}"] = ColwiseParallel()
        plan[f"{prefix}.self_attn.o_proj"] = RowwiseParallel()
        for projection in ("gate_proj", "up_proj"):
            plan[f"{prefix}.mlp.{projection}"] = ColwiseParallel()
        plan[f"{prefix}.mlp.down_proj"] = RowwiseParallel()
    return plan


def parse_args(argv=None):
    """Reject invalid workload budgets before initializing CUDA or distributed state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--driver", choices=("native", "parascale"), required=True)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--global-batch", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--validation-tokens", type=int, default=8192)
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--native-plan", choices=("shared", "layerwise"), default="shared"
    )
    parser.add_argument("--cpu-affinity", type=int, nargs="+")
    args = parser.parse_args(argv)
    if not args.steps > args.warmup >= 0:
        parser.error("steps must exceed nonnegative warmup")
    if (
        args.microbatch <= 0
        or args.global_batch <= 0
        or args.global_batch % args.microbatch
    ):
        parser.error("global-batch must be a positive multiple of microbatch")
    if (
        args.sequence_length < 2
        or args.validation_tokens <= 0
        or args.validation_tokens % args.sequence_length
    ):
        parser.error(
            "validation-tokens must be a positive multiple of sequence-length >= 2"
        )
    if not math.isfinite(args.adam_eps) or args.adam_eps <= 0:
        parser.error("adam-eps must be finite and positive")
    if args.cpu_affinity is not None and any(cpu < 0 for cpu in args.cpu_affinity):
        parser.error("CPU affinity indices must be nonnegative")
    if args.driver == "parascale" and args.native_plan != "shared":
        parser.error("native-plan only selects the native baseline")
    return args


def summarize_timing(seconds, *, batch, length):
    """Distinguish aggregate throughput from inverse median step latency."""
    measured_tokens = len(seconds) * batch * length
    return {
        "tokens_per_second": measured_tokens / sum(seconds),
        "scored_tokens_per_second": len(seconds) * batch * (length - 1) / sum(seconds),
        "median_step_tokens_per_second": batch * length / statistics.median(seconds),
    }


def main():
    args = parse_args()
    if (
        torch.__version__.split("+")[0] != "2.9.1"
        or transformers.__version__ != "4.51.3"
    ):
        raise RuntimeError("Benchmark requires torch==2.9.1 and transformers==4.51.3")
    if (
        int(os.environ["WORLD_SIZE"]) != args.tp
        or int(os.environ["LOCAL_WORLD_SIZE"]) != args.tp
    ):
        raise ValueError("This benchmark requires a single node with exactly tp ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if args.cpu_affinity is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise ValueError("CPU affinity is unsupported on this platform")
        requested = set(args.cpu_affinity)
        if not requested <= os.sched_getaffinity(0):
            raise ValueError("Requested CPU affinity is outside the allowed set")
        os.sched_setaffinity(0, requested)
    device = torch.device("cuda", local_rank)
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported(
        including_emulation=False
    ):
        raise ValueError("BF16 benchmark requires native CUDA BF16 support")
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=180))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.manual_seed(719)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = LlamaForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        ).to(device)
        model_configuration = model.config.to_dict()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        tokens = {}
        text_hashes = {}
        for split, count in (
            ("train", args.steps * args.global_batch * args.sequence_length),
            ("validation", args.validation_tokens),
        ):
            text = (args.data / f"{split}.txt").read_bytes()
            text_hashes[split] = hashlib.sha256(text).hexdigest()
            encoded = tokenizer(text.decode("utf-8"), add_special_tokens=False)[
                "input_ids"
            ]
            if len(encoded) < count:
                raise ValueError(f"Insufficient packed tokens in {split}")
            tokens[split] = torch.tensor(encoded[:count], dtype=torch.int64)
        if text_hashes["train"] == text_hashes["validation"]:
            raise ValueError("Training and validation text must differ")
        train = tokens["train"].reshape(
            args.steps, args.global_batch, args.sequence_length
        )
        validation = tokens["validation"].reshape(-1, args.sequence_length).to(device)
        input_hashes = {
            key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
            for key, value in tokens.items()
        }
        all_hashes = [None] * world
        dist.all_gather_object(all_hashes, input_hashes)
        assert all(item == input_hashes for item in all_hashes)
        asset_hashes = (
            {
                path.name: file_digest(path)
                for path in sorted(args.model.iterdir())
                if path.is_file()
            }
            if rank == 0
            else {}
        )
        trainer = None
        plan = {}
        if args.driver == "parascale":
            from parascale.parallel import CausalLMLoss, DistributedParallelConfig
            from parascale.parallel.execution import DistributedTrainer
            from parascale.parallel.llama import LlamaStage

            model = LlamaStage(model, tp_size=world)
            plan = model.tensor_parallel_plan()
            trainer = DistributedTrainer(
                model,
                DistributedParallelConfig(
                    tp_size=world, microbatches=args.global_batch // args.microbatch
                ),
                torch.zeros(
                    args.microbatch,
                    args.sequence_length,
                    dtype=torch.int64,
                    device=device,
                ),
                summed_loss=CausalLMLoss(model.vocab_size),
                tensor_plan=plan or None,
                precision=args.precision,
            )
        elif world > 1:
            mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("tp",))
            plan = native_tensor_plan(
                model.config.num_hidden_layers, shared=args.native_plan == "shared"
            )
            parallelize_module(model, mesh, plan)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-5, eps=args.adam_eps, foreach=False
        )

        def forward(batch):
            return (
                model(batch)
                if args.driver == "parascale"
                else model(batch, use_cache=False).logits
            )

        def evaluate():
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in validation.split(4):
                    with torch.autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                        enabled=args.precision == "bf16",
                        cache_enabled=False,
                    ):
                        value = summed_loss(forward(batch), batch).item()
                    assert math.isfinite(value)
                    losses.append(value)
            model.train()
            return sum(losses) / (
                validation.shape[0] * (args.sequence_length - 1)
            ), losses

        initial_nll, initial_blocks = evaluate()
        torch.cuda.reset_peak_memory_stats(device)
        durations, training_losses = [], []
        for step, batch in enumerate(train):
            batch = batch.to(device)
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if trainer is not None:
                loss_value = trainer.forward_backward(batch, batch).loss
            else:
                loss_sum = torch.zeros((), device=device)
                for microbatch in batch.split(args.microbatch):
                    with torch.autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                        enabled=args.precision == "bf16",
                        cache_enabled=False,
                    ):
                        prediction = forward(microbatch)
                        loss = summed_loss(prediction, microbatch) * (
                            1 / (args.global_batch * (args.sequence_length - 1))
                        )
                    loss.backward()
                    loss_sum += loss.detach()
                    del prediction, loss
                loss_value = loss_sum.item()
            optimizer.step()
            torch.cuda.synchronize()
            duration = torch.tensor(time.perf_counter() - start, device=device)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            assert math.isfinite(loss_value)
            if step >= args.warmup:
                durations.append(duration.item())
            training_losses.append(loss_value)
        peak = torch.tensor(torch.cuda.max_memory_allocated(), device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        final_nll, final_blocks = evaluate()
        finite = torch.ones((), dtype=torch.int32, device=device)
        with torch.no_grad():
            for parameter in model.parameters():
                assert parameter.dtype == torch.float32
                local = (
                    parameter.to_local()
                    if hasattr(parameter, "to_local")
                    else parameter
                )
                finite *= torch.isfinite(local).all().to(torch.int32)
                for value in optimizer.state[parameter].values():
                    assert value.dtype == torch.float32
                    local = value.to_local() if hasattr(value, "to_local") else value
                    finite *= torch.isfinite(local).all().to(device, torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        assert finite.item() == 1
        records = [None] * world
        dist.all_gather_object(
            records,
            {
                "rank": rank,
                "initial_nll": initial_nll,
                "final_nll": final_nll,
                "initial_block_sums": initial_blocks,
                "final_block_sums": final_blocks,
                "losses": training_losses,
                "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
                "cpu_affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
            },
        )
        if rank == 0:
            import parascale

            root = Path(parascale.__file__).parent
            result = {
                "status": "diagnostic",
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "asset_sha256": asset_hashes,
                "corpus_sha256": text_hashes,
                "input_sha256": input_hashes,
                "script_sha256": file_digest(Path(__file__)),
                "source_sha256": {
                    str(path.relative_to(root)): hashlib.sha256(
                        path.read_bytes().replace(b"\r\n", b"\n")
                    ).hexdigest()
                    for path in root.rglob("*.py")
                },
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "model_config": model_configuration,
                "optimizer_defaults": optimizer.defaults,
                "tp_plan_styles": {
                    name: type(style).__name__ for name, style in plan.items()
                },
                "cpu_affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "world": world,
                "precision": args.precision,
                "parameter_dtype": "float32",
                "optimizer_state_dtype": "float32",
                "global_batch": args.global_batch,
                "microbatch": args.microbatch,
                "sequence_length": args.sequence_length,
                "validation_scored_tokens": validation.shape[0]
                * (args.sequence_length - 1),
                "training_scored_tokens": args.steps
                * args.global_batch
                * (args.sequence_length - 1),
                "seconds": durations,
                **summarize_timing(
                    durations, batch=args.global_batch, length=args.sequence_length
                ),
                "peak_allocated_bytes": peak.item(),
                "records": records,
                "scope": "Short fixed-shape training with FP32 master parameters and held-out WikiText validation; timings exclude preprocessing, H2D and evaluation; not convergence or production qualification",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "driver": args.driver,
                        "world": world,
                        "initial_nll": initial_nll,
                        "final_nll": final_nll,
                        "tokens_per_second": result["tokens_per_second"],
                        "peak_allocated_bytes": peak.item(),
                    }
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
