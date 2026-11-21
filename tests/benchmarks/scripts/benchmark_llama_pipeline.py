# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午6:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Real PP timing/retention diagnostics; run unchanged against each source snapshot.

Timing sums synchronized forward/backward and optimizer segments, excluding H2D,
preprocessing and memory inspection. Caller-owned results are deleted before the
optimizer in both output modes. Cache bytes count unique local backing storage
within each container, not autograd-saved tensors; containers may overlap.
Final exact state fingerprints run only after timing and peak capture, without
writing checkpoints. Compare per-rank fingerprints/losses across source runs.
"""

import argparse
import hashlib
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

import parascale
from parascale.parallel import CausalLMLoss, DistributedParallelConfig
from parascale.parallel.execution import DistributedTrainer
from parascale.parallel.llama import LlamaStage
from tests.benchmarks.scripts.benchmark_llama_parallel import (
    file_digest,
    summarize_timing,
)
from tests.benchmarks.scripts.validate_llama_checkpoint import state_fingerprint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "data-file", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--pp", type=int, choices=(2,), default=2)
    parser.add_argument(
        "--pipeline-schedule", choices=("gpipe", "1f1b"), default="gpipe"
    )
    for name, default in (
        ("microbatches", 2),
        ("global-batch", 16),
        ("sequence-length", 256),
        ("steps", 16),
        ("warmup", 4),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--return-outputs", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--cpu-affinity", type=int, nargs="+")
    args = parser.parse_args(argv)
    try:
        DistributedParallelConfig(
            args.dp,
            args.tp,
            args.pp,
            args.microbatches,
            pipeline_schedule=args.pipeline_schedule,
        )
    except ValueError as error:
        parser.error(str(error))
    if (
        args.microbatches < args.pp
        or args.global_batch <= 0
        or args.global_batch % (args.dp * args.microbatches)
    ):
        parser.error(
            "require microbatches >= pp and positive global-batch divisible by dp * microbatches"
        )
    if args.sequence_length < 2 or not 0 <= args.warmup < args.steps:
        parser.error("require sequence-length >= 2 and 0 <= warmup < steps")
    if args.cpu_affinity is not None and any(cpu < 0 for cpu in args.cpu_affinity):
        parser.error("CPU affinity indices must be nonnegative")
    if args.output.exists():
        parser.error("output already exists; select a fresh result path")
    return args


def cache_memory(stage):
    """Observe caches without retaining tensors or counting aliased storage twice."""

    def storage_bytes(value):
        storages = {}

        def visit(item):
            if isinstance(item, torch.Tensor):
                local = item.to_local() if hasattr(item, "to_local") else item
                storage = local.untyped_storage()
                storages[(str(local.device), storage.data_ptr())] = storage.nbytes()
            elif isinstance(item, dict):
                for child in item.values():
                    visit(child)
            elif isinstance(item, (tuple, list)):
                for child in item:
                    visit(child)

        visit(value)
        return sum(storages.values())

    result = {}
    for name in ("output_chunks", "fwd_cache"):
        container = getattr(stage, name)
        result[f"{name}_len"] = len(container)
        result[f"{name}_storage_bytes"] = storage_bytes(container)
    return result


def final_state_fingerprints(module, optimizer):
    """Validate FP32 ownership, then reuse the finite exact-state oracle once."""
    for parameter in module.parameters():
        values = (parameter, parameter.grad, *optimizer.state[parameter].values())
        if any(
            not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            for value in values
        ):
            raise ValueError("Model/gradient/optimizer state must remain FP32")
    return {
        "model": state_fingerprint(module.state_dict()),
        "optimizer": state_fingerprint(optimizer.state_dict()),
    }


def main():
    args = parse_args()
    if (
        str(torch.__version__).split("+")[0] != "2.9.1"
        or transformers.__version__ != "4.51.3"
    ):
        raise RuntimeError("Benchmark requires torch==2.9.1 and transformers==4.51.3")
    config = DistributedParallelConfig(
        args.dp,
        args.tp,
        args.pp,
        args.microbatches,
        pipeline_schedule=args.pipeline_schedule,
    )
    config.validate_world_size(int(os.environ["WORLD_SIZE"]))
    if int(os.environ["LOCAL_WORLD_SIZE"]) != config.world_size:
        raise ValueError("Benchmark requires a single CUDA node")
    if args.cpu_affinity is not None:
        if not hasattr(os, "sched_setaffinity") or not set(
            args.cpu_affinity
        ) <= os.sched_getaffinity(0):
            raise ValueError("Requested CPU affinity is unavailable")
        os.sched_setaffinity(0, set(args.cpu_affinity))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=300))
    try:
        rank = dist.get_rank()
        replica, pp, _ = config.coordinates(rank)
        torch.manual_seed(719)
        torch.backends.cuda.matmul.allow_tf32 = False
        source = LlamaForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        ).train()
        model_config = json.loads(source.config.to_json_string(use_diff=False))
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        text = args.data_file.read_bytes()
        encoded = tokenizer(text.decode("utf-8"), add_special_tokens=False)["input_ids"]
        needed = args.steps * args.global_batch * args.sequence_length
        if len(encoded) < needed:
            raise ValueError("Insufficient real packed tokens for the requested steps")
        tokens = torch.tensor(encoded[:needed], dtype=torch.int64).reshape(
            args.steps, args.global_batch, args.sequence_length
        )
        token_hash = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
        identities = [None] * config.world_size
        dist.all_gather_object(identities, token_hash)
        if any(value != token_hash for value in identities):
            raise ValueError("Packed token identity differs across ranks")
        stage = LlamaStage(source, pp, args.pp, tp_size=args.tp).to(device)
        del source
        microbatch = args.global_batch // args.dp // args.microbatches
        shape = (microbatch, args.sequence_length)
        trainer = DistributedTrainer(
            stage,
            config,
            torch.zeros(
                shape if pp == 0 else (*shape, stage.hidden_size),
                device=device,
                dtype=torch.int64 if pp == 0 else torch.float32,
            ),
            summed_loss=CausalLMLoss(stage.vocab_size),
            tensor_plan=stage.tensor_parallel_plan() or None,
            precision=args.precision,
        )
        optimizer = torch.optim.AdamW(
            trainer.parameters(), lr=1e-5, eps=1e-8, foreach=False
        )
        pipeline = trainer.schedule._stage
        expected_count = args.global_batch * (args.sequence_length - 1)
        record = dict(
            rank=rank,
            gpu_uuid=str(torch.cuda.get_device_properties(device).uuid),
            cpu_affinity=sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            torch_num_threads=torch.get_num_threads(),
            torch_num_interop_threads=torch.get_num_interop_threads(),
            seconds=[],
            losses=[],
            memory_after_backward=[],
            memory_after_step=[],
            peak_allocated_bytes=[],
            peak_reserved_bytes=[],
            cache_after_backward=[],
            cache_after_step=[],
        )
        for cpu_batch in tokens:
            batch = cpu_batch.chunk(args.dp)[replica].to(device)
            optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            result = trainer.forward_backward(
                batch, batch, return_outputs=args.return_outputs
            )
            loss, count = float(result.loss), result.normalization_count
            del result
            torch.cuda.synchronize(device)
            forward_seconds = time.perf_counter() - started
            if not math.isfinite(loss) or count != expected_count:
                raise AssertionError(
                    "Nonfinite loss or incorrect global token denominator"
                )
            record["memory_after_backward"].append(torch.cuda.memory_allocated(device))
            record["cache_after_backward"].append(cache_memory(pipeline))
            started = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize(device)
            record["seconds"].append(forward_seconds + time.perf_counter() - started)
            record["memory_after_step"].append(torch.cuda.memory_allocated(device))
            record["peak_allocated_bytes"].append(
                torch.cuda.max_memory_allocated(device)
            )
            record["peak_reserved_bytes"].append(torch.cuda.max_memory_reserved(device))
            record["cache_after_step"].append(cache_memory(pipeline))
            record["losses"].append(loss)
        record["state_error"] = None
        try:
            record["final_state"] = final_state_fingerprints(trainer.module, optimizer)
        except Exception as error:
            record["state_error"] = f"{type(error).__name__}: {error}"
        records = [None] * config.world_size
        dist.all_gather_object(records, record)
        if any(item["state_error"] is not None for item in records):
            raise AssertionError(
                f"Final state validation failed: {[item['state_error'] for item in records]}"
            )
        if rank == 0:
            root = Path(parascale.__file__).resolve().parent
            seconds = [
                max(item["seconds"][i] for item in records) for i in range(args.steps)
            ]
            report = dict(
                status="diagnostic",
                arguments={
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                torch=str(torch.__version__),
                transformers=str(transformers.__version__),
                model_config=model_config,
                optimizer_defaults=optimizer.defaults,
                asset_sha256={
                    path.name: file_digest(path)
                    for path in sorted(args.model.iterdir())
                    if path.is_file()
                },
                data_sha256=hashlib.sha256(text).hexdigest(),
                token_sha256=token_hash,
                source_sha256={
                    str(path.relative_to(root)): hashlib.sha256(
                        path.read_bytes().replace(b"\r\n", b"\n")
                    ).hexdigest()
                    for path in sorted(root.rglob("*.py"))
                },
                script_sha256=file_digest(Path(__file__)),
                timing_helper_sha256=file_digest(
                    Path(__file__).with_name("benchmark_llama_parallel.py")
                ),
                fingerprint_helper_sha256=file_digest(
                    Path(__file__).with_name("validate_llama_checkpoint.py")
                ),
                precision=trainer.precision,
                parameter_dtype="float32",
                optimizer_state_dtype="float32",
                schedule=type(trainer.schedule).__name__,
                pipeline_schedule=config.pipeline_schedule,
                expected_normalization_count=expected_count,
                seconds=seconds,
                measured_seconds=seconds[args.warmup :],
                records=records,
                scope=__doc__
                + " Diagnostic training only; no convergence or checkpoint qualification.",
            )
            report.update(
                summarize_timing(
                    seconds[args.warmup :],
                    batch=args.global_batch,
                    length=args.sequence_length,
                )
            )
            for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
                report[name] = max(max(item[name][args.warmup :]) for item in records)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
            print(
                json.dumps(
                    {
                        "status": "diagnostic",
                        "tokens_per_second": report["tokens_per_second"],
                    }
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
