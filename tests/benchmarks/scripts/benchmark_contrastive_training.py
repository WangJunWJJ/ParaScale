# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午11:48
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Cached-real-batch CLIP update timing; run the numerical parity gate first."""

import argparse
import inspect
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from torch.distributed.nn.functional import all_gather
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import CLIPModel

from parascale.config import ParaScaleConfig
from parascale.runtime.training import TrainEngine
from parascale.workloads.clip import build_clip_contrastive_components
from parascale.workloads.specs.clip import ClipContrastiveSpec
from tests.benchmarks.scripts.validate_contrastive_training import (
    eager,
    file_hash,
    json_hash,
    load_batch,
    state_hash,
)

WARMUP, WINDOWS, STEPS = 10, 3, 50


def profile_run(directory, stem, rank):
    """Optionally capture three warmed-up updates, without changing the workload."""
    if directory is None:
        return nullcontext(None)
    directory.mkdir(parents=True, exist_ok=True)
    trace = directory / f"{stem}.rank{rank}.trace.json"
    if trace.exists():
        raise FileExistsError(trace)
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=WARMUP, warmup=1, active=3, repeat=1),
        on_trace_ready=lambda profiler: profiler.export_chrome_trace(str(trace)),
        record_shapes=True,
    )


def runtime_config(world):
    return ParaScaleConfig(
        training_backend="native_ddp" if world > 1 else "native",
        data_parallel_size=world,
        precision="fp32",
        grad_clip_norm=None,
        ddp_comm_hook="none",
        gradient_accumulation_steps=1,
        metrics_history_limit=1,
    )


class NativeFeatures(CLIPModel):
    """Materialize the trainable scale inside the DDP forward owner."""

    def forward(self, **batch):
        output = super().forward(**batch)
        return (
            output.image_embeds,
            output.text_embeds,
            self.logit_scale.exp().reshape(()),
        )


def native_loss(output, rank, world):
    images, texts, scale = output
    pairs = torch.cat((images, texts), dim=1)
    global_pairs = torch.cat(all_gather(pairs), dim=0) if world > 1 else pairs
    all_images, all_texts = global_pairs.chunk(2, dim=1)
    targets = (
        torch.arange(images.shape[0], device=images.device) + rank * images.shape[0]
    )
    return (
        F.cross_entropy((images @ all_texts.T) * scale, targets)
        + F.cross_entropy((texts @ all_images.T) * scale, targets)
    ) * 0.5


def finite_state(model, optimizer):
    values = list(model.parameters())
    values += [
        v
        for state in optimizer.state.values()
        for v in state.values()
        if isinstance(v, torch.Tensor)
    ]
    valid = torch.stack([torch.isfinite(v).all() for v in values]).all().to(torch.int32)
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not valid.item():
        raise FloatingPointError("nonfinite model or optimizer state on a rank")


class Timing:
    def __init__(self, model, optimizer, device, profiler=None):
        self.model, self.optimizer, self.device = model, optimizer, device
        self.profiler = profiler
        self.index, self.windows, self.steps, self.losses = 0, [], [], []

    def start(self):
        boundary = self.index == 0 or (
            self.index >= WARMUP and (self.index - WARMUP) % STEPS == 0
        )
        if boundary:
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self.step_start = time.perf_counter()
        if boundary:
            self.window_start = self.step_start

    def finish(self, loss):
        torch.cuda.synchronize()
        end = time.perf_counter()
        self.steps.append(end - self.step_start)
        self.losses.append(float(loss))
        self.index += 1
        boundary = self.index == WARMUP or (
            self.index > WARMUP and (self.index - WARMUP) % STEPS == 0
        )
        if boundary:
            record = {
                "steps": len(self.steps),
                "wall_seconds": end - self.window_start,
                "step_seconds": self.steps,
                "losses": self.losses,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            self.windows.append(record)
            bad = torch.tensor(
                int(not all(map(math.isfinite, self.losses))), device=self.device
            )
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            if bad.item():
                raise FloatingPointError("nonfinite loss on a rank")
            finite_state(self.model, self.optimizer)
            self.steps, self.losses = [], []
        if self.profiler is not None:
            self.profiler.step()


def run(args, rank, world, device, profiler=None):
    batch, data = load_batch(args)
    identities = [None] * world
    dist.all_gather_object(identities, data)
    if any(item != data for item in identities):
        raise ValueError("different global data/tensor identities across ranks")
    local = args.global_batch // world
    batch = {
        k: v[rank * local : (rank + 1) * local].to(device) for k, v in batch.items()
    }
    torch.manual_seed(42)
    if args.implementation == "parascale":
        spec = ClipContrastiveSpec(
            model_type="hf_clip",
            data_type="datacomp_wds",
            pretrained_model_name_or_path=str(args.model),
            data_dir=str(args.data_dir),
            streaming=True,
            batch_size=local,
            num_batches=1,
            text_length=77,
        )
        model, unused_optimizer, unused_batches, objective = (
            build_clip_contrastive_components(spec)
        )
        del unused_optimizer, unused_batches
        hf = eager(model.wrapped_model)
    else:
        hf = eager(
            NativeFeatures.from_pretrained(
                args.model,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
        )
        model = hf
    for name in ("logit_scale", "logit_bias"):
        value = getattr(hf, name, None)
        if value is not None and value.ndim == 0:
            setattr(hf, name, torch.nn.Parameter(value.detach().reshape(1)))
    model.to(device)
    identity = {"initial_state_sha256": state_hash(hf), "config": hf.config.to_dict()}
    identity["config_sha256"] = json_hash(identity["config"])
    dist.all_gather_object(identities, identity)
    if any(item != identity for item in identities):
        raise ValueError("different initial model/config across ranks")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-4, momentum=0.9, foreach=False
    )
    timing = Timing(hf, optimizer, device, profiler)
    total = WARMUP + WINDOWS * STEPS
    torch.manual_seed(42)
    if args.implementation == "parascale":
        engine = TrainEngine(runtime_config(world))
        original_record = engine.record_step

        def record(metrics=None):
            state = original_record(metrics)
            timing.finish(metrics["loss"])
            return state

        def batches():
            for _ in range(total):
                timing.start()
                yield batch

        engine.record_step = record
        engine.fit(
            batches(),
            max_steps=total,
            model=model,
            optimizer=optimizer,
            loss_fn=objective,
        )
        if engine.state.global_step != total:
            raise RuntimeError("product did not complete every optimizer update")
    else:
        if world > 1:
            model = DDP(model, device_ids=[device.index], gradient_as_bucket_view=True)
        for _ in range(total):
            timing.start()
            loss = native_loss(model(**batch), rank, world)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            timing.finish(loss.item())
            del loss
    sources = {
        name: file_hash(module.__file__)
        for name, module in list(sys.modules.items())
        if name.startswith("parascale")
        and (getattr(module, "__file__", None) or "").endswith(".py")
    }
    return {
        "rank": rank,
        "gpu": torch.cuda.get_device_name(device),
        "windows": timing.windows,
        "final_state_sha256": state_hash(hf),
        "source_sha256": json_hash(sources),
    }, {
        "data": data,
        "model": identity,
        "source": sources,
        "script_sha256": file_hash(__file__),
        "validator_sha256": file_hash(inspect.getfile(load_batch)),
        "hf_clip_source_sha256": file_hash(inspect.getfile(CLIPModel)),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", choices=["parascale", "pytorch"], required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--global-batch", type=int, choices=[32], default=32)
    args = parser.parse_args()
    if args.output.exists() or not args.model.is_dir() or not args.data_dir.is_dir():
        parser.error("requires local model/data directories and a new output path")
    rank, world, local = (
        int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    if world not in (1, 2, 4) or not torch.cuda.is_available():
        parser.error("requires torchrun with 1/2/4 CUDA ranks")
    runtime_config(world)  # Validate before loading weights or initializing CUDA.
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    try:
        with profile_run(args.profile_dir, args.output.stem, rank) as profiler:
            result, provenance = run(
                args, rank, world, torch.device("cuda", local), profiler
            )
        results = [None] * world
        dist.all_gather_object(results, result)
        if len({r["source_sha256"] for r in results}) != 1:
            raise ValueError("ranks executed different product sources")
        windows = [
            {
                "window": i,
                "max_rank_wall_seconds": max(
                    r["windows"][i]["wall_seconds"] for r in results
                ),
                "global_pairs_per_second": 32
                * STEPS
                / max(r["windows"][i]["wall_seconds"] for r in results),
            }
            for i in range(1, WINDOWS + 1)
        ]
        report = {
            "status": "passed",
            "measurement_kind": (
                "diagnostic_profile" if args.profile_dir else "throughput"
            ),
            "performance_claim_eligible": args.profile_dir is None,
            "implementation": args.implementation,
            "world_size": world,
            "global_batch": 32,
            "warmup": WARMUP,
            "steps_per_window": STEPS,
            "precision": "fp32",
            "attention": "eager",
            "tf32": False,
            "gradient_accumulation_steps": 1,
            "grad_clip_norm": None,
            "comm_hook": "none",
            "model_path": str(args.model.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "product_source_path": inspect.getfile(TrainEngine),
            "optimizer": {"type": "sgd", "lr": 1e-4, "momentum": 0.9},
            "windows": windows,
            "ranks": results,
            "provenance": provenance,
            "scope": "cached identical 32 pairs repeated; synchronized update windows include engine CPU bookkeeping; exclude decoding, tokenization, H2D, setup and boundary finite scans; not end-to-end data-pipeline throughput or a parity certificate",
        }
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
    except Exception as exc:
        target = (
            args.output
            if rank == 0
            else args.output.with_name(f"{args.output.stem}.rank{rank}.failure.json")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "status": "failed",
                    "rank": rank,
                    "error": repr(exc),
                    "implementation": args.implementation,
                    "script_sha256": file_hash(__file__),
                },
                stream,
                indent=2,
            )
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
