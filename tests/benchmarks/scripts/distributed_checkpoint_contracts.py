# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午5:40
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Real CUDA/DDP checkpoint continuation contract; no mock or CPU fallback.

Run from the repository root on a single host with at least two GPUs:
    timeout --signal=TERM --kill-after=10s 180s torchrun --standalone \
        --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
        tests/benchmarks/scripts/distributed_checkpoint_contracts.py \
        --timeout-seconds=120 --output-dir=/tmp/parascale-checkpoint-contracts

Artifacts use a fresh run subdirectory and are retained for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import uuid
from datetime import timedelta
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--save-step", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.save_step < args.steps <= 32:
        parser.error("require 1 <= save-step < steps <= 32")
    if not 10 <= args.timeout_seconds <= 600:
        parser.error("timeout-seconds must be between 10 and 600")
    return args


def run(args):
    # Set before importing Torch or creating a CUDA context.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    import numpy as np
    import torch
    import torch.distributed as dist

    from parascale import CheckpointManager, ParaScaleConfig
    from parascale.runtime.training import TrainEngine
    from parascale.strategy import StrategyPlan

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if (
        world_size < 2
        or int(os.environ.get("LOCAL_WORLD_SIZE", world_size)) != world_size
    ):
        raise RuntimeError(
            "This contract requires single-host torchrun with at least two ranks"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError(
            "One visible CUDA GPU per rank is required; no fallback is permitted"
        )
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout_seconds))
    device = torch.device("cuda", local_rank)

    def check(operation):
        error = None
        try:
            operation()
        except Exception as exc:
            error = f"rank {rank}: {type(exc).__name__}: {exc}"
        errors = [None] * world_size
        dist.all_gather_object(errors, error)
        failures = [item for item in errors if item is not None]
        if failures:
            raise AssertionError("; ".join(failures))

    def equal(actual, expected, path="state"):
        if isinstance(expected, torch.Tensor):
            assert isinstance(actual, torch.Tensor), path
            assert actual.dtype == expected.dtype and actual.shape == expected.shape, (
                path
            )
            assert torch.equal(actual.cpu(), expected.cpu()), f"{path}: tensors differ"
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys(), f"{path}: keys differ"
            for key in expected:
                equal(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, (tuple, list)):
            assert len(actual) == len(expected), f"{path}: length differs"
            for index, value in enumerate(expected):
                equal(actual[index], value, f"{path}[{index}]")
        else:
            assert actual == expected, f"{path}: {actual!r} != {expected!r}"

    def rng_state():
        algorithm, keys, position, has_gauss, cached = np.random.get_state()
        return {
            "python": random.getstate(),
            "numpy": (
                algorithm,
                keys.tolist(),
                int(position),
                int(has_gauss),
                float(cached),
            ),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        }

    def seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    class ReplaySource:
        """Deterministic batches with observable process RNG consumption."""

        def __len__(self):
            return args.steps * 2

        def __iter__(self):
            self.cursor = 0
            self.consume_rng()
            return self

        @staticmethod
        def consume_rng():
            random.random()
            np.random.random()
            torch.rand(1)
            torch.rand(1, device=device)

        def __next__(self):
            if self.cursor == len(self):
                raise StopIteration
            self.consume_rng()
            batch = torch.arange(128, device=device, dtype=torch.float32).reshape(8, 16)
            batch = batch / 128 + (rank + self.cursor) / 100
            self.cursor += 1
            return batch

    def create_engine():
        config = ParaScaleConfig(
            training_backend="native_ddp",
            precision="fp16",
            gradient_accumulation_steps=2,
            ddp_comm_hook="none",
            ddp_bucket_cap_mb=25,
            ddp_gradient_as_bucket_view=False,
        )
        return TrainEngine(
            config=config,
            strategy_plan=StrategyPlan(
                backend="native_ddp", dp_size=world_size, precision="fp16"
            ),
        )

    def create_model():
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.35),
            torch.nn.Linear(32, 4),
        ).to(device)
        return model, torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    def loss_fn(output, _batch):
        assert output.dtype == torch.float16, "FP16 autocast must actually execute"
        return output.float().square().mean() * 0.001

    run_id = [uuid.uuid4().hex if rank == 0 else None]
    dist.broadcast_object_list(run_id, src=0)
    root = args.output_dir.resolve() / run_id[0]
    manager = CheckpointManager(str(root))
    seed(71)
    model, optimizer = create_model()
    continuous = create_engine()
    continuous.amp_scaler = torch.cuda.amp.GradScaler(init_scale=128, growth_interval=1)
    seed(1000 + rank)
    continuous.fit(
        ReplaySource(),
        max_steps=args.steps,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        checkpoint_manager=manager,
        checkpoint_interval=args.save_step,
    )
    torch.cuda.synchronize()
    expected_parameters = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    expected_rng = rng_state()
    expected_scaler = continuous.precision.state_dict()
    expected_optimizer = optimizer.state_dict()
    manifest = manager.read_manifest(args.save_step)
    payload = None

    def validate_sidecars():
        nonlocal payload
        assert manager.validate_manifest(manifest).ok, "manifest integrity failure"
        entries = [
            entry for entry in manifest.files if entry.get("role") == "runtime_state"
        ]
        assert len(entries) == world_size
        assert {entry["rank"] for entry in entries} == set(range(world_size))
        assert len({entry["path"] for entry in entries}) == world_size
        all_payloads = []
        for entry in sorted(entries, key=lambda item: item["rank"]):
            path = manager.resolve_payload_path(manifest, entry)
            state = torch.load(path, map_location="cpu", weights_only=True)
            assert state["rank"] == entry["rank"] and state["world_size"] == world_size
            assert (
                state["precision_state_dict"]["amp_scaler"]["scale"]
                == 128 * 2**args.save_step
            )
            all_payloads.append(state)
        for left in range(world_size):
            for right in range(left):
                a, b = all_payloads[left]["rng_state"], all_payloads[right]["rng_state"]
                assert a["python"] != b["python"], "Python rank streams collapsed"
                assert a["numpy"] != b["numpy"], "NumPy rank streams collapsed"
                assert not torch.equal(a["torch_cpu"], b["torch_cpu"]), (
                    "CPU rank streams collapsed"
                )
                assert not torch.equal(a["torch_cuda"][left], b["torch_cuda"][right]), (
                    "CUDA rank streams collapsed"
                )
        payload = all_payloads[rank]

    check(validate_sidecars)
    seed(9000 + rank)
    resumed_model, resumed_optimizer = create_model()
    resumed = create_engine()
    resumed.load_checkpoint(
        manager, args.save_step, model=resumed_model, optimizer=resumed_optimizer
    )
    check(lambda: equal(rng_state(), payload["rng_state"], "loaded_rank_rng"))
    check(
        lambda: equal(
            resumed.precision.state_dict(),
            payload["precision_state_dict"],
            "pending_scaler",
        )
    )
    resumed.fit(
        ReplaySource(),
        max_steps=args.steps - args.save_step,
        model=resumed_model,
        optimizer=resumed_optimizer,
        loss_fn=loss_fn,
    )
    torch.cuda.synchronize()

    def validate_continuation():
        assert isinstance(
            resumed.training_backend.model, torch.nn.parallel.DistributedDataParallel
        )
        assert resumed.amp_scaler is not None and resumed.amp_scaler.is_enabled()
        assert resumed.state.global_step == continuous.state.global_step == args.steps
        assert (
            resumed.state.consumed_micro_batches
            == continuous.state.consumed_micro_batches
            == args.steps * 2
        )
        assert expected_scaler["amp_scaler"]["scale"] == 128 * 2**args.steps
        equal(resumed_model.state_dict(), expected_parameters, "parameters")
        equal(resumed_optimizer.state_dict(), expected_optimizer, "optimizer")
        equal(resumed.precision.state_dict(), expected_scaler, "scaler")
        equal(rng_state(), expected_rng, "continuation_rng")

    check(validate_continuation)
    digest = hashlib.sha256(
        expected_rng["torch_cuda"][local_rank].numpy().tobytes()
    ).hexdigest()
    ranks = [None] * world_size
    dist.all_gather_object(
        ranks, {"rank": rank, "cuda_rng_sha256": digest, "exact_continuation": True}
    )
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "backend": "native_ddp",
                    "precision": "fp16",
                    "gradient_accumulation_steps": 2,
                    "world_size": world_size,
                    "steps": args.steps,
                    "save_step": args.save_step,
                    "checkpoint_dir": str(root),
                    "ranks": ranks,
                }
            ),
            flush=True,
        )


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    started = time.monotonic()

    def deadline():
        print(
            json.dumps(
                {"status": "timeout", "rank": rank, "seconds": args.timeout_seconds}
            ),
            file=sys.stderr,
            flush=True,
        )
        os._exit(124)

    watchdog = threading.Timer(args.timeout_seconds, deadline)
    watchdog.daemon = True
    watchdog.start()
    try:
        run(args)
        return 0
    except Exception as exc:
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        raise
    finally:
        dist = sys.modules.get("torch.distributed")
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()
        watchdog.cancel()


if __name__ == "__main__":
    raise SystemExit(main())
