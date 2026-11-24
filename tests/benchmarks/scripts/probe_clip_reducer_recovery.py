# -*- coding: utf-8 -*-
# @Time : 2026/11/23 下午2:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Two-update cold/warm DDP diagnostic; not a throughput or recovery gate.

Run cold then warm in separate four-rank torchrun launches with the same output
directory and immutable checkpoint. Only rank zero retains tensor snapshots.
"""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from parascale.commands.common import load_config_file
from parascale.runtime.train_runner import run_train_from_config
from parascale.runtime.training.coordination import coordinated_call
from parascale.runtime.training.fit_loop import FitLoopRunner
from parascale.runtime.training.random_state import capture_rng_state
from tests.benchmarks.tools.prepare_flickr8k_retrieval import file_hash


def fingerprint(value):
    digest = hashlib.sha256()

    def visit(item):
        digest.update(type(item).__name__.encode())
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
        digest.update(b";")

    visit(value)
    return digest.hexdigest()


def tensor_differences(left, right):
    assert left.keys() == right.keys(), "tensor names differ"
    result = {"differing_elements": 0, "outside_tolerance": 0, "max_abs": 0.0}
    changed = []
    for name, a in left.items():
        b = right[name]
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
        count = int((a != b).sum())
        if not count:
            continue
        delta = (a.double() - b.double()).abs()
        outside = int((delta > 1e-5 + 1e-4 * b.double().abs()).sum())
        maximum = float(delta.max())
        result["differing_elements"] += count
        result["outside_tolerance"] += outside
        result["max_abs"] = max(result["max_abs"], maximum)
        changed.append({"name": name, "count": count, "max_abs": maximum})
    result["changed"] = sorted(changed, key=lambda row: row["max_abs"], reverse=True)[
        :10
    ]
    return result


def cpu_tensors(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def optimizer_tensors(optimizer):
    return {
        f"{key}.{name}": tensor.detach().cpu().clone()
        for key, slot in optimizer.state_dict()["state"].items()
        for name, tensor in slot.items()
        if isinstance(tensor, torch.Tensor)
    }


def reducer_state(model):
    return {
        "has_rebuilt_buckets": bool(model._has_rebuilt_buckets),
        "logging": model._get_ddp_logging_data(),
    }


def runtime_identity(engine):
    return {
        "backend": fingerprint(engine.training_backend.state_dict()),
        "rng": fingerprint(capture_rng_state(torch)),
        "data": fingerprint(engine.data_resume.capture()),
        "global_step": engine.state.global_step,
    }


def validate_step_control(cold, current):
    for key in ("step", "input_sha256", "rng_before"):
        assert cold[key] == current[key], f"cold/warm {key} differs"


def validate_warm_inputs(output, rank, report):
    cold = json.loads((output / f"cold-rank{rank}.json").read_text())
    for key in (
        "initial",
        "checkpoint_sha256",
        "attention_classes",
        "sdpa",
        "deterministic_algorithms",
    ):
        assert cold[key] == report[key], f"cold/warm {key} differs"
    required = [output / f"cold-local-rank{rank}.pt"]
    if rank == 0:
        required.extend(output / f"cold-step{step}.pt" for step in (26, 27))
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    return cold


def probe_fit(original, runner, dataloader, *, args, **kwargs):
    engine = runner.engine
    backend = engine.training_backend
    model, optimizer = backend.model, backend.optimizer
    assert isinstance(model, torch.nn.parallel.DistributedDataParallel)
    assert dist.get_world_size() == 4 and kwargs["max_steps"] == 2
    assert engine.state.global_step == 25
    rank = dist.get_rank()
    initial = coordinated_call(
        lambda: runtime_identity(engine), phase="probe initial state"
    )
    report = {
        "mode": args.mode,
        "rank": rank,
        "initial": initial,
        "checkpoint_sha256": file_hash(
            args.checkpoints / "step-00000025/backend_state.pt"
        ),
        "reducer_initial": reducer_state(model),
        "attention_classes": sorted(
            {
                f"{type(module).__module__}.{type(module).__name__}"
                for module in model.modules()
                if "Attention" in type(module).__name__
            }
        ),
        "sdpa": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        },
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "steps": [],
    }

    def preflight():
        if (args.output / f"{args.mode}-rank{rank}.json").exists():
            raise FileExistsError("diagnostic output already exists")
        assert not model._has_rebuilt_buckets
        if args.mode == "warm":
            return validate_warm_inputs(args.output, rank, report)
        if any(args.output.glob("cold-*.pt")):
            raise FileExistsError(
                "incomplete cold snapshots exist; use a fresh directory"
            )

    cold = coordinated_call(preflight, phase="probe preflight")
    if args.mode == "warm":
        iterator = engine.data_resume.prepare_iterator(dataloader)
        try:
            batch = coordinated_call(lambda: next(iterator), phase="probe warmup input")
            # Suppress only the optimizer update; use the product forward/backward.
            with patch.object(engine, "step", return_value=False):
                engine._run_step(
                    batch, model=model, optimizer=optimizer, loss_fn=kwargs["loss_fn"]
                )
        finally:
            engine.data_resume.close_iterator()
            iterator.close()
        from parascale.checkpoint import CheckpointManager

        engine.load_checkpoint(
            CheckpointManager(str(args.checkpoints)), 25, dataloader=dataloader
        )
        optimizer.zero_grad(set_to_none=True)

        def validate_restoration():
            assert backend is engine.training_backend and model is backend.model
            assert runtime_identity(engine) == initial, (
                "warmup failed to restore complete state"
            )
            assert all(parameter.grad is None for parameter in model.parameters())

        coordinated_call(validate_restoration, phase="probe warmup restoration")
        report["warmup_restored"] = True

    step = engine.step
    run_step = engine._run_step
    current = {}
    local_gradients = {}

    def measured_step(batch, **step_kwargs):
        nonlocal current
        current = {
            "step": engine.state.global_step + 1,
            "input_sha256": fingerprint(batch),
            "rng_before": fingerprint(capture_rng_state(torch)),
        }
        if cold is not None:
            coordinated_call(
                lambda: validate_step_control(
                    cold["steps"][len(report["steps"])], current
                ),
                phase="probe input and RNG identity",
            )
        metrics = run_step(batch, **step_kwargs)
        current["loss"] = metrics["loss"]
        report["steps"].append(current)
        return metrics

    def measured_update(active_optimizer=None):
        def capture_gradients():
            gradients = cpu_tensors(
                {
                    name: parameter.grad
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and parameter.grad is not None
                }
            )
            assert len(gradients) == sum(p.requires_grad for p in model.parameters())
            expected_rebuilt = args.mode == "warm" or current["step"] == 27
            assert bool(model._has_rebuilt_buckets) == expected_rebuilt
            current["gradient_sha256"] = fingerprint(gradients)
            current["reducer"] = reducer_state(model)
            if current["step"] == 26:
                local = cpu_tensors(local_gradients)
                assert local.keys() == gradients.keys()
                current["local_gradient_sha256"] = fingerprint(local)
                path = args.output / f"cold-local-rank{rank}.pt"
                if args.mode == "cold":
                    torch.save(local, path)
                else:
                    reference = torch.load(path, map_location="cpu", weights_only=True)
                    current["local_cold_vs_warm"] = tensor_differences(reference, local)
                local_gradients.clear()
            return gradients

        gradients = coordinated_call(
            capture_gradients,
            phase="probe gradient sampling",
            identity=lambda _: current["gradient_sha256"],
        )
        updated = step(active_optimizer)
        assert updated is not False

        def retain_or_compare():
            if rank != 0:
                return
            snapshot = {
                "gradients": gradients,
                "model": cpu_tensors(model.state_dict()),
                "optimizer": optimizer_tensors(optimizer),
            }
            path = args.output / f"cold-step{current['step']}.pt"
            if args.mode == "cold":
                torch.save(snapshot, path)
            else:
                reference = torch.load(path, map_location="cpu", weights_only=True)
                current["cold_vs_warm"] = {
                    key: tensor_differences(reference[key], value)
                    for key, value in snapshot.items()
                }

        coordinated_call(retain_or_compare, phase="probe update snapshots")
        return updated

    def gradient_hook(name):
        def capture(gradient):
            if current.get("step") == 26:
                # Clone on device before AccumulateGrad/DDP reduction; no CPU sync.
                local_gradients[name] = gradient.detach().clone()
            return gradient

        return capture

    handles = [
        parameter.register_hook(gradient_hook(name))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    try:
        with (
            patch.object(engine, "step", measured_update),
            patch.object(engine, "_run_step", measured_step),
        ):
            result = original(runner, dataloader, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    assert result.global_step == 27 and len(report["steps"]) == 2
    report["final_data"] = engine.data_resume.capture()
    report["final_rng"] = fingerprint(capture_rng_state(torch))
    coordinated_call(
        lambda: (args.output / f"{args.mode}-rank{rank}.json").write_text(
            json.dumps(report, indent=2)
        ),
        phase="probe report publication",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("cold", "warm"), required=True)
    args = parser.parse_args()
    config = load_config_file(args.config)
    assert config["parascale"]["training_backend"] == "native_ddp"
    assert config["parascale"]["precision"] == "fp32"
    assert config["training"]["batch_size"] == 32
    assert config["training"].get("gradient_accumulation_steps", 1) == 1
    assert config["parascale"].get("grad_clip_norm") is None
    args.output.mkdir(parents=True, exist_ok=True)
    config["data"]["cpu_prefetch"] = True
    config["runtime"]["run_dir"] = str(args.output / args.mode)
    config["training"].update(
        max_steps=2,
        checkpoint_interval=0,
        skip_final_checkpoint=True,
        checkpoint_dir=str(args.checkpoints),
    )
    original = FitLoopRunner.run

    def instrument(runner, dataloader, **kwargs):
        return probe_fit(original, runner, dataloader, args=args, **kwargs)

    try:
        with patch.object(FitLoopRunner, "run", instrument):
            run_train_from_config(config, resume_step=25)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
