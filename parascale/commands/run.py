# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午7:02
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Train, serve, and benchmark command implementation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from parascale.commands.common import emit_json, load_config_file
from parascale.commands.plan import build_plan_payload, section
from parascale.configuration import resolve_config, write_config_artifacts
from parascale.runtime import build_benchmark_plan, build_runtime_context
from parascale.runtime.backends.devices import set_current_device
from parascale.runtime.factory import (
    build_optimizer_for_model,
    build_serving_model_from_checkpoint,
    build_training_components,
)
from parascale.runtime.inference import InferenceRunner
from parascale.runtime.orchestrator import (
    _destroy_distributed_for_cli,
)
from parascale.runtime.orchestrator import (
    run_benchmark_from_config as _run_benchmark_from_config,
)
from parascale.runtime.orchestrator import (
    run_serve_from_config as _run_serve_from_config,
)
from parascale.runtime.orchestrator import (
    run_train_from_config as _run_train_from_config,
)
from parascale.workloads.inference import build_inference_components


def sync_orchestrator_factories() -> None:
    import parascale.runtime.orchestrator as orchestrator

    orchestrator.build_training_components = build_training_components
    orchestrator.build_optimizer_for_model = build_optimizer_for_model
    orchestrator.build_serving_model_from_checkpoint = (
        build_serving_model_from_checkpoint
    )


def run_train_from_config(
    config_data: Dict[str, Any],
    resume_step: int | None = None,
) -> Dict[str, Any]:
    sync_orchestrator_factories()
    return _run_train_from_config(config_data, resume_step=resume_step)


def run_serve_from_config(
    config_data: Dict[str, Any],
    checkpoint: str | None = None,
) -> Dict[str, Any]:
    sync_orchestrator_factories()
    return _run_serve_from_config(config_data, checkpoint=checkpoint)


def run_benchmark_from_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    sync_orchestrator_factories()
    return _run_benchmark_from_config(config_data)


def run_inference_from_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    model, batches, task = build_inference_components(config_data)
    runtime = section(config_data, "runtime")
    inference = section(config_data, "inference")
    device, memory_getter = _resolve_inference_device(runtime)
    payload = InferenceRunner(
        model=model,
        task=task,
        device=device,
        memory_getter=memory_getter,
    ).run(
        batches,
        warmup_steps=int(inference.get("warmup_steps", 1) or 0),
    )
    return {
        "mode": "infer",
        "dry_run": False,
        "runtime_status": "real_local" if device != "cpu" else "local_cpu",
        "capability_level": _inference_capability(task, device),
        "task": task,
        "workload": str(inference.get("workload", "clip_synthetic")),
        **payload,
    }


def _resolve_inference_device(runtime: Dict[str, Any]) -> tuple[str, Any]:
    accelerator = str(runtime.get("accelerator", "auto") or "auto").lower()
    if accelerator == "cpu":
        return "cpu", lambda: None
    try:
        import torch
    except Exception:
        return "cpu", lambda: None
    if accelerator == "npu":
        import torch_npu  # noqa: F401
    requested = runtime.get("device")
    device = set_current_device(
        torch,
        local_rank=int(runtime.get("local_rank", 0) or 0),
        requested_device=str(requested) if requested else None,
    )
    device_text = str(device)
    memory = (
        (lambda: torch.cuda)
        if device_text.startswith("cuda")
        else (lambda: torch.npu) if device_text.startswith("npu") else (lambda: None)
    )
    return device_text, memory


def _inference_capability(task: str, device: str) -> str:
    if device.startswith("npu"):
        return f"ascend_{task}_smoke"
    if device.startswith("cuda"):
        return f"cuda_{task}_smoke"
    return f"cpu_{task}_smoke"


def build_train_dry_run_payload(config_data: Dict[str, Any]) -> Dict[str, Any]:
    payload = build_plan_payload(config_data)
    resolved_config = resolve_config(config_data, dry_run=True)
    runtime = section(config_data, "runtime")
    training = section(config_data, "training")
    payload.update(
        {
            "mode": "train",
            "dry_run": True,
            "runtime_status": "plan_only",
            "capability_level": "dry_run",
            "synthetic": False,
            "mock": False,
            "backend": training.get("backend")
            or section(config_data, "parascale").get("training_backend", "auto"),
            "entrypoint_status": "dry_run_only",
            "next_step": (
                "Provide model, optimizer, dataloader and step_fn to TrainEngine "
                "for real execution."
            ),
            "resolved_config": resolved_config.to_dict(),
        }
    )
    if runtime:
        payload["runtime"] = runtime
    return payload


def build_serve_dry_run_payload(
    config_data: Dict[str, Any] | None = None,
    checkpoint: str | None = None,
) -> Dict[str, Any]:
    config_data = config_data or {}
    serving = section(config_data, "serving")
    payload: Dict[str, Any] = {
        "mode": "serve",
        "dry_run": True,
        "runtime_status": "plan_only",
        "capability_level": "dry_run",
        "mock": bool(serving.get("mock", False)),
        "checkpoint": checkpoint or serving.get("checkpoint"),
        "entrypoint_status": "dry_run_only",
        "next_step": (
            "Load a concrete model or ParaScale checkpoint manifest before "
            "starting real serving."
        ),
    }
    if config_data:
        payload["config_sections"] = sorted(config_data.keys())
    if serving:
        payload["serving"] = serving
    return payload


def build_benchmark_dry_run_payload(config_data: Dict[str, Any]) -> Dict[str, Any]:
    payload = build_plan_payload(config_data)
    resolved_config = resolve_config(config_data, dry_run=True)
    context = build_runtime_context(config_data, mode="benchmark")
    payload.update(
        {
            "mode": "benchmark",
            "dry_run": True,
            "runtime_status": "plan_only",
            "capability_level": "dry_run",
            "synthetic": False,
            "entrypoint_status": "dry_run_only",
            "metrics": [
                "step_time_ms",
                "tokens_per_second",
                "images_per_second",
                "peak_memory_bytes",
            ],
            "benchmark_plan": build_benchmark_plan(context).to_dict(),
            "resolved_config": resolved_config.to_dict(),
        }
    )
    return payload


def cmd_train(args: argparse.Namespace) -> int:
    config_data = load_config_file(args.config)
    config_artifacts = _write_command_config_artifacts(
        config_data,
        output_path=args.output,
        mode="train",
        dry_run=bool(args.dry_run),
    )
    if not args.dry_run:
        try:
            payload = run_train_from_config(config_data, args.resume_step)
            payload.setdefault("config_artifacts", config_artifacts)
            emit_json(payload, args.output)
        finally:
            _destroy_distributed_for_cli()
        return 0
    payload = build_train_dry_run_payload(config_data)
    payload["config_artifacts"] = config_artifacts
    emit_json(payload, args.output)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    config_data = load_config_file(args.config) if args.config else {}
    if not args.dry_run:
        payload = run_serve_from_config(config_data, args.checkpoint)
        emit_json(payload, args.output)
        return 0
    emit_json(
        build_serve_dry_run_payload(config_data, checkpoint=args.checkpoint),
        args.output,
    )
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    config_data = load_config_file(args.config)
    payload = run_inference_from_config(config_data)
    emit_json(payload, args.output)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    config_data = load_config_file(args.config)
    config_artifacts = _write_command_config_artifacts(
        config_data,
        output_path=args.output,
        mode="benchmark",
        dry_run=bool(args.dry_run),
    )
    if not args.dry_run:
        try:
            payload = run_benchmark_from_config(config_data)
        finally:
            _destroy_distributed_for_cli()
    else:
        payload = build_benchmark_dry_run_payload(config_data)
    payload.setdefault("config_artifacts", config_artifacts)
    emit_json(payload, args.output)
    return 0


def _write_command_config_artifacts(
    config_data: Dict[str, Any],
    *,
    output_path: str | None,
    mode: str,
    dry_run: bool,
) -> Dict[str, str | None]:
    runtime = config_data.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        config_data["runtime"] = runtime
    configured = runtime.get("run_dir")
    if configured:
        run_dir = Path(str(configured))
    elif output_path:
        output = Path(output_path)
        run_dir = output.parent / output.stem
        runtime["run_dir"] = str(run_dir)
    else:
        run_dir = Path("runs") / mode
        runtime["run_dir"] = str(run_dir)
    resolution = config_data.get("_resolution", {})
    emergency = (
        dict(resolution.get("emergency_overrides", {}))
        if isinstance(resolution, dict)
        else {}
    )
    return write_config_artifacts(
        config_data,
        run_dir,
        emergency_overrides=emergency,
        dry_run=dry_run,
    )
