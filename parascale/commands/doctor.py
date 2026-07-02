# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午2:03
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Doctor command payload builders."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
from typing import Any, Dict

from parascale.commands.common import emit_json
from parascale.commands.diagnostics import evaluate_diagnostics
from parascale.core import AscendDeviceBackend, CpuDeviceBackend, NvidiaDeviceBackend


def build_doctor_payload() -> Dict[str, Any]:
    dependencies = {
        "torch": importlib.util.find_spec("torch") is not None,
        "torch_npu": importlib.util.find_spec("torch_npu") is not None,
        "deepspeed": importlib.util.find_spec("deepspeed") is not None,
        "yaml": importlib.util.find_spec("yaml") is not None,
    }
    device_backends = [
        CpuDeviceBackend().capability(),
        NvidiaDeviceBackend().capability(),
        AscendDeviceBackend().capability(),
    ]
    rank_env = {
        name: os.environ.get(name)
        for name in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
        if os.environ.get(name) is not None
    }
    return {
        "mode": "doctor",
        "runtime_status": "diagnostic",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "dependencies": dependencies,
        "torch_runtime": inspect_torch_runtime(),
        "distributed_runtime": inspect_distributed_runtime(),
        "ascend_runtime": inspect_ascend_runtime(),
        "device_backends": device_backends,
        "rank_env": rank_env,
        "notes": [
            (
                "Hardware-dependent training and serving paths require matching "
                "torch, distributed, CUDA/NPU, and launcher support."
            ),
            (
                "Mock and synthetic CLI paths are explicit diagnostics, "
                "not production readiness signals."
            ),
        ],
    }


def cmd_doctor(args: argparse.Namespace) -> int:
    requirements = list(args.require)
    if args.strict:
        requirements = ["core", "torch", *requirements]
    payload = build_doctor_payload()
    if "deepspeed" in requirements:
        payload["deepspeed_runtime"] = inspect_deepspeed_runtime()
    report = evaluate_diagnostics(payload, requirements)
    payload.update(report.to_dict())
    emit_json(payload, args.output)
    return 0 if report.ok else 2


def inspect_torch_runtime() -> Dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"available": False, "reason": "torch is not installed"}
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        devices = []
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_memory": int(props.total_memory),
                        "capability": list(torch.cuda.get_device_capability(index)),
                    }
                )
        return {
            "available": True,
            "version": str(torch.__version__),
            "cuda_available": cuda_available,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cuda_device_count": len(devices),
            "cuda_devices": devices,
            "bf16_supported": (
                bool(torch.cuda.is_bf16_supported()) if cuda_available else False
            ),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def inspect_deepspeed_runtime() -> Dict[str, Any]:
    """Import DeepSpeed only when the command explicitly requires it."""

    if importlib.util.find_spec("deepspeed") is None:
        return {"available": False, "reason": "deepspeed is not installed"}
    try:
        import deepspeed

        return {
            "available": True,
            "version": str(getattr(deepspeed, "__version__", "unknown")),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def inspect_distributed_runtime() -> Dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"available": False, "reason": "torch is not installed"}
    try:
        import torch.distributed as dist

        return {
            "available": bool(dist.is_available()),
            "initialized": bool(dist.is_available() and dist.is_initialized()),
            "recommended_backends": recommended_collective_backends(),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def recommended_collective_backends() -> Dict[str, bool]:
    recommended = {"gloo": True, "nccl": False, "hccl": False}
    try:
        import torch

        recommended["nccl"] = bool(torch.cuda.is_available())
    except Exception:
        recommended["nccl"] = False
    recommended["hccl"] = importlib.util.find_spec("torch_npu") is not None
    return recommended


def inspect_ascend_runtime() -> Dict[str, Any]:
    if importlib.util.find_spec("torch_npu") is None:
        return {"available": False, "reason": "torch_npu is not installed"}
    try:
        import torch
        import torch_npu  # noqa: F401

        npu = getattr(torch, "npu", None)
        device_count = (
            int(npu.device_count())
            if npu is not None and hasattr(npu, "device_count")
            else 0
        )
        return {
            "available": True,
            "device_count": device_count,
            "current_device": (
                int(npu.current_device())
                if npu is not None and hasattr(npu, "current_device")
                else None
            ),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}
