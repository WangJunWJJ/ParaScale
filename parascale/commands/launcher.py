# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午7:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Benchmark launcher command builders and retry helpers."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from parascale.checkpoint import CheckpointManager
from parascale.commands.common import load_config_file
from parascale.commands.scenario import (
    apply_pipeline_cache_args,
    build_matrix_config,
    section,
)
from parascale.configuration import write_config_artifacts


def benchmark_matrix_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env["NCCL_SOCKET_IFNAME"] = env.get("PARASCALE_NCCL_SOCKET_IFNAME", "lo")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
    env.setdefault("PARASCALE_MODEL_DIRS", "/yolo_models:/models")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    return env


def benchmark_matrix_command(
    *,
    backend: str,
    config_path: Path,
    result_path: Path,
    nproc_per_node: int,
    master_port: int,
) -> list[str]:
    if backend in {"native_ddp", "fsdp"}:
        torchrun = shutil.which("torchrun") or "torchrun"
        return [
            torchrun,
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={int(nproc_per_node)}",
            f"--master_port={int(master_port)}",
            "-m",
            "parascale.cli",
            "benchmark",
            "--config",
            str(config_path),
            "--output",
            str(result_path),
        ]
    if backend in {"deepspeed", "deepspeed_zero2", "deepspeed_zero3"}:
        deepspeed = shutil.which("deepspeed") or "deepspeed"
        return [
            deepspeed,
            f"--num_gpus={int(nproc_per_node)}",
            "--module",
            "parascale.cli",
            "benchmark",
            "--config",
            str(config_path),
            "--output",
            str(result_path),
        ]
    raise ValueError(f"unsupported matrix backend: {backend}")


def train_matrix_command(
    *,
    backend: str,
    config_path: Path,
    result_path: Path,
    nproc_per_node: int,
    master_port: int,
    resume_step: int | None = None,
) -> list[str]:
    if backend in {"native_ddp", "fsdp"}:
        torchrun = shutil.which("torchrun") or "torchrun"
        command = [
            torchrun,
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={int(nproc_per_node)}",
            f"--master_port={int(master_port)}",
            "-m",
            "parascale.cli",
            "train",
            "--config",
            str(config_path),
            "--output",
            str(result_path),
        ]
    elif backend in {"deepspeed", "deepspeed_zero2", "deepspeed_zero3"}:
        deepspeed = shutil.which("deepspeed") or "deepspeed"
        command = [
            deepspeed,
            f"--num_gpus={int(nproc_per_node)}",
            "--module",
            "parascale.cli",
            "train",
            "--config",
            str(config_path),
            "--output",
            str(result_path),
        ]
    else:
        raise ValueError(f"unsupported stability backend: {backend}")
    if resume_step is not None:
        command.extend(["--resume-step", str(int(resume_step))])
    return command


def run_matrix_command(
    command: list[str],
    *,
    env: Dict[str, str],
    backend: str,
    run_id: str,
    error_path: Path,
    log_path: Path,
) -> Dict[str, Any]:
    executable = (
        shutil.which(command[0]) if not Path(command[0]).is_file() else command[0]
    )
    if executable is None:
        payload = {
            "backend": backend,
            "status": "error",
            "returncode": 127,
            "command": command,
            "log": str(log_path),
            "error": f"launcher not available: {command[0]}",
        }
        error_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {"run_id": run_id, **payload}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        completed = subprocess.run(
            command,
            env=env,
            check=False,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode == 0:
        if error_path.exists():
            error_path.unlink()
        return {
            "run_id": run_id,
            "backend": backend,
            "status": "ok",
            "returncode": 0,
            "command": command,
            "log": str(log_path),
        }
    log_tail = read_log_tail(log_path)
    oom_detected = text_indicates_oom(log_tail.lower())
    failure_details = classify_launcher_failure(
        log_tail,
        returncode=int(completed.returncode),
    )
    payload = {
        "backend": backend,
        "status": "error",
        "returncode": int(completed.returncode),
        "command": command,
        "log": str(log_path),
        "oom_detected": oom_detected,
        "log_tail": log_tail,
        "error": "benchmark failed with OOM" if oom_detected else "benchmark failed",
        **failure_details,
    }
    error_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"run_id": run_id, **payload}


def run_matrix_command_until_checkpoint(
    command: list[str],
    *,
    env: Dict[str, str],
    backend: str,
    run_id: str,
    error_path: Path,
    log_path: Path,
    checkpoint_root: Path,
    checkpoint_step: int,
    timeout_seconds: float = 3600.0,
    poll_interval_seconds: float = 0.2,
) -> Dict[str, Any]:
    """SIGKILL a launcher only after the target checkpoint validates."""
    executable = (
        shutil.which(command[0]) if not Path(command[0]).is_file() else command[0]
    )
    if executable is None:
        payload = {
            "backend": backend,
            "status": "error",
            "returncode": 127,
            "command": command,
            "log": str(log_path),
            "failure_type": "launcher_missing",
            "error": f"launcher not available: {command[0]}",
        }
        error_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {"run_id": run_id, **payload}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
        while True:
            if _checkpoint_is_complete(checkpoint_root, checkpoint_step):
                _kill_process_group(process)
                process.wait(timeout=30)
                checkpoint_ok = _checkpoint_is_valid(
                    checkpoint_root,
                    checkpoint_step,
                )
                if not checkpoint_ok:
                    break
                if error_path.exists():
                    error_path.unlink()
                return {
                    "run_id": run_id,
                    "phase": "train",
                    "backend": backend,
                    "status": "interrupted",
                    "returncode": int(process.returncode or 0),
                    "command": command,
                    "log": str(log_path),
                    "intentional_kill": True,
                    "checkpoint_ok": True,
                    "checkpoint_step": int(checkpoint_step),
                }
            returncode = process.poll()
            if returncode is not None:
                break
            if time.monotonic() >= deadline:
                _kill_process_group(process)
                process.wait(timeout=30)
                returncode = process.returncode
                break
            time.sleep(max(0.01, float(poll_interval_seconds)))

    log_tail = read_log_tail(log_path)
    failure_details = classify_launcher_failure(
        log_tail,
        returncode=int(returncode or 1),
    )
    if time.monotonic() >= deadline:
        failure_details["failure_type"] = "checkpoint_wait_timeout"
    payload = {
        "backend": backend,
        "status": "error",
        "returncode": int(returncode or 1),
        "command": command,
        "log": str(log_path),
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_ok": False,
        "log_tail": log_tail,
        "error": "launcher exited before a valid checkpoint was available",
        **failure_details,
    }
    error_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"run_id": run_id, **payload}


def _checkpoint_is_valid(checkpoint_root: Path, checkpoint_step: int) -> bool:
    try:
        manager = CheckpointManager(str(checkpoint_root))
        return bool(manager.validate(int(checkpoint_step)).ok)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return False


def _checkpoint_is_complete(checkpoint_root: Path, checkpoint_step: int) -> bool:
    """Check atomic manifest and payload sizes without hashing the hot checkpoint."""
    try:
        manager = CheckpointManager(str(checkpoint_root))
        manifest = manager.read_manifest(int(checkpoint_step))
        for entry in manifest.files:
            if entry.get("error") or "path" not in entry:
                return False
            path = manager.resolve_payload_path(manifest, entry)
            if not path.exists():
                return False
            expected_size = entry.get("size_bytes")
            if path.is_file() and expected_size is not None:
                if path.stat().st_size != int(expected_size):
                    return False
        return True
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return False


def _kill_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    for pid in reversed(_process_descendants(process.pid)):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _process_descendants(root_pid: int) -> list[int]:
    """Return descendants using Linux procfs, including detached process groups."""
    children: Dict[int, list[int]] = {}
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(
                encoding="utf-8",
                errors="replace",
            )
            parent_line = next(
                line for line in status.splitlines() if line.startswith("PPid:")
            )
            parent_pid = int(parent_line.split(":", 1)[1].strip())
        except (OSError, StopIteration, ValueError):
            continue
        children.setdefault(parent_pid, []).append(int(entry.name))

    descendants: list[int] = []
    pending = list(children.get(int(root_pid), []))
    while pending:
        pid = pending.pop()
        descendants.append(pid)
        pending.extend(children.get(pid, []))
    return descendants


def read_log_tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-int(max_chars) :]


def matrix_result_is_oom(result: Dict[str, Any]) -> bool:
    if bool(result.get("oom_detected")):
        return True
    text = " ".join(
        str(result.get(key, "")) for key in ("error", "log_tail", "stderr_tail")
    ).lower()
    return text_indicates_oom(text)


def text_indicates_oom(text: str) -> bool:
    patterns = [
        "out of memory",
        "cuda oom",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
        "ncclunhandledcudaerror",
        "hip out of memory",
        "deepspeed oom",
    ]
    return any(pattern in text for pattern in patterns)


def classify_launcher_failure(text: str, *, returncode: int) -> Dict[str, Any]:
    """Extract a stable failure category and distributed evidence from logs."""
    lowered = text.lower()
    if text_indicates_oom(lowered):
        failure_type = "oom"
    elif (
        "collective operation timeout" in lowered
        or "processgroupnccl" in lowered and "timeout" in lowered
        or "hccl" in lowered and "timeout" in lowered
    ):
        failure_type = "distributed_timeout"
    elif "dataloader worker" in lowered and (
        "exited" in lowered or "killed" in lowered
    ):
        failure_type = "dataloader_worker_failure"
    elif "checkpoint" in lowered and (
        "checksum" in lowered or "manifest" in lowered or "failed" in lowered
    ):
        failure_type = "checkpoint_failure"
    else:
        failure_type = "process_failure"

    details: Dict[str, Any] = {"failure_type": failure_type}
    rank_match = re.search(r"\[rank(\d+)\]|\brank\s*:\s*(\d+)", text, re.IGNORECASE)
    if rank_match:
        details["failed_rank"] = int(rank_match.group(1) or rank_match.group(2))
    collective_match = re.search(r"OpType=([A-Z_]+)", text)
    if collective_match:
        details["collective"] = collective_match.group(1)
    sequence_match = re.search(r"SeqNum=(\d+)", text)
    if sequence_match:
        details["collective_sequence"] = int(sequence_match.group(1))
    signal_match = re.search(r"\((SIG[A-Z0-9]+)\)", text)
    if signal_match:
        details["signal"] = signal_match.group(1)
    if int(returncode) < 0 and "signal" not in details:
        details["signal_number"] = abs(int(returncode))
    return details


def oom_retry_policy_payload() -> Dict[str, Any]:
    return {
        "enabled_when": (
            "launcher failed and log matches CUDA/NCCL/DeepSpeed OOM patterns"
        ),
        "actions": [
            "halve_batch_size",
            "enable_activation_checkpointing",
            "retry_same_backend",
            "fallback_to_fsdp",
            "fallback_to_deepspeed_zero2",
            "fallback_to_deepspeed_zero3",
        ],
    }


def run_oom_retry_sequence(
    *,
    scenario: str,
    scenario_config: Dict[str, Any],
    base_run_spec: Dict[str, Any],
    failed_backend: str,
    failed_batch_size: int | None,
    output_dir: Path,
    env: Dict[str, str],
    args: argparse.Namespace,
    commands: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    retry_results: list[Dict[str, Any]] = []
    retry_batch_size = max(1, int((failed_batch_size or args.batch_size or 1) // 2))
    retry_backends = oom_retry_backends(failed_backend)
    for retry_index, retry_backend in enumerate(retry_backends, start=1):
        retry_run_spec = dict(base_run_spec)
        retry_run_spec["run_id"] = (
            f"{base_run_spec['run_id']}_oom_retry{retry_index}_{retry_backend}"
        )
        config_path = output_dir / f"{retry_run_spec['run_id']}.config.json"
        result_path = output_dir / f"{retry_run_spec['run_id']}.json"
        error_path = output_dir / f"{retry_run_spec['run_id']}.error.json"
        log_path = output_dir / f"{retry_run_spec['run_id']}.log"
        config_data = build_matrix_config(
            scenario=scenario,
            base_config=load_config_file(scenario_config["base_config"]),
            run_spec=retry_run_spec,
            backend=retry_backend,
            output_dir=output_dir,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            batch_size=retry_batch_size,
            num_samples=args.num_samples,
        )
        apply_oom_retry_config(config_data, retry_backend)
        apply_pipeline_cache_args(config_data, args)
        emergency_overrides = oom_retry_overrides(
            retry_backend,
            batch_size=retry_batch_size,
        )
        config_data["_resolution"] = {
            "emergency_overrides": emergency_overrides,
        }
        artifact_dir = output_dir / retry_run_spec["run_id"]
        config_data.setdefault("runtime", {})["run_dir"] = str(artifact_dir)
        artifacts = write_config_artifacts(
            config_data,
            artifact_dir,
            emergency_overrides=emergency_overrides,
        )
        config_path.write_text(
            json.dumps(config_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        command = benchmark_matrix_command(
            backend=retry_backend,
            config_path=config_path,
            result_path=result_path,
            nproc_per_node=args.nproc_per_node,
            master_port=int(args.master_port) + len(commands),
        )
        commands.append(
            {
                "run_id": retry_run_spec["run_id"],
                "backend": retry_backend,
                "batch_size": retry_batch_size,
                "config": str(config_path),
                "output": str(result_path),
                "log": str(log_path),
                "command": command,
                "retry_of": {
                    "run_id": base_run_spec["run_id"],
                    "backend": failed_backend,
                    "batch_size": failed_batch_size,
                },
                "attempt": retry_index,
                "retry_trigger": "oom",
                "config_artifacts": artifacts,
            }
        )
        result = run_matrix_command(
            command,
            env=env,
            backend=retry_backend,
            run_id=retry_run_spec["run_id"],
            error_path=error_path,
            log_path=log_path,
        )
        result["retry_of"] = {
            "run_id": base_run_spec["run_id"],
            "backend": failed_backend,
            "batch_size": failed_batch_size,
        }
        result["attempt"] = retry_index
        result["retry_trigger"] = "oom"
        result["config_artifacts"] = artifacts
        retry_results.append(result)
        if result.get("status") == "ok":
            _persist_retry_metadata(result_path, result)
            break
        if not matrix_result_is_oom(result):
            result["retry_terminated"] = True
            result["retry_termination_reason"] = "non_oom_failure"
            _persist_retry_metadata(error_path, result)
            break
        _persist_retry_metadata(error_path, result)
    return retry_results


def _persist_retry_metadata(path: Path, result: Dict[str, Any]) -> None:
    payload: Dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (json.JSONDecodeError, OSError):
            payload = {}
    for key in (
        "attempt",
        "retry_trigger",
        "retry_terminated",
        "retry_termination_reason",
        "retry_of",
        "config_artifacts",
    ):
        if key in result:
            payload[key] = result[key]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def oom_retry_backends(failed_backend: str) -> list[str]:
    order = [
        failed_backend,
        "fsdp",
        "deepspeed_zero2",
        "deepspeed_zero3",
    ]
    unique: list[str] = []
    for backend in order:
        if backend not in unique:
            unique.append(backend)
    return unique


def apply_oom_retry_config(config_data: Dict[str, Any], backend: str) -> None:
    parascale = section(config_data, "parascale")
    model = section(config_data, "model")
    training = section(config_data, "training")
    parascale["enable_activation_checkpointing"] = True
    model["activation_checkpointing"] = True
    training["oom_retry"] = True
    if backend == "fsdp":
        parascale["training_backend"] = "fsdp"
        parascale["fsdp_state_dict_type"] = "sharded"
    elif backend in {"deepspeed", "deepspeed_zero2", "deepspeed_zero3"}:
        parascale["training_backend"] = "deepspeed"
        parascale["zero_optimization"] = True
        parascale["zero_stage"] = 3 if backend.endswith("zero3") else 2


def oom_retry_overrides(backend: str, *, batch_size: int) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {"training.batch_size": int(batch_size)}
    if backend == "fsdp":
        overrides["backend.training_backend"] = "fsdp"
    elif backend in {"deepspeed", "deepspeed_zero2", "deepspeed_zero3"}:
        overrides["backend.training_backend"] = "deepspeed"
        overrides["backend.zero_stage"] = 3 if backend.endswith("zero3") else 2
    else:
        overrides["backend.training_backend"] = backend
    return overrides
