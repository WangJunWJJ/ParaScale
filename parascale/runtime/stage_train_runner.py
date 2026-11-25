# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Single-node native DP/TP/PP training through the unified runtime entrypoint."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from parascale.optimizers import build_optimizer
from parascale.parallel.checkpoint_catalog import (
    _safe_path,
    inspect_stage_checkpoint,
    latest_stage_checkpoint,
    prune_stage_checkpoints,
)
from parascale.parallel.execution import DistributedTrainer
from parascale.workloads.causal_lm import build_causal_lm_stage_components

from .runner_common import _write_runtime_config_artifacts
from .stage_identity import build_stage_checkpoint_context
from .stage_plan import build_stage_training_plan
from .training.coordination import coordinated_call
from .training.data_state import DataResumeController
from .training.engine import TrainState
from .training.stage_checkpoint import StageTrainingCheckpoint
from .training.stage_loop import StageUpdateLoop


def _validate_stage_launch(plan: Any) -> tuple[int, int]:
    required = (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    )
    if any(not os.environ.get(key) for key in required):
        raise ValueError("native_parallel requires complete torchrun launcher metadata")
    rank, local_rank, world, local_world = (
        int(os.environ[name]) for name in required[:4]
    )
    if (
        world != plan.topology.world_size
        or local_world != world
        or rank != local_rank
        or not 0 <= rank < world
        or not 0 < int(os.environ["MASTER_PORT"]) < 65536
    ):
        raise ValueError(
            "native_parallel requires matching single-node ranks and topology"
        )
    return rank, local_rank


def _resume_request(config: dict, step: Any, latest: bool) -> int | None:
    step = config.get("training", {}).get("resume_step") if step is None else step
    if step is not None and (type(step) is not int or step < 0):
        raise ValueError("resume_step must be a nonnegative integer")
    if type(latest) is not bool or (latest and step is not None):
        raise ValueError("resume_latest and resume_step are mutually exclusive")
    return step


def _rank_zero(operation: Any, *, phase: str) -> Any:
    value = coordinated_call(
        lambda: operation() if dist.get_rank() == 0 else None, phase=phase
    )
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _prepare_checkpoint_root(root: Path, *, resuming: bool) -> None:
    if ".." in root.parts:
        raise ValueError("Checkpoint parent traversal is not allowed")
    existing = root
    while not os.path.lexists(existing):
        existing = existing.parent
    _safe_path(existing)
    root.mkdir(parents=True, exist_ok=True)
    _safe_path(root)
    if not resuming and any(root.glob("step-*")):
        raise ValueError(
            "Checkpoint root contains prior steps; use resume or a new directory"
        )


def _validate_save_targets(plan: Any, *, initial_step: int) -> None:
    for path in plan.checkpoint_dir.glob("step-*"):
        try:
            step = int(path.name[5:])
        except ValueError:
            continue
        # Future manifests can also invalidate retention's newest-anchor rule,
        # even when their payload is corrupt and outside this update window.
        if path.name == f"step-{step:08d}" and step > initial_step:
            raise ValueError(
                f"Checkpoint output already exists: {path}. Nothing was trained; "
                "preserve or explicitly relocate the conflicting snapshot before resuming."
            )


def _initialize(plan: Any) -> torch.device:
    rank, local_rank = _validate_stage_launch(plan)
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError("native_parallel requires one visible CUDA device per rank")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if dist.is_initialized():
        if (
            dist.get_backend() != "nccl"
            or dist.get_world_size() != plan.topology.world_size
            or dist.get_rank() != rank
        ):
            raise ValueError(
                "Initialized process group conflicts with native_parallel launch"
            )
    else:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=plan.timeout_seconds),
            device_id=device,
        )
    return device


def run_stage_train_from_config(
    config_data: dict,
    resume_step: int | None = None,
    *,
    resume_latest: bool = False,
) -> dict[str, Any]:
    """Run additional optimizer updates; caller owns process-group destruction.

    Use a dedicated trusted checkpoint root with one writer. The CLI keeps the
    group alive through coordinated result output and destroys it in finally.
    """
    plan = build_stage_training_plan(config_data)
    resume_step = _resume_request(plan.config_data, resume_step, resume_latest)
    device = _initialize(plan)
    coordinated_call(
        lambda: plan.to_dict(), phase="training plan", identity=lambda value: value
    )
    coordinated_call(
        lambda: (resume_step, resume_latest),
        phase="resume request",
        identity=lambda value: value,
    )
    root = plan.checkpoint_dir
    _rank_zero(
        lambda: _prepare_checkpoint_root(
            root, resuming=resume_latest or resume_step is not None
        ),
        phase="checkpoint directory",
    )
    artifacts = _rank_zero(
        lambda: _write_runtime_config_artifacts(
            plan.config_data, plan.config, strategy_selected=False
        ),
        phase="configuration artifacts",
    )

    def select():
        if resume_latest:
            selected, skipped = latest_stage_checkpoint(root)
            result = {
                "path": str(selected),
                "skipped": skipped,
                "step": int(selected.name[5:]),
            }
        elif resume_step is not None:
            selected = root / f"step-{resume_step:08d}"
            metadata = inspect_stage_checkpoint(selected)
            result = {"path": str(selected), "skipped": [], "step": metadata["step"]}
        else:
            result = {"path": None, "skipped": [], "step": 0}
        _validate_save_targets(plan, initial_step=result["step"])
        return result

    selection = _rank_zero(select, phase="checkpoint selection")
    stage, cursor, loss = coordinated_call(
        lambda: build_causal_lm_stage_components(
            plan.config_data,
            plan.topology,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        ),
        phase="stage construction",
    )
    context = coordinated_call(
        lambda: build_stage_checkpoint_context(stage, cursor, plan),
        phase="training identity",
        identity=lambda value: value,
    )

    def prepare_stage():
        stage.to(device)
        shape = (plan.workload.batch_size, plan.workload.sequence_length)
        example = (
            torch.zeros(shape, dtype=torch.int64, device=device)
            if stage.stage_index == 0
            else (
                torch.zeros(
                    (*shape, stage.hidden_size), dtype=torch.float32, device=device
                )
            )
        )
        return example, stage.tensor_parallel_plan() or None

    example, tensor_plan = coordinated_call(prepare_stage, phase="stage placement")
    trainer = DistributedTrainer(
        stage,
        plan.topology,
        example,
        summed_loss=loss,
        tensor_plan=tensor_plan,
        checkpoint_context=context,
        precision=plan.config.precision,
    )
    del example

    def prepare_optimizer():
        optimizer = build_optimizer(trainer.module, plan.config_data)
        scheduler = (
            plan.scheduler_spec.build(optimizer) if plan.scheduler_spec else None
        )
        progress = SimpleNamespace(
            state=TrainState(initialized=True),
            scheduler=scheduler,
            scheduler_metadata=plan.scheduler_spec.to_metadata()
            if plan.scheduler_spec
            else None,
        )
        progress.data_resume = DataResumeController(progress)
        recovery = StageTrainingCheckpoint(
            progress,
            cursor,
            optimizer,
            microbatches=plan.topology.microbatches,
            scheduler_spec=plan.scheduler_spec,
        )
        return optimizer, progress, recovery

    optimizer, progress, recovery = coordinated_call(
        prepare_optimizer, phase="optimizer construction"
    )
    if selection["path"]:
        trainer.load_checkpoint(selection["path"], optimizer, runtime=recovery)
    last_checkpoint = {}

    def save(step):
        path = root / f"step-{step:08d}"
        trainer.save_checkpoint(path, optimizer, step=step, runtime=recovery)
        retention = _rank_zero(
            lambda: prune_stage_checkpoints(path, keep_last=plan.keep_last),
            phase="checkpoint retention",
        )
        last_checkpoint.update(path=str(path), step=step, retention=retention)

    torch.cuda.reset_peak_memory_stats(device)
    result = StageUpdateLoop(
        trainer, optimizer, progress, cursor, plan, device, save
    ).run()
    peak = torch.tensor(torch.cuda.max_memory_allocated(device), device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    return {
        "mode": "train",
        "dry_run": False,
        "runtime_status": "real_distributed",
        "backend": "native_parallel",
        "workload": "causal_lm",
        "mock": False,
        "synthetic": False,
        "precision": plan.config.precision,
        "plan": plan.to_dict(),
        "config_artifacts": artifacts,
        "resume_selection": selection,
        "checkpoint": last_checkpoint,
        "peak_memory_allocated_bytes": peak.item(),
        **result,
    }
