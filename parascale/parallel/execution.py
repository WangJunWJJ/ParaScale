# -*- coding: utf-8 -*-
# @Time : 2026/11/21 下午6:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Explicit CUDA stage execution using PyTorch's distributed primitives.

This does not enable TP/PP for the ordinary workload CLI. The caller supplies
one local stage, its input example, a TP plan and an initialized process group.
"""

from __future__ import annotations

import inspect
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from .distributed import DistributedParallelConfig
from .loss import SummedLoss


def _validate_example_input(example: Any, stage: int) -> None:
    import torch

    if (
        not isinstance(example, torch.Tensor)
        or example.ndim < 1
        or example.shape[0] < 1
    ):
        raise ValueError("example input must contain one nonempty microbatch")
    if example.dtype != torch.float32 and not (
        stage == 0 and example.dtype == torch.int64
    ):
        raise ValueError("stage input must be float32, or int64 on the first stage")


def _validate_constructor(module, config, example, loss_fn, summed_loss, tensor_plan):
    """Check local arguments without creating subgroups or mutating the module."""
    import torch
    import torch.distributed as dist

    if torch.__version__.split("+")[0] != "2.9.1":
        raise RuntimeError(
            "DistributedTrainer requires the tested torch==2.9.1 runtime; install parascale[parallel]"
        )
    config.validate_world_size(dist.get_world_size())
    _validate_example_input(example, config.coordinates(dist.get_rank())[1])
    if dist.get_backend() != "nccl" or example.device.type != "cuda":
        raise ValueError("DistributedTrainer currently requires CUDA/NCCL")
    if int(os.environ.get("LOCAL_WORLD_SIZE", config.world_size)) != config.world_size:
        raise ValueError("Only single-node execution has been validated")
    if (loss_fn is None) == (summed_loss is None):
        raise ValueError("Provide exactly one of loss_fn or summed_loss")
    if loss_fn is not None and not callable(loss_fn):
        raise ValueError("loss_fn must be callable")
    if summed_loss is not None and (
        not callable(summed_loss)
        or not callable(getattr(summed_loss, "normalization_count", None))
    ):
        raise ValueError("summed_loss requires a callable normalization_count")
    if any(
        p.device != example.device or p.dtype != torch.float32
        for p in module.parameters()
    ):
        raise ValueError("Stage parameters must be float32 on the input device")
    if any(buffer.device != example.device for buffer in module.buffers()):
        raise ValueError("Stage buffers must be on the input device")
    if config.tp_size > 1 and not tensor_plan:
        raise ValueError("TP requires an explicit model-specific tensor_plan")
    if config.tp_size == 1 and tensor_plan:
        raise ValueError("tensor_plan requires tp_size > 1")
    if tensor_plan:
        missing = set(tensor_plan) - set(dict(module.named_modules()))
        if missing:
            raise ValueError(f"Unknown tensor_plan modules: {sorted(missing)}")
    if config.pp_size > 1 and summed_loss is not None:
        from torch.distributed.pipelining import ScheduleGPipe

        if "scale_grads" not in inspect.signature(ScheduleGPipe).parameters:
            raise RuntimeError("Summed loss requires explicit GPipe scale_grads=False")


@dataclass
class ParallelStepResult:
    """DP-averaged loss and local output (only on the last pipeline stage)."""

    loss: float
    output: Any = None
    normalization_count: int | None = None


class DistributedTrainer:
    """Fixed-shape, FP32 stage training on one to four CUDA devices.

    loss_fn is a scalar mean over equal-sized samples/tokens. Alternatively,
    summed_loss explicitly supplies a scalar sum and valid-unit count, including
    ignored labels and locally empty targets. Each TP/PP group receives the same local
    data; different DP replicas receive different samples. Caller owns optimizer
    steps, data sampling, process-group lifetime and job-level error handling.
    """

    def __init__(
        self,
        module: Any,
        config: DistributedParallelConfig,
        example_input: Any,
        loss_fn: Any = None,
        *,
        tensor_plan: dict[str, Any] | None = None,
        summed_loss: SummedLoss | None = None,
    ) -> None:
        import torch
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh

        if not dist.is_initialized():
            raise RuntimeError("Initialize a CUDA/NCCL process group first")
        error, signature = None, None
        try:
            _validate_constructor(
                module, config, example_input, loss_fn, summed_loss, tensor_plan
            )
            signature = (
                config.dp_size,
                config.tp_size,
                config.pp_size,
                config.microbatches,
                summed_loss is not None,
            )
        except Exception as exc:
            error = str(exc)
        records = [(error, signature)]
        if dist.get_world_size() > 1:
            records = [None] * dist.get_world_size()
            dist.all_gather_object(records, (error, signature))
        errors = [
            f"rank {rank}: {item[0]}" for rank, item in enumerate(records) if item[0]
        ]
        if errors:
            raise ValueError(
                f"Parallel constructor preflight failed: {'; '.join(errors)}"
            )
        if any(item[1] != records[0][1] for item in records):
            raise ValueError(
                "Parallel constructor preflight failed: topology, microbatches and loss mode must agree"
            )
        self.config = config
        self.device = example_input.device
        self.rank = dist.get_rank()
        self.dp_rank, self.pp_rank, self.tp_rank = config.coordinates(self.rank)
        # FSDP2 joins DP and TP submeshes; singleton PP must not split them.
        dimensions = [
            (name, size)
            for name, size in (
                ("dp", config.dp_size),
                ("pp", config.pp_size),
                ("tp", config.tp_size),
            )
            if size > 1
        ] or [("dp", 1)]
        self.mesh = init_device_mesh(
            "cuda",
            tuple(size for _, size in dimensions),
            mesh_dim_names=tuple(name for name, _ in dimensions),
        )
        self.input_shape = tuple(example_input.shape)
        self.input_dtype = example_input.dtype
        self.loss_fn = loss_fn
        self.summed_loss = summed_loss
        self._normalization_count = None
        self._loss_scale = None
        self.module = module
        # TP only synchronizes planned layers and FSDP2 shards local state.
        # Establish one complete initial stage per DP/TP replica first.
        with torch.no_grad():
            for axis, size in (("tp", config.tp_size), ("dp", config.dp_size)):
                if size > 1:
                    group = self.mesh[axis].get_group()
                    source = dist.get_global_rank(group, 0)
                    for tensor in list(module.parameters()) + list(module.buffers()):
                        dist.broadcast(tensor, src=source, group=group)
        if config.tp_size > 1:
            from torch.distributed.tensor.parallel import parallelize_module

            parallelize_module(module, self.mesh["tp"], tensor_plan)
        self.execution_module = module
        if config.dp_size > 1:
            if config.tp_size > 1:
                from torch.distributed.fsdp import fully_shard

                fully_shard(module, mesh=self.mesh["dp"])
            else:
                from torch.nn.parallel import DistributedDataParallel

                self.execution_module = DistributedDataParallel(
                    module,
                    device_ids=[self.device.index],
                    process_group=self.mesh["dp"].get_group(),
                )
        self.schedule = None
        if config.pp_size > 1:
            from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

            stage = PipelineStage(
                self.execution_module,
                self.pp_rank,
                config.pp_size,
                self.device,
                example_input,
                group=self.mesh["pp"].get_group(),
            )
            kwargs = {}
            if "scale_grads" in inspect.signature(ScheduleGPipe).parameters:
                kwargs["scale_grads"] = False
            self.schedule = ScheduleGPipe(
                stage, config.microbatches, loss_fn=self._microbatch_loss, **kwargs
            )

    def parameters(self):
        """Return only this rank's local stage parameters for optimizer creation."""
        return self.module.parameters()

    def named_parameters(self):
        """Preserve caller stage names without DDP's module prefix."""
        return self.module.named_parameters()

    def _microbatch_loss(self, output: Any, target: Any) -> Any:
        if self.summed_loss is not None:
            loss = self.summed_loss(output, target)
            if loss.ndim != 0:
                raise ValueError("summed_loss must return a scalar sum")
            return loss * self._loss_scale
        loss = self.loss_fn(output, target)
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar mean loss")
        return loss / self.config.microbatches

    def forward_backward(self, inputs: Any, targets: Any) -> ParallelStepResult:
        """Accumulate gradients for one local batch; caller zeroes and steps them.

        All ranks must call this with identical shapes within their TP/PP group.
        Last-stage targets must match the loss output contract. No optimizer or
        data-loader work is hidden inside this operation.
        """
        import torch
        import torch.distributed as dist

        self._prepare_step(inputs, targets)
        losses = []
        output = None
        if self.schedule is None:
            outputs = []
            for index, (batch, target) in enumerate(
                zip(
                    inputs.chunk(self.config.microbatches),
                    targets.chunk(self.config.microbatches),
                )
            ):
                # DDP must suppress synchronization during forward as well as backward.
                accumulate = (
                    self.config.dp_size > 1
                    and self.config.tp_size == 1
                    and index < self.config.microbatches - 1
                )
                context = (
                    self.execution_module.no_sync() if accumulate else nullcontext()
                )
                with context:
                    prediction = self.execution_module(batch)
                    loss = self._microbatch_loss(prediction, target)
                    loss.backward()
                losses.append(loss.detach())
                outputs.append(prediction.detach())
            output = torch.cat(outputs)
        elif self.pp_rank == 0:
            self.schedule.step(inputs)
        elif self.pp_rank == self.config.pp_size - 1:
            output = self.schedule.step(target=targets, losses=losses)
        else:
            self.schedule.step()
        value = torch.zeros((), device=self.device)
        if losses:
            value.copy_(torch.stack([loss.detach() for loss in losses]).sum())
        if self.config.pp_size > 1:
            source = (
                self.dp_rank * self.config.pp_size + self.config.pp_size - 1
            ) * self.config.tp_size + self.tp_rank
            dist.broadcast(value, src=source, group=self.mesh["pp"].get_group())
        if self.config.dp_size > 1:
            dist.all_reduce(value, group=self.mesh["dp"].get_group())
            value.div_(self.config.dp_size)
        if not torch.isfinite(value):
            raise FloatingPointError("Nonfinite distributed loss")
        return ParallelStepResult(
            value.item(),
            output.detach() if output is not None else None,
            self._normalization_count,
        )

    def _prepare_step(self, inputs: Any, targets: Any) -> None:
        """Agree on preflight errors and DP-only counts before any model work."""
        import torch
        import torch.distributed as dist

        error = None
        count = torch.zeros((), dtype=torch.int64, device=self.device)
        try:
            expected_batch = self.input_shape[0] * self.config.microbatches
            if inputs.shape[0] != expected_batch or targets.shape[0] != expected_batch:
                raise ValueError(
                    "batch size must equal example microbatch * microbatches"
                )
            if inputs.device != self.device or targets.device != self.device:
                raise ValueError("inputs and targets must be on the stage device")
            if self.pp_rank == 0 and (
                tuple(inputs.shape[1:]) != self.input_shape[1:]
                or inputs.dtype != self.input_dtype
            ):
                raise ValueError(
                    "input shape/dtype differs from the fixed stage example"
                )
            if self.summed_loss is not None and self.pp_rank == self.config.pp_size - 1:
                count = self.summed_loss.normalization_count(targets)
                if (
                    not isinstance(count, torch.Tensor)
                    or count.ndim != 0
                    or count.dtype != torch.int64
                    or count.device != self.device
                    or count.item() < 0
                ):
                    raise ValueError(
                        "normalization_count must return a nonnegative device int64 scalar"
                    )
        except Exception as exc:
            error = exc
        failed = torch.tensor(int(error is not None), device=self.device)
        if self.config.world_size > 1:
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise ValueError(
                f"Parallel step preflight failed: {error or 'another rank rejected its batch'}"
            ) from error
        if self.summed_loss is None:
            return
        if self.pp_rank == self.config.pp_size - 1 and self.config.dp_size > 1:
            dist.all_reduce(count, group=self.mesh["dp"].get_group())
        if self.config.pp_size > 1:
            source = (
                self.dp_rank * self.config.pp_size + self.config.pp_size - 1
            ) * self.config.tp_size + self.tp_rank
            dist.broadcast(count, src=source, group=self.mesh["pp"].get_group())
        # Counts are already global across DP. Do not sum replicated TP/PP counts.
        bounds = torch.stack((count, -count))
        if self.config.world_size > 1:
            dist.all_reduce(bounds, op=dist.ReduceOp.MAX)
        maximum, negative_minimum = bounds.tolist()
        if maximum <= 0 or maximum != -negative_minimum:
            raise ValueError(
                "Global normalization_count must be positive and agree across ranks"
            )
        self._normalization_count = maximum
        # DDP/FSDP2 average gradients across DP, but not across TP or PP.
        self._loss_scale = self.config.dp_size / self._normalization_count

    def save_checkpoint(self, directory: Any, optimizer: Any, *, step: int) -> None:
        """Save a new, same-topology checkpoint; all ranks must participate."""
        from .distributed_checkpoint import save_checkpoint

        save_checkpoint(self, directory, optimizer, step=step)

    def load_checkpoint(self, directory: Any, optimizer: Any) -> int:
        """Load a trusted local checkpoint; topology changes are rejected."""
        from .distributed_checkpoint import load_checkpoint

        return load_checkpoint(self, directory, optimizer)
