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


@dataclass
class ParallelStepResult:
    """DP-averaged loss and local output (only on the last pipeline stage)."""

    loss: float
    output: Any = None


class DistributedTrainer:
    """Fixed-shape, FP32 stage training on one to four CUDA devices.

    Loss must be a scalar mean over equal-sized samples/tokens, without ignored
    target-dependent denominators. Each TP/PP group receives the same local
    data; different DP replicas receive different samples. Caller owns optimizer
    steps, data sampling, process-group lifetime and job-level error handling.
    """

    def __init__(
        self,
        module: Any,
        config: DistributedParallelConfig,
        example_input: Any,
        loss_fn: Any,
        *,
        tensor_plan: dict[str, Any] | None = None,
    ) -> None:
        import torch
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh

        if not dist.is_initialized():
            raise RuntimeError("Initialize a CUDA/NCCL process group first")
        if torch.__version__.split("+")[0] != "2.9.1":
            raise RuntimeError(
                "DistributedTrainer requires the tested torch==2.9.1 runtime; install parascale[parallel]"
            )
        config.validate_world_size(dist.get_world_size())
        if dist.get_backend() != "nccl" or example_input.device.type != "cuda":
            raise ValueError("DistributedTrainer currently requires CUDA/NCCL")
        if (
            int(os.environ.get("LOCAL_WORLD_SIZE", config.world_size))
            != config.world_size
        ):
            raise ValueError("Only single-node execution has been validated")
        if example_input.ndim < 1 or example_input.shape[0] < 1:
            raise ValueError("example_input must contain one nonempty microbatch")
        if example_input.dtype != torch.float32:
            raise ValueError("Only float32 stage inputs are currently validated")
        if any(
            p.device != example_input.device or p.dtype != torch.float32
            for p in module.parameters()
        ):
            raise ValueError("Stage parameters must be float32 on the input device")
        if any(buffer.device != example_input.device for buffer in module.buffers()):
            raise ValueError("Stage buffers must be on the input device")
        if config.tp_size > 1 and not tensor_plan:
            raise ValueError("TP requires an explicit model-specific tensor_plan")
        if config.tp_size == 1 and tensor_plan:
            raise ValueError("tensor_plan requires tp_size > 1")
        if tensor_plan:
            missing = set(tensor_plan) - set(dict(module.named_modules()))
            if missing:
                raise ValueError(f"Unknown tensor_plan modules: {sorted(missing)}")
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
        self.loss_fn = loss_fn
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

        expected_batch = self.input_shape[0] * self.config.microbatches
        if inputs.shape[0] != expected_batch or targets.shape[0] != expected_batch:
            raise ValueError("batch size must equal example microbatch * microbatches")
        if inputs.device != self.device or targets.device != self.device:
            raise ValueError("inputs and targets must be on the stage device")
        if self.pp_rank == 0 and tuple(inputs.shape[1:]) != self.input_shape[1:]:
            raise ValueError("input shape differs from the fixed stage example")
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
            value.item(), output.detach() if output is not None else None
        )

    def save_checkpoint(self, directory: Any, optimizer: Any, *, step: int) -> None:
        """Save a new, same-topology checkpoint; all ranks must participate."""
        from .distributed_checkpoint import save_checkpoint

        save_checkpoint(self, directory, optimizer, step=step)

    def load_checkpoint(self, directory: Any, optimizer: Any) -> int:
        """Load a trusted local checkpoint; topology changes are rejected."""
        from .distributed_checkpoint import load_checkpoint

        return load_checkpoint(self, directory, optimizer)
