# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Pinned PyTorch pipeline forwarding without obscuring DDP/FSDP ownership."""

from typing import Any

import torch
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe

from .precision import autocast_context


def validate_pipeline_contract(stage: Any, config: Any) -> None:
    """Agree on adjacent static tensor metadata before any pipeline send/receive."""
    import torch.distributed as dist

    def metadata(tensors):
        return [(tuple(tensor.shape), str(tensor.dtype)) for tensor in tensors]

    local = (metadata(stage.inputs_meta), metadata(stage.get_outputs_meta()))
    records = [None] * config.world_size
    dist.all_gather_object(records, local)
    for rank, (inputs, outputs) in enumerate(records):
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"Invalid pipeline boundary on rank {rank}: expected one input/output tensor"
            )
        _, pipeline_rank, _ = config.coordinates(rank)
        if pipeline_rank + 1 < config.pp_size:
            following = rank + config.tp_size
            if outputs != records[following][0]:
                raise ValueError(
                    f"Invalid pipeline boundary {rank}->{following}: "
                    f"output {outputs} does not match input {records[following][0]}"
                )


class PrecisionPipelineStage(PipelineStage):
    """Use the same autocast policy for inferred metadata and actual forwards."""

    def __init__(
        self,
        module: Any,
        index: int,
        count: int,
        device: Any,
        example: Any,
        *,
        group: Any,
        precision: str,
    ) -> None:
        self.precision = precision
        self.retain_outputs = False
        with autocast_context(precision, device):
            super().__init__(module, index, count, device, example, group=group)

    def forward_maybe_with_nosync(self, *args: Any, **kwargs: Any) -> Any:
        """Keep the native wrapper's synchronization behavior inside forward AMP."""
        with autocast_context(self.precision, self.device):
            return super().forward_maybe_with_nosync(*args, **kwargs)

    def forward_one_chunk(self, *args: Any, **kwargs: Any) -> Any:
        """Keep native backward caches, dropping only optional output references."""
        output = super().forward_one_chunk(*args, **kwargs)
        if not self.retain_outputs:
            self.output_chunks.clear()
        return output


class _OutputPolicy:
    """Make output ownership independent of the native scheduling algorithm."""

    def eval(
        self,
        *args: Any,
        target: Any = None,
        losses: Any = None,
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run pinned native evaluation while preserving subsequent training."""
        if self.scale_grads is not False:
            raise ValueError("Pipeline evaluation requires scale_grads=False")
        stage = self._stage
        original_backward = stage.has_backward
        original_retain_outputs = stage.retain_outputs
        try:
            with torch.no_grad():
                if not self._stage_initialized:
                    # PyTorch 2.9.1 must allocate backward-capable receive buffers
                    # before native eval temporarily disables backward execution.
                    stage.has_backward = self._has_backward
                    args_split, kwargs_split = self._split_inputs(args, kwargs)
                    self._initialize_stage(args_split[0], kwargs_split[0])
                return super().eval(
                    *args,
                    target=target,
                    losses=losses,
                    return_outputs=return_outputs,
                    **kwargs,
                )
        finally:
            stage.fwd_cache.clear()
            stage.output_chunks.clear()
            self._internal_losses.clear()
            stage.has_backward = original_backward
            stage.retain_outputs = original_retain_outputs

    def step(self, *args: Any, return_outputs: bool = False, **kwargs: Any) -> Any:
        self._stage.retain_outputs = return_outputs
        try:
            return super().step(*args, **kwargs)
        finally:
            self._stage.output_chunks.clear()
            self._stage.retain_outputs = False

    def _merge_outputs(self, output_chunks: list[Any]) -> Any:
        if not self._stage.retain_outputs:
            return None
        return super()._merge_outputs(output_chunks)


class OutputAwareGPipe(_OutputPolicy, ScheduleGPipe):
    """Native all-forward/all-backward scheduling with opt-in outputs."""


class OutputAware1F1B(_OutputPolicy, Schedule1F1B):
    """Native alternating forward/backward scheduling with opt-in outputs."""


def pipeline_schedule_type(name: str) -> type:
    """Resolve the validated schedule without changing native microbatch ordering."""
    return {"gpipe": OutputAwareGPipe, "1f1b": OutputAware1F1B}[name]
