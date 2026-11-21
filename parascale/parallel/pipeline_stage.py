# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Pinned PyTorch pipeline forwarding without obscuring DDP/FSDP ownership."""

from typing import Any

from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

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


class OutputAwareGPipe(ScheduleGPipe):
    """Preserve native scheduling while making merged outputs opt-in per step."""

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
