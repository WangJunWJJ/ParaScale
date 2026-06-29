# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.optim as optim

from parascale.optimizers import (
    ExperimentalZeroOptimizer,
    ZeroStage,
    build_zero_plan,
    create_native_zero_redundancy_optimizer,
    wrap_zero_optimizer,
)
from parascale.parallel import (
    LocalPipelineExecutor,
    OneBitCompressor,
    SequenceParallelAdapter,
    SequenceParallelConfig,
    TensorParallelAdapter,
    TopKCompressor,
    build_gradient_compressor,
    build_pipeline_stages,
    column_parallel_linear,
    row_parallel_linear,
)


def test_sequence_parallel_adapter_plans_and_scatters_sequence_dim():
    adapter = SequenceParallelAdapter(
        SequenceParallelConfig(sp_size=2, tp_size=2, sequence_dim=1),
        rank=1,
        world_size=2,
    )
    tensor = torch.arange(2 * 6 * 4).view(2, 6, 4)

    spec = adapter.shard_spec(tuple(tensor.shape))
    shard = adapter.scatter(tensor)

    assert spec.start == 3
    assert spec.end == 6
    assert spec.local_shape == (2, 3, 4)
    assert torch.equal(shard, tensor[:, 3:6, :])
    assert (
        adapter.plan(tuple(tensor.shape))["status"]
        == "metadata_and_single_process_scatter"
    )


def test_topk_compressor_round_trip_shape_and_tracks_error_feedback():
    compressor = TopKCompressor(compression_ratio=0.25, error_feedback=True)
    tensor = torch.tensor([1.0, -5.0, 0.5, 3.0])

    compressed, metadata = compressor.compress(tensor, tensor_id=7)
    restored = compressor.decompress(compressed, metadata)
    stats = compressor.stats().to_dict()

    assert compressed.numel() == 1
    assert restored.shape == tensor.shape
    assert restored.abs().max().item() == 5.0
    assert stats["algorithm"] == "topk"
    assert stats["tracked_tensors"] == 1


def test_one_bit_compressor_round_trip_shape():
    compressor = OneBitCompressor(error_feedback=False)
    tensor = torch.tensor([-2.0, 0.0, 4.0])

    compressed, metadata = compressor.compress(tensor)
    restored = compressor.decompress(compressed, metadata)

    assert compressed.dtype == torch.uint8
    assert restored.shape == tensor.shape
    assert metadata["algorithm"] == "one_bit"


def test_build_gradient_compressor_factory():
    assert isinstance(build_gradient_compressor({"algorithm": "topk"}), TopKCompressor)
    assert isinstance(
        build_gradient_compressor({"algorithm": "1bit"}), OneBitCompressor
    )


def test_tensor_parallel_adapter_shards_and_local_linear_matches_slices():
    adapter = TensorParallelAdapter(tp_size=2, rank=1, dim=-1)
    tensor = torch.arange(2 * 4).view(2, 4)
    shard = adapter.shard(tensor)

    assert adapter.shard_spec(tuple(tensor.shape)).start == 2
    assert torch.equal(shard, tensor[:, 2:4])

    weight = torch.arange(3 * 4, dtype=torch.float32).view(3, 4)
    inputs = torch.ones(2, 4)
    row_output = row_parallel_linear(inputs, weight, tp_size=2, rank=0)
    col_output = column_parallel_linear(inputs, weight, tp_size=3, rank=1)

    assert row_output.shape == (2, 3)
    assert col_output.shape == (2, 1)


def test_pipeline_stage_plan_and_local_executor():
    stages = build_pipeline_stages(num_layers=7, pp_size=3, virtual_chunks=2)
    executor = LocalPipelineExecutor([lambda x: x + 1, lambda x: x * 2])

    assert [stage.to_dict()["num_layers"] for stage in stages] == [2, 2, 3]
    assert stages[0].virtual_chunks == 2
    assert executor.run(3) == 8
    assert executor.run_microbatches([1, 2]) == [4, 6]
    assert executor.plan()["status"] == "local_sequential_schedule"


def test_zero_plan_is_truthful_about_experimental_status():
    disabled = build_zero_plan(stage=0, world_size=8)
    stage3 = build_zero_plan(stage=3, world_size=8, offload_params=True)

    assert disabled.stage == ZeroStage.DISABLED
    assert disabled.implementation_status == "disabled"
    assert stage3.estimated_memory_savings > 1
    assert "requires_backend_parameter_sharding" in stage3.implementation_status


def test_experimental_zero_optimizer_wraps_base_optimizer_state():
    model = nn.Linear(2, 1)
    base = optim.SGD(model.parameters(), lr=0.1)
    wrapped = wrap_zero_optimizer(base, {"stage": 1, "world_size": 1})

    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    wrapped.step()
    wrapped.zero_grad()
    state = wrapped.state_dict()

    assert isinstance(wrapped, ExperimentalZeroOptimizer)
    assert state["zero_plan"]["stage"] == 1
    assert "base_optimizer" in state
    assert wrapped.get_memory_stats()["total_parameters"] == 3


def test_native_zero_stage_boundary_requires_stage1_and_process_group():
    model = nn.Linear(2, 1)
    with pytest.raises(NotImplementedError):
        create_native_zero_redundancy_optimizer(
            model.parameters(), optim.AdamW, stage=2, lr=0.1
        )
    with pytest.raises(RuntimeError):
        create_native_zero_redundancy_optimizer(
            model.parameters(), optim.AdamW, stage=1, lr=0.1
        )
