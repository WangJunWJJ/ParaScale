# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

from dataclasses import dataclass

from parascale import (
    ClusterTopology,
    ParaScaleConfig,
    RuntimeProfile,
    StrategyPlan,
    build_parallel_plan,
    build_strategy_plan,
    tune_strategy_from_runtime,
)

GB = 1024**3


def test_parallel_plan_is_declarative_and_serializable():
    config = ParaScaleConfig(
        training_backend="fsdp",
        data_parallel_size=4,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )
    strategy = StrategyPlan(
        backend="fsdp", dp_size=4, tp_size=2, pp_size=1, zero_stage=0
    )

    plan = build_parallel_plan(config, strategy)
    payload = plan.to_dict()

    assert payload["backend"] == "fsdp"
    assert payload["dimensions"]["data"]["size"] == 4
    assert payload["dimensions"]["tensor"]["size"] == 2
    assert payload["sharding"] == "fsdp"
    assert "DataParallel" not in str(payload)


@dataclass
class ModelProfile:
    total_params: int
    total_memory: int
    num_layers: int
    model_type: str = "transformer"


@dataclass
class HardwareProfile:
    num_gpus: int
    gpu_memory: int
    available_memory: int
    gpus_per_node: int
    num_nodes: int = 1


def test_strategy_plan_is_torch_free_and_serializable():
    plan = StrategyPlan()

    assert plan.validate(1)
    assert plan.to_dict()["backend"] == "native"


def test_auto_strategy_keeps_small_single_gpu_native():
    model = ModelProfile(
        total_params=20_000_000,
        total_memory=20_000_000 * 4,
        num_layers=4,
        model_type="mlp",
    )
    hardware = HardwareProfile(
        num_gpus=1,
        gpu_memory=24 * GB,
        available_memory=22 * GB,
        gpus_per_node=1,
    )

    plan = build_strategy_plan(
        model, hardware, ParaScaleConfig(training_backend="auto")
    )

    assert plan.backend == "native"
    assert plan.dp_size == 1
    assert plan.tp_size == 1
    assert plan.pp_size == 1


def test_auto_strategy_selects_fsdp_for_large_multi_gpu_model():
    model = ModelProfile(
        total_params=2_000_000_000,
        total_memory=2_000_000_000 * 4,
        num_layers=32,
    )
    hardware = HardwareProfile(
        num_gpus=8,
        gpu_memory=80 * GB,
        available_memory=70 * GB,
        gpus_per_node=8,
    )

    plan = build_strategy_plan(
        model, hardware, ParaScaleConfig(training_backend="auto")
    )

    assert plan.backend == "fsdp"
    assert plan.precision == "bf16"
    assert plan.validate(8)
    assert plan.batch_policy == "token_budget"


def test_auto_strategy_selects_native_ddp_for_clip_benchmark_path():
    model = ModelProfile(
        total_params=150_000_000,
        total_memory=1_200_000_000,
        num_layers=18,
        model_type="clip_medium",
    )
    hardware = HardwareProfile(
        num_gpus=2,
        gpu_memory=24 * GB,
        available_memory=20 * GB,
        gpus_per_node=2,
    )
    config = ParaScaleConfig(
        task_type="multimodal",
        model_family="clip",
        optimize_for="throughput",
        training_backend="auto",
    )

    plan = build_strategy_plan(model, hardware, config)

    assert plan.backend == "native_ddp"
    assert plan.precision == "bf16"
    assert plan.ddp_comm_hook == "bf16_compress"
    assert plan.ddp_gradient_as_bucket_view is True
    assert plan.ddp_static_graph is True
    assert plan.communication_plan["ddp_hook"] == "bf16_compress"
    assert plan.validate(2)


def test_user_can_disable_native_ddp_comm_hook_with_none():
    model = ModelProfile(
        total_params=150_000_000,
        total_memory=1_200_000_000,
        num_layers=18,
        model_type="clip_medium",
    )
    hardware = HardwareProfile(
        num_gpus=2,
        gpu_memory=24 * GB,
        available_memory=20 * GB,
        gpus_per_node=2,
    )
    config = ParaScaleConfig(
        task_type="multimodal",
        model_family="clip",
        optimize_for="throughput",
        training_backend="auto",
        ddp_comm_hook="none",
    )

    plan = build_strategy_plan(model, hardware, config)

    assert plan.backend == "native_ddp"
    assert plan.ddp_comm_hook == "none"
    assert plan.communication_plan["ddp_hook"] == "none"


def test_strategy_plan_carries_native_ddp_bucket_cap():
    model = ModelProfile(
        total_params=150_000_000,
        total_memory=1_200_000_000,
        num_layers=18,
        model_type="clip_medium",
    )
    hardware = HardwareProfile(
        num_gpus=4,
        gpu_memory=48 * GB,
        available_memory=44 * GB,
        gpus_per_node=4,
    )
    config = ParaScaleConfig(
        task_type="multimodal",
        model_family="clip",
        optimize_for="throughput",
        training_backend="native_ddp",
        precision="bf16",
        ddp_bucket_cap_mb=100,
    )

    plan = build_strategy_plan(model, hardware, config)

    assert plan.ddp_comm_hook == "bf16_compress"
    assert plan.ddp_bucket_cap_mb == 100
    assert plan.communication_plan["bucket_cap_mb"] == 100


def test_auto_strategy_selects_native_ddp_for_yolo_without_hook_by_default():
    model = ModelProfile(
        total_params=80_000_000,
        total_memory=320_000_000,
        num_layers=30,
        model_type="yolo_world",
    )
    hardware = HardwareProfile(
        num_gpus=2,
        gpu_memory=24 * GB,
        available_memory=20 * GB,
        gpus_per_node=2,
    )
    config = ParaScaleConfig(
        task_type="vision",
        model_family="yolo_world",
        optimize_for="throughput",
        training_backend="auto",
    )

    plan = build_strategy_plan(model, hardware, config)

    assert plan.backend == "native_ddp"
    assert plan.precision == "bf16"
    assert plan.ddp_comm_hook == "none"
    assert plan.validate(2)


def test_auto_strategy_selects_deepspeed_zero3_offload_under_aggregate_pressure():
    model = ModelProfile(
        total_params=70_000_000_000,
        total_memory=70_000_000_000 * 4,
        num_layers=80,
    )
    hardware = HardwareProfile(
        num_gpus=8,
        gpu_memory=40 * GB,
        available_memory=34 * GB,
        gpus_per_node=8,
    )

    plan = build_strategy_plan(
        model, hardware, ParaScaleConfig(training_backend="auto")
    )

    assert plan.backend == "deepspeed"
    assert plan.zero_stage == 3
    assert plan.zero_offload is True
    assert plan.activation_checkpointing is True
    assert plan.validate(8)


def test_user_requested_deepspeed_backend_is_respected():
    model = ModelProfile(
        total_params=500_000_000,
        total_memory=500_000_000 * 4,
        num_layers=24,
    )
    hardware = HardwareProfile(
        num_gpus=4,
        gpu_memory=40 * GB,
        available_memory=36 * GB,
        gpus_per_node=4,
    )
    config = ParaScaleConfig(
        training_backend="deepspeed",
        zero_stage=2,
        batching_strategy="length_bucket",
    )

    plan = build_strategy_plan(model, hardware, config)

    assert plan.backend == "deepspeed"
    assert plan.zero_stage == 2
    assert plan.batch_policy == "length_bucket"


def test_strategy_plan_consumes_cluster_topology_metadata():
    model = ModelProfile(
        total_params=2_000_000_000,
        total_memory=2_000_000_000 * 4,
        num_layers=32,
    )
    topology = ClusterTopology.from_dicts(
        [
            {
                "hostname": "gpu-0",
                "devices": {"kind": "cuda", "count": 4, "memory_bytes": 80 * GB},
            },
            {
                "hostname": "npu-0",
                "devices": {"kind": "npu", "count": 2, "memory_bytes": 64 * GB},
            },
        ]
    )
    hardware = {
        "num_gpus": 1,
        "gpu_memory": 80 * GB,
        "available_memory": 70 * GB,
        "gpus_per_node": 4,
        "cluster_topology": topology,
    }

    plan = build_strategy_plan(
        model, hardware, ParaScaleConfig(training_backend="auto")
    )

    assert plan.validate(6)
    assert plan.topology["world_size"] == 6
    assert plan.topology["is_heterogeneous"] is True
    assert plan.topology["cross_group_parallelism"] == "weighted_data_parallel"
    assert any(
        "Heterogeneous topology detected" in warning for warning in plan.warnings
    )


def test_runtime_tuning_reduces_memory_pressure_and_switches_batching():
    plan = StrategyPlan(
        backend="deepspeed",
        zero_stage=2,
        batch_policy="sample",
        max_tokens_per_batch=10000,
    )
    hardware = HardwareProfile(
        num_gpus=8,
        gpu_memory=40 * GB,
        available_memory=30 * GB,
        gpus_per_node=8,
    )
    runtime = RuntimeProfile(
        peak_memory_per_gpu=31 * GB,
        padding_ratio=0.4,
        oom_count=1,
        batch_tokens=10000,
    )

    result = tune_strategy_from_runtime(plan, runtime, hardware, ParaScaleConfig())

    assert result.plan.activation_checkpointing is True
    assert result.plan.zero_stage == 3
    assert result.plan.zero_offload is True
    assert result.plan.batch_policy == "token_budget"
    assert result.config_updates["max_tokens_per_batch"] == 8000
