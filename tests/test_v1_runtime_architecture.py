# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

from parascale.cli import build_plan_payload
from parascale.config import ParaScaleConfig
from parascale.runtime import (
    BenchmarkProfileStore,
    BenchmarkResult,
    benchmark_result_from_train_payload,
    build_benchmark_plan,
    build_launch_plan,
    build_runtime_context,
    compare_benchmark_results,
)


def _vision_config():
    return {
        "parascale": {
            "task_type": "vision",
            "model_family": "vit",
            "target_scale": "single_node",
            "optimize_for": "throughput",
            "training_backend": "auto",
            "max_patch_tokens_per_batch": 65536,
        },
        "task": {
            "type": "vision",
            "model_family": "vit",
            "modalities": ["image"],
        },
        "model_profile": {
            "total_params": 300_000_000,
            "total_memory": 1_200_000_000,
            "num_layers": 24,
            "model_type": "vision_transformer",
        },
        "hardware_profile": {
            "num_gpus": 8,
            "gpus_per_node": 8,
            "gpu_memory": 24 * 1024**3,
            "available_memory": 22 * 1024**3,
        },
        "training": {
            "workload": "vision_synthetic",
        },
    }


def test_v1_config_accepts_task_and_patch_budget_fields():
    config = ParaScaleConfig(
        task_type="vision",
        model_family="vit",
        target_scale="single_node",
        optimize_for="throughput",
        max_patch_tokens_per_batch=8192,
        resolution_buckets=[[224, 224], [384, 384]],
    )

    payload = config.to_dict()

    assert payload["task_type"] == "vision"
    assert payload["model_family"] == "vit"
    assert payload["max_patch_tokens_per_batch"] == 8192
    assert payload["resolution_buckets"] == [[224, 224], [384, 384]]


def test_runtime_context_captures_workload_strategy_and_budgets():
    context = build_runtime_context(_vision_config(), mode="plan")

    payload = context.to_dict()

    assert payload["workload"]["task_type"] == "vision"
    assert payload["workload"]["model_family"] == "vit"
    assert payload["budgets"]["max_patch_tokens_per_batch"] == 65536
    assert payload["world_size"] == 8
    assert payload["strategy_plan"]["backend"] in {
        "native",
        "native_ddp",
        "fsdp",
        "deepspeed",
    }


def test_launch_plan_recommends_distributed_launcher_for_multigpu_context():
    context = build_runtime_context(_vision_config(), mode="train")
    plan = build_launch_plan(context, config_path="configs/vit.yaml")
    payload = plan.to_dict()

    assert payload["launcher"] in {"torchrun", "deepspeed"}
    assert payload["world_size"] == 8
    assert "--config" in payload["command"]
    assert "configs/vit.yaml" in payload["command"]


def test_launch_plan_builds_explicit_multinode_torchrun_rendezvous_command():
    config = _vision_config()
    config["hardware_profile"]["num_gpus"] = 2
    config["hardware_profile"]["gpus_per_node"] = 1
    context = build_runtime_context(config, mode="train")

    plan = build_launch_plan(
        context,
        config_path="configs/vit.yaml",
        nnodes=2,
        node_rank=1,
        master_addr="10.10.0.1",
        master_port=29901,
    )
    payload = plan.to_dict()

    assert payload["launcher"] == "torchrun"
    assert payload["world_size"] == 2
    assert payload["nproc_per_node"] == 1
    assert payload["nnodes"] == 2
    assert payload["node_rank"] == 1
    assert payload["master_addr"] == "10.10.0.1"
    assert payload["master_port"] == 29901
    assert "--standalone" not in payload["command"]
    assert "--nnodes=2" in payload["command"]
    assert "--node_rank=1" in payload["command"]
    assert "--master_addr=10.10.0.1" in payload["command"]
    assert "--master_port=29901" in payload["command"]


def test_launch_plan_preserves_quoted_entrypoint_arguments():
    context = build_runtime_context(_vision_config(), mode="train")
    plan = build_launch_plan(
        context,
        entrypoint='python -m parascale.cli train --note "hello world"',
        config_path="configs/vit.yaml",
    )

    assert "hello world" in plan.command
    assert plan.command[plan.command.index("--note") + 1] == "hello world"


def test_benchmark_plan_includes_three_layer_validation_for_vision():
    context = build_runtime_context(_vision_config(), mode="benchmark")
    plan = build_benchmark_plan(context)
    payload = plan.to_dict()
    names = {scenario["name"] for scenario in payload["scenarios"]}
    layers = {scenario["goal_layer"] for scenario in payload["scenarios"]}

    assert "baseline_train_smoke" in names
    assert "cost_aware_batching" in names
    assert "train_checkpoint_serve_loop" in names
    assert "layer1_not_worse_than_baselines" in layers
    assert "layer2_win_target_scenarios" in layers
    assert "layer3_system_loop_advantage" in layers


def test_benchmark_result_comparison_contract_scores_backend_speedup():
    comparison = compare_benchmark_results(
        [
            BenchmarkResult("parascale", {"samples_per_second": 120.0}),
            BenchmarkResult("deepspeed", {"samples_per_second": 100.0}),
        ],
        target_backend="parascale",
        baseline_backend="deepspeed",
        primary_metric="samples_per_second",
    )

    assert comparison.passed is True
    assert comparison.speedup == 1.2


def test_benchmark_result_from_train_payload_derives_samples_per_second():
    result = benchmark_result_from_train_payload(
        {
            "backend": "native",
            "steps_per_second": 10.0,
            "last_metrics": {"batch_size": 4},
        }
    )

    assert result.backend == "native"
    assert result.metrics["samples_per_second"] == 40.0


def test_benchmark_profile_store_extracts_runtime_tuning_inputs():
    profile = BenchmarkProfileStore().runtime_profile_from_metrics(
        {
            "peak_memory_bytes": 1234,
            "tokens": 512,
            "dataloader_wait_ms": 25.0,
            "pipeline_shard_read_ms": 9.0,
            "ignored": "not-a-number",
        }
    )

    assert profile["peak_memory_per_gpu"] == 1234
    assert profile["batch_tokens"] == 512
    assert profile["dataloader_wait_ms"] == 25.0
    assert profile["pipeline_shard_read_ms"] == 9.0
    assert "ignored" not in profile


def test_cli_plan_exposes_runtime_context_launch_and_benchmark_plans():
    payload = build_plan_payload(_vision_config())

    assert payload["mode"] == "plan"
    assert payload["runtime_context"]["workload"]["task_type"] == "vision"
    assert payload["launch_plan"]["launcher"] in {"torchrun", "deepspeed"}
    assert payload["benchmark_plan"]["scenarios"]
    assert payload["parallel_plan"]["dimensions"]["data"]["size"] >= 1


def test_cli_plan_passes_multinode_launch_config_to_launch_plan():
    config = _vision_config()
    config["hardware_profile"]["num_gpus"] = 2
    config["hardware_profile"]["gpus_per_node"] = 1
    config["launch"] = {
        "nnodes": 2,
        "node_rank": 1,
        "master_addr": "10.10.0.1",
        "master_port": 29901,
    }

    payload = build_plan_payload(config)
    launch = payload["launch_plan"]

    assert launch["nnodes"] == 2
    assert launch["node_rank"] == 1
    assert launch["master_addr"] == "10.10.0.1"
    assert launch["master_port"] == 29901
