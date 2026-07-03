# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import json
from dataclasses import dataclass
from pathlib import Path

from parascale import (
    BatchRuntimeStats,
    ParaScaleConfig,
    RuntimeProfile,
    StrategyPlan,
    apply_strategy_tuning,
    build_oom_retry_plan,
    build_runtime_profile,
    tune_strategy_from_runtime,
)
from parascale.cli import build_plan_payload

GB = 1024**3


@dataclass
class HardwareProfile:
    num_gpus: int = 8
    gpu_memory: int = 40 * GB
    available_memory: int = 32 * GB
    gpus_per_node: int = 8


def test_1_dry_run_profile_input_normalizes_runtime_stats():
    stats = BatchRuntimeStats(
        batch_tokens=10_000,
        valid_tokens=7_500,
        samples=5,
        step_time_seconds=2.5,
        peak_memory_per_gpu=24 * GB,
    )

    profile = build_runtime_profile(stats)

    assert profile.batch_tokens == 10_000
    assert profile.padding_ratio == 0.25
    assert profile.tokens_per_second == 3_000
    assert profile.samples_per_second == 2
    assert stats.to_runtime_profile().to_dict() == profile.to_dict()


def test_2_runtime_feedback_collects_tokens_padding_and_memory():
    profile = build_runtime_profile(
        {
            "batch_tokens": 4096,
            "valid_tokens": 2048,
            "samples": 4,
            "step_time_seconds": 1.0,
            "peak_memory_per_gpu": 30 * GB,
            "oom_count": 0,
        }
    )

    assert profile.padding_ratio == 0.5
    assert profile.tokens_per_second == 2048
    assert profile.peak_memory_per_gpu == 30 * GB


def test_3_profile_feedback_updates_batch_policy_checkpointing_and_zero():
    plan = StrategyPlan(
        backend="deepspeed",
        zero_stage=2,
        zero_offload=False,
        batch_policy="sample",
        max_tokens_per_batch=8192,
    )
    runtime = RuntimeProfile(
        peak_memory_per_gpu=31 * GB,
        padding_ratio=0.4,
        oom_count=1,
        batch_tokens=8192,
    )
    config = ParaScaleConfig(training_backend="deepspeed", zero_stage=2)

    tuning = tune_strategy_from_runtime(plan, runtime, HardwareProfile(), config)
    apply_strategy_tuning(config, tuning)

    assert tuning.plan.batch_policy == "token_budget"
    assert tuning.plan.activation_checkpointing is True
    assert tuning.plan.zero_stage == 3
    assert tuning.plan.zero_offload is True
    assert config.batching_strategy == "token_budget"
    assert config.enable_activation_checkpointing is True
    assert config.zero_stage == 3
    assert config.zero_offload is True


def test_4_oom_retry_plan_reduces_tokens_and_uses_safe_backend_knobs():
    plan = StrategyPlan(
        backend="fsdp",
        batch_policy="token_budget",
        max_tokens_per_batch=10_000,
        fsdp_state_dict_type="full",
    )
    runtime = RuntimeProfile(
        peak_memory_per_gpu=35 * GB,
        padding_ratio=0.1,
        batch_tokens=10_000,
    )

    retry = build_oom_retry_plan(plan, runtime, HardwareProfile(), ParaScaleConfig())

    assert retry.actions[0] == "oom_retry"
    assert retry.plan.activation_checkpointing is True
    assert retry.plan.max_tokens_per_batch == 8000
    assert retry.plan.fsdp_state_dict_type == "sharded"
    assert retry.config_updates["enable_activation_checkpointing"] is True
    assert retry.config_updates["fsdp_state_dict_type"] == "sharded"
    assert retry.decisions[0].action == "oom_retry"
    assert retry.decisions[0].evidence["oom_count"] == 1


def test_5_runtime_tuning_explains_padding_memory_and_dataloader_wait():
    plan = StrategyPlan(
        backend="fsdp",
        batch_policy="sample",
        max_tokens_per_batch=10_000,
    )
    runtime = RuntimeProfile(
        peak_memory_per_gpu=31 * GB,
        padding_ratio=0.4,
        batch_tokens=10_000,
        dataloader_wait_ms=50.0,
        images_per_second=80.0,
        pipeline_tokenizer_ms=32.0,
        pipeline_processor_ms=40.0,
        pipeline_image_decode_ms=3.0,
    )
    config = ParaScaleConfig(
        training_backend="fsdp",
        dataloader_num_workers=2,
        dataloader_prefetch_factor=2,
    )

    tuning = tune_strategy_from_runtime(plan, runtime, HardwareProfile(), config)
    actions = [decision.action for decision in tuning.decisions]

    assert "reduce_memory_pressure" in actions
    assert "switch_to_token_budget_batching" in actions
    assert "reduce_input_pipeline_jitter" in actions
    assert tuning.observed_profile["dataloader_wait_ms"] == 50.0
    assert tuning.observed_profile["pipeline_tokenizer_ms"] == 32.0
    assert tuning.thresholds["padding_ratio"] == 0.25
    assert tuning.config_updates["dataloader_persistent_workers"] is True
    pipeline_decision = [
        decision
        for decision in tuning.decisions
        if decision.action == "reduce_input_pipeline_jitter"
    ][0]
    assert pipeline_decision.evidence["dominant_pipeline_stage"] == "tokenizer_ms"
    assert "pipeline_breakdown_ms" in pipeline_decision.evidence


def test_6_plan_payload_exposes_human_readable_explain_block():
    payload = build_plan_payload(
        {
            "parascale": {
                "task_type": "multimodal",
                "model_family": "clip",
                "training_backend": "auto",
                "optimize_for": "throughput",
            },
            "model_profile": {
                "total_params": 150_000_000,
                "total_memory": 1_200_000_000,
                "num_layers": 18,
                "model_type": "clip_medium",
            },
            "hardware_profile": {
                "num_gpus": 2,
                "gpus_per_node": 2,
                "gpu_memory": 24 * GB,
                "available_memory": 20 * GB,
            },
            "runtime_profile": {
                "peak_memory_per_gpu": 19 * GB,
                "padding_ratio": 0.5,
                "batch_tokens": 8192,
                "dataloader_wait_ms": 60.0,
            },
        }
    )

    assert payload["strategy_plan"]["backend"] == "native_ddp"
    assert payload["strategy_plan"]["communication_plan"]["backend"] == "native_ddp"
    assert payload["communication_plan"]["ddp_hook"] == "bf16_compress"
    assert payload["communication_plan"]["use_no_sync"] is False
    assert payload["explain"]["communication_plan"]["ddp_hook"] == "bf16_compress"
    assert payload["explain"]["selected_backend"] == "native_ddp"
    assert payload["explain"]["runtime_decisions"]
    assert payload["explain"]["recommended_config_updates"]
    assert "Runtime tuner recommends" in payload["explain"]["summary"]


def test_6b_benchmark_profile_payload_drives_recommended_strategy_plan():
    payload = build_plan_payload(
        {
            "parascale": {
                "task_type": "multimodal",
                "model_family": "clip",
                "training_backend": "auto",
                "optimize_for": "throughput",
            },
            "model_profile": {
                "total_params": 150_000_000,
                "total_memory": 1_200_000_000,
                "num_layers": 18,
                "model_type": "clip_medium",
            },
            "hardware_profile": {
                "num_gpus": 2,
                "gpus_per_node": 2,
                "gpu_memory": 24 * GB,
                "available_memory": 20 * GB,
            },
            "benchmark_profile": {
                "metrics": {
                    "peak_memory_bytes": 19 * GB,
                    "padding_ratio": 0.5,
                    "tokens": 8192,
                    "dataloader_wait_ms": 60.0,
                    "stable_pipeline_host_to_device_ms": 42.0,
                }
            },
        }
    )

    assert payload["runtime_profile_source"] == "benchmark_profile.metrics"
    assert payload["runtime_tuning"]["observed_profile"]["batch_tokens"] == 8192
    assert payload["recommended_strategy_plan"]["batch_policy"] == "token_budget"
    assert payload["recommended_config_updates"]["cuda_prefetch"] is True
    assert payload["explain"]["runtime_decisions"]


def test_6c_benchmark_result_path_drives_recommended_strategy_plan():
    result_dir = Path(".pytest-parascale") / "benchmark-profile-path"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "benchmark.json"
    result_path.write_text(
        json.dumps(
            {
                "mode": "benchmark",
                "metrics": {
                    "peak_memory_bytes": 19 * GB,
                    "padding_ratio": 0.5,
                    "tokens": 8192,
                    "dataloader_wait_ms": 60.0,
                    "stable_pipeline_shard_read_ms": 35.0,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = build_plan_payload(
        {
            "parascale": {
                "task_type": "multimodal",
                "model_family": "clip",
                "training_backend": "auto",
            },
            "model_profile": {
                "total_params": 150_000_000,
                "total_memory": 1_200_000_000,
                "num_layers": 18,
                "model_type": "clip_medium",
            },
            "hardware_profile": {
                "num_gpus": 2,
                "gpus_per_node": 2,
                "gpu_memory": 24 * GB,
                "available_memory": 20 * GB,
            },
            "benchmark_result_path": str(result_path),
        }
    )

    assert payload["runtime_profile_source"] == "benchmark_result_path"
    assert payload["recommended_strategy_plan"]["batch_policy"] == "token_budget"
    assert (
        payload["recommended_config_updates"]["dataset_local_cache_dir"]
        == ".parascale_cache/dataset"
    )


def test_7_runtime_tuning_recommends_local_cache_for_slow_shard_reads():
    plan = StrategyPlan(backend="deepspeed", zero_stage=2)
    runtime = RuntimeProfile(
        dataloader_wait_ms=55.0,
        images_per_second=20.0,
        pipeline_shard_read_ms=35.0,
        pipeline_sample_decode_ms=5.0,
    )
    config = ParaScaleConfig(
        training_backend="deepspeed",
        zero_stage=2,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
    )

    tuning = tune_strategy_from_runtime(plan, runtime, HardwareProfile(), config)
    decision = [
        item
        for item in tuning.decisions
        if item.action == "reduce_input_pipeline_jitter"
    ][0]

    assert decision.evidence["dominant_pipeline_stage"] == "shard_read_ms"
    assert "enable_dataset_local_cache" in tuning.actions
    assert (
        tuning.config_updates["dataset_local_cache_dir"] == ".parascale_cache/dataset"
    )
    assert decision.expected_trade_off
    assert decision.to_dict()["expected_trade_off"] == decision.expected_trade_off


def test_8_runtime_tuning_recommends_cuda_prefetch_for_h2d_bottleneck():
    plan = StrategyPlan(backend="native_ddp")
    runtime = RuntimeProfile(
        dataloader_wait_ms=50.0,
        images_per_second=16.0,
        pipeline_host_to_device_ms=25.0,
        pipeline_sample_decode_ms=2.0,
    )
    config = ParaScaleConfig(
        training_backend="native_ddp",
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
    )

    tuning = tune_strategy_from_runtime(plan, runtime, HardwareProfile(), config)
    decision = [
        item
        for item in tuning.decisions
        if item.action == "reduce_input_pipeline_jitter"
    ][0]

    assert decision.evidence["dominant_pipeline_stage"] == "host_to_device_ms"
    assert "prefetch_to_device" in tuning.actions
    assert tuning.config_updates["cuda_prefetch"] is True
