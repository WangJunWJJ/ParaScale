# -*- coding: utf-8 -*-
# @Time : 2026/5/3 下午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Runtime feedback tuning for strategy plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .plan import StrategyPlan
from .profiler import RuntimeProfile
from .utils import get_value


@dataclass
class TuningDecision:
    action: str
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    threshold: Dict[str, Any] = field(default_factory=dict)
    config_updates: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "evidence": dict(self.evidence),
            "threshold": dict(self.threshold),
            "config_updates": dict(self.config_updates),
        }


@dataclass
class StrategyTuningResult:
    plan: StrategyPlan
    config_updates: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    decisions: List[TuningDecision] = field(default_factory=list)
    observed_profile: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "config_updates": dict(self.config_updates),
            "actions": list(self.actions),
            "warnings": list(self.warnings),
            "decisions": [decision.to_dict() for decision in self.decisions],
            "observed_profile": dict(self.observed_profile),
            "thresholds": dict(self.thresholds),
        }


def _clone_plan(plan: StrategyPlan) -> StrategyPlan:
    data = plan.to_dict()
    data.pop("strategy_type", None)
    return StrategyPlan(**data)


def _memory_budget(hardware_profile: Any, config: Any) -> int:
    available = int(
        get_value(hardware_profile, "available_memory", 0)
        or get_value(hardware_profile, "gpu_memory", 0)
        or 0
    )
    margin = float(get_value(config, "strategy_memory_margin", 0.9) or 0.9)
    return int(available * margin)


def _pipeline_breakdown(runtime_profile: RuntimeProfile) -> Dict[str, float]:
    return {
        "shard_read_ms": runtime_profile.pipeline_shard_read_ms,
        "tar_open_ms": runtime_profile.pipeline_tar_open_ms,
        "sample_decode_ms": runtime_profile.pipeline_sample_decode_ms,
        "sample_tensor_build_ms": runtime_profile.pipeline_sample_tensor_build_ms,
        "sample_build_ms": runtime_profile.pipeline_sample_build_ms,
        "collate_ms": runtime_profile.pipeline_collate_ms,
        "image_decode_ms": runtime_profile.pipeline_image_decode_ms,
        "prompt_template_ms": runtime_profile.pipeline_prompt_template_ms,
        "processor_ms": runtime_profile.pipeline_processor_ms,
        "tokenizer_ms": runtime_profile.pipeline_tokenizer_ms,
        "image_processor_ms": runtime_profile.pipeline_image_processor_ms,
        "host_to_device_ms": runtime_profile.pipeline_host_to_device_ms,
        "cuda_prefetch_h2d_ms": runtime_profile.pipeline_cuda_prefetch_h2d_ms,
        "cuda_prefetch_wait_ms": runtime_profile.pipeline_cuda_prefetch_wait_ms,
        "label_build_ms": runtime_profile.pipeline_label_build_ms,
        "processor_unaccounted_ms": (runtime_profile.pipeline_processor_unaccounted_ms),
    }


def _dominant_pipeline_stage(runtime_profile: RuntimeProfile) -> tuple[str, float]:
    leaf_breakdown = {
        "shard_read_ms": runtime_profile.pipeline_shard_read_ms,
        "tar_open_ms": runtime_profile.pipeline_tar_open_ms,
        "sample_decode_ms": runtime_profile.pipeline_sample_decode_ms,
        "sample_tensor_build_ms": runtime_profile.pipeline_sample_tensor_build_ms,
        "sample_build_ms": runtime_profile.pipeline_sample_build_ms,
        "collate_ms": runtime_profile.pipeline_collate_ms,
        "image_decode_ms": runtime_profile.pipeline_image_decode_ms,
        "prompt_template_ms": runtime_profile.pipeline_prompt_template_ms,
        "tokenizer_ms": runtime_profile.pipeline_tokenizer_ms,
        "image_processor_ms": runtime_profile.pipeline_image_processor_ms,
        "host_to_device_ms": runtime_profile.pipeline_host_to_device_ms,
        "cuda_prefetch_wait_ms": runtime_profile.pipeline_cuda_prefetch_wait_ms,
        "label_build_ms": runtime_profile.pipeline_label_build_ms,
        "processor_unaccounted_ms": (runtime_profile.pipeline_processor_unaccounted_ms),
    }
    if not leaf_breakdown:
        return "unknown", 0.0
    stage, value = max(leaf_breakdown.items(), key=lambda item: item[1])
    if value <= 0 and runtime_profile.pipeline_processor_ms > 0:
        return "processor_ms", runtime_profile.pipeline_processor_ms
    return stage, float(value)


def _pipeline_actions(stage: str) -> List[str]:
    if stage == "shard_read_ms":
        return ["enable_dataset_local_cache", "rebalance_wds_shards"]
    if stage == "tar_open_ms":
        return ["increase_shard_size", "enable_dataset_local_cache"]
    if stage == "sample_decode_ms":
        return ["use_faster_image_decoder", "increase_decode_workers"]
    if stage == "sample_tensor_build_ms":
        return ["cache_processed_images", "move_resize_normalize_to_workers"]
    if stage == "sample_build_ms":
        return ["profile_sample_build", "cache_processed_samples"]
    if stage == "collate_ms":
        return ["simplify_collate", "prebatch_or_cache_collated_fields"]
    if stage == "tokenizer_ms":
        return ["cache_tokenized_prompts", "enable_length_bucket_batching"]
    if stage == "image_processor_ms":
        return ["cache_processed_images", "move_resize_normalize_to_workers"]
    if stage == "image_decode_ms":
        return ["cache_decoded_images_or_use_wds", "increase_decode_workers"]
    if stage == "host_to_device_ms":
        return ["enable_pinned_memory", "prefetch_to_device"]
    if stage == "cuda_prefetch_wait_ms":
        return ["increase_prefetch_overlap", "check_cuda_stream_synchronization"]
    if stage == "prompt_template_ms":
        return ["cache_conversation_templates"]
    return ["profile_processor_internals"]


def tune_strategy_from_runtime(
    plan: StrategyPlan,
    runtime_profile: RuntimeProfile,
    hardware_profile: Any,
    config: Any,
) -> StrategyTuningResult:
    tuned = _clone_plan(plan)
    updates: Dict[str, Any] = {}
    actions: List[str] = []
    warnings: List[str] = []
    decisions: List[TuningDecision] = []

    budget = _memory_budget(hardware_profile, config)
    memory_pressure_threshold = budget
    near_memory_threshold = int(budget * 0.9) if budget > 0 else 0
    padding_threshold = 0.25
    dataloader_wait_threshold_ms = float(
        get_value(config, "tuner_dataloader_wait_threshold_ms", 20.0) or 20.0
    )
    near_memory_limit = budget > 0 and runtime_profile.peak_memory_per_gpu >= int(
        budget * 0.9
    )
    memory_pressure = runtime_profile.oom_count > 0 or (
        budget > 0 and runtime_profile.peak_memory_per_gpu > budget
    )

    if memory_pressure:
        tuned.activation_checkpointing = True
        updates["enable_activation_checkpointing"] = True
        actions.append("enable_activation_checkpointing")

        if tuned.max_tokens_per_batch:
            tuned.max_tokens_per_batch = max(1, int(tuned.max_tokens_per_batch * 0.8))
            updates["max_tokens_per_batch"] = tuned.max_tokens_per_batch
            actions.append("reduce_max_tokens_per_batch")

        if tuned.backend == "deepspeed":
            tuned.zero_stage = max(3, int(tuned.zero_stage or 0))
            tuned.zero_offload = True
            updates["zero_stage"] = tuned.zero_stage
            updates["zero_offload"] = True
            actions.append("enable_zero3_offload")
        elif tuned.backend == "fsdp":
            tuned.fsdp_state_dict_type = "sharded"
            tuned.checkpoint_policy = "fsdp_sharded"
            updates["fsdp_state_dict_type"] = "sharded"
            actions.append("use_sharded_fsdp_state_dict")

        warnings.append(
            "Runtime memory pressure detected; plan was tightened for retry safety."
        )
        decisions.append(
            TuningDecision(
                action="reduce_memory_pressure",
                reason="Peak memory or OOM indicates the current plan is not safe enough for retry.",
                evidence={
                    "peak_memory_per_gpu": runtime_profile.peak_memory_per_gpu,
                    "memory_budget": budget,
                    "oom_count": runtime_profile.oom_count,
                },
                threshold={
                    "memory_pressure_bytes": memory_pressure_threshold,
                    "oom_count": "> 0",
                },
                config_updates=dict(updates),
            )
        )
    elif near_memory_limit and not tuned.activation_checkpointing:
        tuned.activation_checkpointing = True
        updates["enable_activation_checkpointing"] = True
        actions.append("enable_activation_checkpointing")
        decisions.append(
            TuningDecision(
                action="enable_activation_checkpointing",
                reason="Peak memory is close to the configured memory budget.",
                evidence={
                    "peak_memory_per_gpu": runtime_profile.peak_memory_per_gpu,
                    "memory_budget": budget,
                },
                threshold={"near_memory_limit_bytes": near_memory_threshold},
                config_updates={"enable_activation_checkpointing": True},
            )
        )

    if runtime_profile.padding_ratio >= 0.25 and tuned.batch_policy != "token_budget":
        previous_policy = tuned.batch_policy
        tuned.batch_policy = "token_budget"
        updates["batching_strategy"] = "token_budget"
        actions.append("switch_to_token_budget_batching")
        if tuned.max_tokens_per_batch is None:
            tuned.max_tokens_per_batch = runtime_profile.batch_tokens or int(
                get_value(config, "max_tokens_per_batch", 8192) or 8192
            )
            updates["max_tokens_per_batch"] = tuned.max_tokens_per_batch
        decisions.append(
            TuningDecision(
                action="switch_to_token_budget_batching",
                reason="Padding waste is high; token-budget batching should reduce wasted tokens.",
                evidence={
                    "padding_ratio": runtime_profile.padding_ratio,
                    "batch_tokens": runtime_profile.batch_tokens,
                    "previous_batch_policy": previous_policy,
                },
                threshold={"padding_ratio": padding_threshold},
                config_updates={
                    key: updates[key]
                    for key in ["batching_strategy", "max_tokens_per_batch"]
                    if key in updates
                },
            )
        )

    if runtime_profile.dataloader_wait_ms >= dataloader_wait_threshold_ms:
        num_workers = int(get_value(config, "dataloader_num_workers", 4) or 4)
        prefetch = int(get_value(config, "dataloader_prefetch_factor", 2) or 2)
        dominant_stage, dominant_ms = _dominant_pipeline_stage(runtime_profile)
        pipeline_breakdown = _pipeline_breakdown(runtime_profile)
        stage_actions = _pipeline_actions(dominant_stage)
        updates.setdefault(
            "dataloader_num_workers", max(num_workers + 1, num_workers * 2)
        )
        updates.setdefault("dataloader_prefetch_factor", max(prefetch, 2))
        updates.setdefault("dataloader_persistent_workers", True)
        if dominant_stage == "host_to_device_ms":
            updates.setdefault("pin_memory", True)
            updates.setdefault("non_blocking_transfer", True)
            updates.setdefault("cuda_prefetch", True)
        if dominant_stage in {"shard_read_ms", "tar_open_ms"}:
            updates.setdefault("dataset_local_cache_dir", ".parascale_cache/dataset")
        if dominant_stage in {
            "tokenizer_ms",
            "image_processor_ms",
            "image_decode_ms",
            "prompt_template_ms",
            "sample_decode_ms",
            "sample_tensor_build_ms",
            "sample_build_ms",
        }:
            updates.setdefault("pipeline_cache", True)
        actions.append("increase_dataloader_parallelism")
        actions.extend(action for action in stage_actions if action not in actions)
        decisions.append(
            TuningDecision(
                action="reduce_input_pipeline_jitter",
                reason=(
                    "Dataloader wait time is high; the input pipeline profile "
                    f"points to {dominant_stage} as the dominant stage."
                ),
                evidence={
                    "dataloader_wait_ms": runtime_profile.dataloader_wait_ms,
                    "samples_per_second": runtime_profile.samples_per_second,
                    "images_per_second": runtime_profile.images_per_second,
                    "dominant_pipeline_stage": dominant_stage,
                    "dominant_pipeline_stage_ms": dominant_ms,
                    "pipeline_breakdown_ms": pipeline_breakdown,
                },
                threshold={"dataloader_wait_ms": dataloader_wait_threshold_ms},
                config_updates={
                    "dataloader_num_workers": updates["dataloader_num_workers"],
                    "dataloader_prefetch_factor": updates["dataloader_prefetch_factor"],
                    "dataloader_persistent_workers": updates[
                        "dataloader_persistent_workers"
                    ],
                    **{
                        key: updates[key]
                        for key in [
                            "pin_memory",
                            "non_blocking_transfer",
                            "cuda_prefetch",
                            "pipeline_cache",
                            "dataset_local_cache_dir",
                        ]
                        if key in updates
                    },
                },
            )
        )

    tuned.reasons.extend(actions)
    tuned.warnings.extend(warnings)
    return StrategyTuningResult(
        plan=tuned,
        config_updates=updates,
        actions=actions,
        warnings=warnings,
        decisions=decisions,
        observed_profile=runtime_profile.to_dict(),
        thresholds={
            "memory_budget_bytes": budget,
            "near_memory_limit_bytes": near_memory_threshold,
            "padding_ratio": padding_threshold,
            "dataloader_wait_ms": dataloader_wait_threshold_ms,
            "pipeline_stage_ms": "dominant stage is selected by max pipeline timer",
        },
    )


def apply_strategy_tuning(config: Any, tuning: StrategyTuningResult) -> Any:
    if isinstance(config, dict):
        config.update(tuning.config_updates)
        return config

    for key, value in tuning.config_updates.items():
        if hasattr(config, key):
            setattr(config, key, value)
    validate = getattr(config, "_validate", None)
    if callable(validate):
        validate()
    return config


def build_oom_retry_plan(
    plan: StrategyPlan,
    runtime_profile: RuntimeProfile,
    hardware_profile: Any,
    config: Any,
) -> StrategyTuningResult:
    retry_profile = RuntimeProfile(**runtime_profile.to_dict())
    retry_profile.oom_count = max(1, retry_profile.oom_count)
    result = tune_strategy_from_runtime(plan, retry_profile, hardware_profile, config)

    if "oom_retry" not in result.actions:
        result.actions.insert(0, "oom_retry")

    result.plan.batch_policy = "token_budget"
    result.plan.activation_checkpointing = True
    result.config_updates["batching_strategy"] = "token_budget"
    result.config_updates["enable_activation_checkpointing"] = True

    if result.plan.max_tokens_per_batch is None:
        base_tokens = retry_profile.batch_tokens or int(
            get_value(config, "max_tokens_per_batch", 8192) or 8192
        )
        result.plan.max_tokens_per_batch = max(1, int(base_tokens * 0.8))
        result.config_updates["max_tokens_per_batch"] = result.plan.max_tokens_per_batch

    if result.plan.backend == "deepspeed":
        result.plan.zero_stage = max(3, int(result.plan.zero_stage or 0))
        result.plan.zero_offload = True
        result.config_updates["zero_stage"] = result.plan.zero_stage
        result.config_updates["zero_offload"] = True
    elif result.plan.backend == "fsdp":
        result.plan.fsdp_state_dict_type = "sharded"
        result.plan.checkpoint_policy = "fsdp_sharded"
        result.config_updates["fsdp_state_dict_type"] = "sharded"

    result.plan.reasons.append("oom_retry")
    result.observed_profile = retry_profile.to_dict()
    result.thresholds.setdefault("oom_count", "> 0")
    result.decisions.insert(
        0,
        TuningDecision(
            action="oom_retry",
            reason="Previous step hit OOM; retry plan uses safer memory settings before rerunning.",
            evidence={
                "oom_count": retry_profile.oom_count,
                "peak_memory_per_gpu": retry_profile.peak_memory_per_gpu,
                "batch_tokens": retry_profile.batch_tokens,
            },
            threshold={"oom_count": "> 0"},
            config_updates=dict(result.config_updates),
        ),
    )
    return result
