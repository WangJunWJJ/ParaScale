# -*- coding: utf-8 -*-
# @Time : 2026/5/3 下午9:57
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Production-oriented auto parallel strategy planning."""

from __future__ import annotations

from typing import Any, List

from parascale.communication import build_communication_plan, recommend_ddp_hook

from .plan import BackendName, BatchPolicy, StrategyPlan
from .utils import get_value, largest_divisor_at_most


def _estimate_training_memory(
    model_memory: int, precision: str, activation_factor: float
) -> int:
    precision_scale = 0.5 if precision in {"fp16", "bf16"} else 1.0
    persistent = model_memory * (1.0 + 1.0 + 2.0) * precision_scale
    activations = model_memory * activation_factor * precision_scale
    return int(persistent + activations)


def build_strategy_plan(
    model_profile: Any,
    hardware_profile: Any,
    config: Any,
) -> StrategyPlan:
    topology = _extract_topology(hardware_profile)
    world_size = max(1, int(get_value(hardware_profile, "num_gpus", 1)))
    topology_metadata = {}
    if topology is not None and topology.world_size > 0:
        world_size = topology.world_size
        topology_plan = topology.build_parallel_plan()
        topology_metadata = {
            "world_size": topology.world_size,
            "device_kinds": topology.device_kinds,
            "is_heterogeneous": topology.is_heterogeneous,
            "placement_policy": topology_plan.placement_policy,
            "cross_group_parallelism": topology_plan.cross_group_parallelism,
            "groups": [group.to_dict() for group in topology_plan.groups],
        }
    gpus_per_node = max(
        1, int(get_value(hardware_profile, "gpus_per_node", world_size))
    )
    available_memory = int(
        get_value(hardware_profile, "available_memory", 0)
        or get_value(hardware_profile, "gpu_memory", 0)
        or 8 * 1024**3
    )
    gpu_memory = int(
        get_value(hardware_profile, "gpu_memory", available_memory) or available_memory
    )

    total_params = int(get_value(model_profile, "total_params", 0))
    model_memory = int(get_value(model_profile, "total_memory", total_params * 4))
    num_layers = max(1, int(get_value(model_profile, "num_layers", 1) or 1))
    model_type = str(get_value(model_profile, "model_type", "unknown"))
    task_type = str(get_value(config, "task_type", "") or "").lower()
    model_family = str(get_value(config, "model_family", "") or "").lower()

    requested_backend = get_value(config, "training_backend", "native")
    optimize_for = str(get_value(config, "optimize_for", "balanced") or "balanced")
    precision = get_value(config, "precision", "bf16" if world_size >= 8 else "fp32")
    if precision == "fp32" and world_size >= 8:
        precision = "bf16"

    multimodal_family = model_family in {
        "clip",
        "siglip",
        "vlm",
        "blip",
        "llava",
    } or model_type in {"clip", "clip_medium", "siglip", "vlm"}
    vision_family = model_family in {
        "yolo",
        "yolo_world",
        "detection",
        "vit",
        "swin",
    } or model_type in {"vision", "detection", "yolo", "yolo_world"}
    if (
        precision == "fp32"
        and world_size > 1
        and optimize_for in {"throughput", "balanced"}
        and (
            task_type in {"vision", "multimodal"} or multimodal_family or vision_family
        )
    ):
        precision = "bf16"

    activation_factor = 1.0 if model_type == "transformer" else 0.5
    estimated_total = _estimate_training_memory(
        model_memory, precision, activation_factor
    )
    memory_margin = float(get_value(config, "strategy_memory_margin", 0.9))
    memory_budget = int(available_memory * memory_margin)
    params_b = total_params / 1e9
    model_exceeds_gpu = estimated_total > memory_budget
    explicit_offload = bool(get_value(config, "zero_offload", False))
    explicit_zero_stage = int(get_value(config, "zero_stage", 0) or 0)
    single_node_production = world_size > 1 and world_size <= 8
    native_ddp_fits = (
        single_node_production
        and not model_exceeds_gpu
        and params_b < 1.0
        and not explicit_offload
        and explicit_zero_stage < 3
    )

    max_tp = int(get_value(config, "max_tensor_parallel_size", 8))
    max_pp = int(get_value(config, "max_pipeline_parallel_size", 16))
    tp_size = 1
    pp_size = 1
    reasons: List[str] = []
    warnings: List[str] = []
    if topology_metadata:
        reasons.append(
            f"Using ClusterTopology world_size={topology_metadata['world_size']} "
            f"placement={topology_metadata['placement_policy']}."
        )
        if topology_metadata["is_heterogeneous"]:
            warnings.append(
                "Heterogeneous topology detected; default plan uses topology-aware weighted data parallel metadata."
            )

    if requested_backend == "auto":
        if (
            explicit_offload
            or explicit_zero_stage >= 3
            or estimated_total > memory_budget * max(1, world_size)
        ):
            backend: BackendName = "deepspeed"
            reasons.append(
                "Selected DeepSpeed because ZeRO-3/offload or aggregate memory pressure is required."
            )
        elif native_ddp_fits and (task_type == "multimodal" or multimodal_family):
            backend = "native_ddp"
            reasons.append(
                "Selected native-DDP for CLIP/VLM-style multimodal training; current benchmark policy prefers bf16 gradient compression on single-node GPUs."
            )
        elif native_ddp_fits and (task_type == "vision" or vision_family):
            backend = "native_ddp"
            reasons.append(
                "Selected native-DDP for small/medium vision training where full parameter replication is acceptable and communication overhead is lower than sharded backends."
            )
        elif model_exceeds_gpu or params_b >= 1.0 or world_size >= 8:
            backend = "fsdp"
            reasons.append("Selected FSDP for large model or multi-GPU sharding.")
        else:
            backend = "native"
            reasons.append(
                "Selected native backend for small model and low memory pressure."
            )
    else:
        backend = requested_backend
        reasons.append(f"Using user-requested backend: {backend}.")

    if backend == "deepspeed":
        zero_stage = max(
            explicit_zero_stage, 3 if model_exceeds_gpu or explicit_offload else 2
        )
        zero_offload = explicit_offload or estimated_total > memory_budget * max(
            1, world_size
        )
        checkpoint_policy = "deepspeed"
        fsdp_state_dict_type = get_value(config, "fsdp_state_dict_type", "full")
    elif backend == "fsdp":
        zero_stage = 0
        zero_offload = False
        checkpoint_policy = f"fsdp_{get_value(config, 'fsdp_state_dict_type', 'full')}"
        fsdp_state_dict_type = get_value(
            config, "fsdp_state_dict_type", "sharded" if world_size >= 8 else "full"
        )
    else:
        zero_stage = (
            explicit_zero_stage
            if bool(get_value(config, "zero_optimization", False))
            else 0
        )
        zero_offload = False
        checkpoint_policy = "rank0_file"
        fsdp_state_dict_type = get_value(config, "fsdp_state_dict_type", "full")

    ddp_comm_hook = str(get_value(config, "ddp_comm_hook", "auto") or "auto")
    ddp_comm_hook_explicit = "ddp_comm_hook" in set(
        getattr(config, "_explicit_fields", frozenset())
    )
    ddp_bucket_cap_mb = get_value(config, "ddp_bucket_cap_mb", None)
    ddp_gradient_as_bucket_view = bool(
        get_value(config, "ddp_gradient_as_bucket_view", True)
    )
    ddp_static_graph = bool(get_value(config, "ddp_static_graph", False))
    if backend == "native_ddp":
        ddp_gradient_as_bucket_view = True
        if task_type == "multimodal" or multimodal_family:
            ddp_static_graph = True
            if ddp_comm_hook == "auto":
                hook_plan = recommend_ddp_hook(
                    precision=precision,
                    task_type=task_type,
                    model_family=model_family,
                )
                ddp_comm_hook = hook_plan.hook
            elif ddp_comm_hook == "none" and not ddp_comm_hook_explicit:
                hook_plan = recommend_ddp_hook(
                    precision=precision,
                    task_type=task_type,
                    model_family=model_family,
                )
                ddp_comm_hook = hook_plan.hook
            if ddp_comm_hook != "none":
                reasons.append(
                    f"Enabled {ddp_comm_hook} for native-DDP based on verified CLIP/DataComp benchmark policy."
                )
        elif task_type == "vision" or vision_family:
            if ddp_comm_hook == "auto":
                ddp_comm_hook = "none"
            reasons.append(
                "Kept native-DDP communication hook disabled for detection by default until a detection-specific hook benchmark proves a win."
            )
        elif ddp_comm_hook == "auto":
            ddp_comm_hook = "none"

    requested_tp = int(get_value(config, "tensor_parallel_size", 1) or 1)
    requested_pp = int(get_value(config, "pipeline_parallel_size", 1) or 1)

    if backend in {"fsdp", "deepspeed"}:
        if requested_tp > 1:
            tp_size = largest_divisor_at_most(
                world_size, min(requested_tp, max_tp, gpus_per_node)
            )
            warnings.append(
                "TP with FSDP/DeepSpeed is experimental in v1 and requires explicit backend support."
            )
        if requested_pp > 1:
            pp_size = largest_divisor_at_most(
                world_size // max(1, tp_size), min(requested_pp, max_pp, num_layers)
            )
            warnings.append(
                "PP with FSDP/DeepSpeed is experimental in v1 and requires benchmark validation."
            )
        if tp_size == 1 and pp_size == 1 and world_size > 1:
            reasons.append(
                "Using backend-managed sharding/data parallelism without eager TP/PP wrappers."
            )
    elif world_size > 1 and params_b >= 1.0:
        tp_size = largest_divisor_at_most(world_size, min(max_tp, gpus_per_node))
        if tp_size > 1:
            reasons.append(f"Chose TP={tp_size} within node-local GPU budget.")

    remaining = max(1, world_size // max(1, tp_size))
    if (
        backend == "native"
        and world_size >= 16
        and num_layers >= 8
        and params_b >= 10.0
    ):
        pp_size = largest_divisor_at_most(remaining, min(max_pp, num_layers, remaining))
        if pp_size > 1:
            reasons.append(f"Chose PP={pp_size} for very large layered model.")

    dp_size = max(1, world_size // max(1, tp_size * pp_size))
    if dp_size * tp_size * pp_size != world_size:
        warnings.append(
            "Parallel factors did not exactly divide world_size; falling back to pure data parallel."
        )
        dp_size, tp_size, pp_size = world_size, 1, 1

    estimated_per_gpu = max(1, estimated_total // max(1, tp_size * pp_size))
    if backend in {"fsdp", "deepspeed"}:
        estimated_per_gpu = max(1, estimated_per_gpu // max(1, dp_size))

    activation_checkpointing = bool(
        get_value(config, "enable_activation_checkpointing", False)
    )
    if estimated_per_gpu > memory_budget:
        activation_checkpointing = True
        reasons.append("Enabled activation checkpointing due to memory pressure.")

    batch_policy: BatchPolicy = get_value(config, "batching_strategy", "sample")
    max_tokens = get_value(config, "max_tokens_per_batch", None)
    if batch_policy == "sample" and (
        model_type == "transformer" or estimated_per_gpu > memory_budget * 0.7
    ):
        batch_policy = "token_budget"
        if max_tokens is None:
            max_tokens = 8192 if gpu_memory < 40 * 1024**3 else 16384
        reasons.append("Selected token-budget batching to reduce padding waste.")

    if precision == "fp32" and estimated_per_gpu > memory_budget * 0.5:
        warnings.append(
            "fp32 may be inefficient for this plan; prefer bf16/fp16 if hardware supports it."
        )

    gradient_accumulation_steps = int(
        get_value(config, "gradient_accumulation_steps", 1) or 1
    )
    trainable_ratio = get_value(config, "trainable_ratio", None)
    if trainable_ratio is None:
        trainable_ratio = get_value(model_profile, "trainable_ratio", None)
    communication_plan = build_communication_plan(
        backend=backend,
        precision=precision,
        task_type=task_type,
        model_family=model_family,
        gradient_accumulation_steps=gradient_accumulation_steps,
        trainable_ratio=trainable_ratio,
        dataloader_wait_ms=0.0,
        bucket_cap_mb=ddp_bucket_cap_mb,
    ).to_dict()
    if backend == "native_ddp" and ddp_comm_hook != "none":
        communication_plan["ddp_hook"] = ddp_comm_hook
    elif backend == "native_ddp":
        communication_plan["ddp_hook"] = "none"

    return StrategyPlan(
        backend=backend,
        dp_size=dp_size,
        tp_size=tp_size,
        pp_size=pp_size,
        zero_stage=zero_stage,
        zero_offload=zero_offload,
        precision=precision,
        fsdp_state_dict_type=fsdp_state_dict_type,
        ddp_comm_hook=ddp_comm_hook,
        ddp_bucket_cap_mb=ddp_bucket_cap_mb,
        ddp_gradient_as_bucket_view=ddp_gradient_as_bucket_view,
        ddp_static_graph=ddp_static_graph,
        activation_checkpointing=activation_checkpointing,
        batch_policy=batch_policy,
        max_tokens_per_batch=max_tokens,
        checkpoint_policy=checkpoint_policy,
        estimated_memory_per_gpu=estimated_per_gpu,
        estimated_total_training_memory=estimated_total,
        reasons=reasons,
        warnings=warnings,
        topology=topology_metadata,
        communication_plan=communication_plan,
    )


def _extract_topology(hardware_profile: Any) -> Any:
    from parascale.core.cluster import ClusterTopology

    candidate = get_value(hardware_profile, "topology", None)
    if candidate is None:
        candidate = get_value(hardware_profile, "cluster_topology", None)
    if isinstance(candidate, ClusterTopology):
        return candidate
    if isinstance(candidate, dict) and "nodes" in candidate:
        return ClusterTopology.from_dicts(candidate["nodes"])
    if isinstance(candidate, list):
        return ClusterTopology.from_dicts(candidate)
    return None
