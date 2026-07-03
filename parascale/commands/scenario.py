# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午7:13
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Benchmark scenario and config builders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name)
    if isinstance(value, dict):
        return value
    value = {}
    data[name] = value
    return value


def benchmark_matrix_scenario_config(
    scenario: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if scenario == "clip-datacomp-golden":
        return {
            "base_config": args.base_config
            or "configs/golden/clip_datacomp_vit_b.json",
            "output_dir": "runs/golden/clip_datacomp_vit_b",
            "markdown": "runs/golden/clip_datacomp_vit_b/report.md",
            "title": "DataComp CLIP ViT-B Golden Path",
            "workload_label": "DataComp WDS + pretrained CLIP ViT-B/32",
            "runs": [{"run_id": args.run_id or "clip_datacomp_vit_b"}],
        }
    if scenario == "vlm-lora-golden":
        return {
            "base_config": args.base_config or "configs/golden/vlm_lora_small.json",
            "output_dir": "runs/golden/vlm_lora_small",
            "markdown": "runs/golden/vlm_lora_small/report.md",
            "title": "Small VLM LoRA Golden Path",
            "workload_label": "DataComp WDS + LLaVA-OneVision 0.5B LoRA",
            "runs": [{"run_id": args.run_id or "vlm_lora_small"}],
        }
    if scenario == "vlm-lora-hf-clip":
        return {
            "base_config": args.base_config
            or "tests/benchmarks/configs/benchmark_vlm_lora_hf_clip_native_ddp.json",
            "output_dir": "runs/benchmarks/vlm_lora_hf_clip_backend_matrix",
            "markdown": (
                "tests/benchmarks/reports/"
                "vlm_lora_hf_clip_backend_matrix_report.zh.md"
            ),
            "title": "VLM LoRA HF CLIP Backend Matrix Report",
            "workload_label": (
                "DataComp WDS + real openai/clip-vit-base-patch32 frozen encoder "
                "+ LoRA adapter"
            ),
            "runs": [{"run_id": args.run_id or "hf_clip_lora"}],
        }
    if scenario == "vlm-lora-real":
        return {
            "base_config": args.base_config
            or "tests/benchmarks/configs/benchmark_vlm_lora_real_native_ddp.json",
            "output_dir": "runs/benchmarks/vlm_lora_real_backend_matrix",
            "markdown": (
                "tests/benchmarks/reports/" "vlm_lora_real_backend_matrix_report.zh.md"
            ),
            "title": "Real VLM LoRA Backend Matrix Report",
            "workload_label": (
                "DataComp WDS + real VLM processor adapter + PEFT LoRA "
                "target-module injection"
            ),
            "runs": [{"run_id": args.run_id or "real_vlm_lora"}],
        }
    if scenario == "yolo-world-large":
        variants = list(args.variants or ["m", "l", "x"])
        return {
            "base_config": args.base_config
            or (
                "tests/benchmarks/configs/"
                "benchmark_yolo_world_objects365_official_native_ddp.json"
            ),
            "output_dir": "runs/benchmarks/yolo_world_large_backend_matrix",
            "markdown": (
                "tests/benchmarks/reports/"
                "yolo_world_large_backend_matrix_report.zh.md"
            ),
            "title": "YOLO-World Large Backend Matrix Report",
            "workload_label": (
                "Objects365 cached YOLO data + YOLO-World m/l/x official loss"
            ),
            "runs": [
                {
                    "run_id": f"yolov8{variant}-worldv2",
                    "variant": variant,
                    "model_name": f"yolov8{variant}-worldv2",
                }
                for variant in variants
            ],
        }
    raise ValueError(f"unsupported benchmark matrix scenario: {scenario}")


def matrix_batch_sizes(args: argparse.Namespace) -> list[int | None]:
    sweep = getattr(args, "batch_size_sweep", None)
    if sweep:
        sizes = sorted({int(size) for size in sweep if int(size) > 0})
        if not sizes:
            raise ValueError("--batch-size-sweep must contain positive integers.")
        return sizes
    return [int(args.batch_size)] if args.batch_size is not None else [None]


def build_matrix_config(
    *,
    scenario: str,
    base_config: Dict[str, Any],
    run_spec: Dict[str, Any],
    backend: str,
    output_dir: Path,
    max_steps: int | None,
    warmup_steps: int | None,
    batch_size: int | None,
    num_samples: int | None,
) -> Dict[str, Any]:
    config_data = json.loads(json.dumps(base_config))
    parascale = section(config_data, "parascale")
    training = section(config_data, "training")
    data = section(config_data, "data")
    model = section(config_data, "model")
    profile = section(config_data, "model_profile")
    effective_backend = "deepspeed" if backend.startswith("deepspeed") else backend
    parascale["training_backend"] = effective_backend
    ckpt_dir = output_dir / f"{run_spec['run_id']}_{backend}_ckpt"
    parascale["checkpoint_save_path"] = str(ckpt_dir)
    training["checkpoint_dir"] = str(ckpt_dir)
    if max_steps is not None:
        training["max_steps"] = int(max_steps)
        training["benchmark_steps"] = int(max_steps)
    if warmup_steps is not None:
        training["warmup_steps"] = int(warmup_steps)
    if batch_size is not None:
        parascale["batch_size"] = int(batch_size)
        data["batch_size"] = int(batch_size)
    if num_samples is not None:
        data["num_samples"] = int(num_samples)
    if backend == "fsdp":
        parascale["fsdp_sharding_strategy"] = "full_shard"
        parascale["fsdp_state_dict_type"] = "full"
        parascale["fsdp_use_orig_params"] = True
    elif scenario in {"vlm-lora-real", "vlm-lora-golden"} and backend == "native_ddp":
        parascale["ddp_find_unused_parameters"] = True
        parascale["ddp_static_graph"] = False
        parascale["enable_activation_checkpointing"] = False
        model["activation_checkpointing"] = False
    elif effective_backend == "deepspeed":
        parascale["zero_optimization"] = True
        zero_stage = 3 if backend.endswith("zero3") else 2
        parascale["zero_stage"] = zero_stage
    if scenario == "yolo-world-large":
        model_name = run_spec["model_name"]
        model["path"] = f"/models/{model_name}.pt"
        model["variant"] = model_name
        profile["model_type"] = model_name
        if model_name.startswith("yolov8m"):
            profile["total_params"] = 29_000_000
        elif model_name.startswith("yolov8l"):
            profile["total_params"] = 48_000_000
        elif model_name.startswith("yolov8x"):
            profile["total_params"] = 73_000_000
    return config_data


def apply_pipeline_cache_args(
    config_data: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    parascale = section(config_data, "parascale")
    data = section(config_data, "data")
    config_data["parascale"] = parascale
    config_data["data"] = data
    dataset_cache_dir = getattr(args, "dataset_local_cache_dir", None)
    if dataset_cache_dir:
        parascale["dataset_local_cache_dir"] = str(dataset_cache_dir)
        data["dataset_local_cache_dir"] = str(dataset_cache_dir)
    if bool(getattr(args, "cuda_prefetch", False)):
        parascale["cuda_prefetch"] = True
    prefetch_device = getattr(args, "cuda_prefetch_device", None)
    if prefetch_device:
        parascale["cuda_prefetch_device"] = str(prefetch_device)
    if bool(getattr(args, "pipeline_cache", False)):
        parascale["pipeline_cache"] = True
        data["pipeline_cache"] = True
    cache_dir = getattr(args, "pipeline_cache_dir", None)
    if cache_dir:
        parascale["pipeline_cache_dir"] = str(cache_dir)
        data["pipeline_cache_dir"] = str(cache_dir)
    for attr in [
        "pipeline_cache_max_entries",
        "pipeline_cache_max_bytes",
        "pipeline_cache_ttl_seconds",
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            parascale[attr] = value
            data[attr] = value
    if bool(getattr(args, "prompt_template_cache", False)):
        parascale["prompt_template_cache"] = True
        data["prompt_template_cache"] = True
    prompt_dir = getattr(args, "prompt_template_cache_dir", None)
    if prompt_dir:
        parascale["prompt_template_cache_dir"] = str(prompt_dir)
        data["prompt_template_cache_dir"] = str(prompt_dir)
    if bool(getattr(args, "preprocess_in_workers", False)):
        parascale["preprocess_in_workers"] = True
        data["preprocess_in_workers"] = True
