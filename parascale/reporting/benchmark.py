# -*- coding: utf-8 -*-
# @Time : 2026/6/9 下午5:59
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Benchmark planning for ParaScale's three practical goals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from parascale.contracts import MetricContract
from parascale.runtime.context import RuntimeContext


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    goal_layer: str
    task_type: str
    backend: str
    metrics: List[str]
    acceptance: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "goal_layer": self.goal_layer,
            "task_type": self.task_type,
            "backend": self.backend,
            "metrics": list(self.metrics),
            "acceptance": dict(self.acceptance),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class BenchmarkPlan:
    scenarios: List[BenchmarkScenario]
    compare_backends: List[str]
    primary_metrics: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compare_backends": list(self.compare_backends),
            "primary_metrics": list(self.primary_metrics),
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    metrics: Dict[str, float]
    status: str = "ok"
    notes: List[str] = field(default_factory=list)

    def metric(self, name: str, default: float = 0.0) -> float:
        value = self.metrics.get(name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "metrics": dict(self.metrics),
            "status": self.status,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class BenchmarkComparison:
    target_backend: str
    baseline_backend: str
    primary_metric: str
    target_value: float
    baseline_value: float
    higher_is_better: bool = True
    speedup: float = 0.0
    passed: bool = False
    tolerance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_backend": self.target_backend,
            "baseline_backend": self.baseline_backend,
            "primary_metric": self.primary_metric,
            "target_value": self.target_value,
            "baseline_value": self.baseline_value,
            "higher_is_better": self.higher_is_better,
            "speedup": self.speedup,
            "passed": self.passed,
            "tolerance": self.tolerance,
        }


def build_benchmark_plan(
    context: RuntimeContext, compare_backends: List[str] | None = None
) -> BenchmarkPlan:
    compare = compare_backends or _default_compare_backends(context)
    metrics = [
        "step_time_ms",
        "dataloader_wait_ms",
        "checkpoint_time_ms",
        *MetricContract().stable_metric_names(),
    ]
    scenarios = [
        BenchmarkScenario(
            name="baseline_train_smoke",
            goal_layer="layer1_not_worse_than_baselines",
            task_type=context.workload.task_type,
            backend=context.strategy_plan.backend,
            metrics=metrics,
            acceptance={"must_checkpoint": True, "must_resume": True},
            notes=[
                "Establish native/FSDP/DeepSpeed parity before claiming performance wins."
            ],
        )
    ]
    if context.workload.task_type in {"vision", "multimodal"}:
        scenarios.append(
            BenchmarkScenario(
                name="cost_aware_batching",
                goal_layer="layer2_win_target_scenarios",
                task_type=context.workload.task_type,
                backend=context.strategy_plan.backend,
                metrics=[
                    "images_per_second",
                    "patch_tokens_per_second",
                    "padding_ratio",
                    "decode_time_ms",
                    "augment_time_ms",
                    "host_to_device_time_ms",
                    "peak_memory_bytes",
                ],
                acceptance={
                    "requires_patch_token_budget": context.config.max_patch_tokens_per_batch
                    is not None
                },
                notes=[
                    "This is ParaScale's primary path to outperform generic distributed trainers."
                ],
            )
        )
    if context.workload.task_type == "multimodal":
        scenarios.extend(
            [
                BenchmarkScenario(
                    name="vlm_lora_finetune",
                    goal_layer="layer3_vlm_productivity",
                    task_type="multimodal",
                    backend=context.strategy_plan.backend,
                    metrics=[
                        "tokens_per_second",
                        "images_per_second",
                        "padding_ratio",
                        "trainable_parameter_ratio",
                        "peak_memory_bytes",
                    ],
                    acceptance={
                        "requires_lora_adapter": True,
                        "video_understanding_required": False,
                    },
                    notes=[
                        "First multimodal training target: image-text VLM LoRA/QLoRA style finetuning."
                    ],
                ),
                BenchmarkScenario(
                    name="clip_style_contrastive",
                    goal_layer="layer3_multimodal_retrieval",
                    task_type="multimodal",
                    backend=context.strategy_plan.backend,
                    metrics=[
                        "image_text_pairs_per_second",
                        "embedding_throughput",
                        "contrastive_loss",
                        "padding_ratio",
                        "peak_memory_bytes",
                    ],
                    acceptance={
                        "requires_image_text_pairs": True,
                        "requires_symmetric_loss": True,
                    },
                    notes=[
                        "Second multimodal target: CLIP-style image-text contrastive training."
                    ],
                ),
            ]
        )
    scenarios.append(
        BenchmarkScenario(
            name="train_checkpoint_serve_loop",
            goal_layer="layer3_system_loop_advantage",
            task_type=context.workload.task_type,
            backend=context.strategy_plan.backend,
            metrics=[
                "checkpoint_time_ms",
                "restore_time_ms",
                "serve_latency_ms",
                "serve_throughput",
            ],
            acceptance={"mock_serving_allowed": False},
            notes=[
                "Validates that training artifacts are directly consumable by serving runtime."
            ],
        )
    )
    return BenchmarkPlan(
        scenarios=scenarios, compare_backends=compare, primary_metrics=metrics
    )


def _default_compare_backends(context: RuntimeContext) -> List[str]:
    if context.world_size <= 1:
        return ["native"]
    backends = ["native", "fsdp"]
    if (
        context.strategy_plan.zero_stage >= 2
        or context.strategy_plan.backend == "deepspeed"
    ):
        backends.append("deepspeed")
    return backends


def compare_benchmark_results(
    results: List[BenchmarkResult],
    *,
    target_backend: str,
    baseline_backend: str = "deepspeed",
    primary_metric: str = "samples_per_second",
    higher_is_better: bool = True,
    tolerance: float = 0.02,
) -> BenchmarkComparison:
    by_backend = {result.backend: result for result in results if result.status == "ok"}
    if target_backend not in by_backend:
        raise ValueError(f"target backend result is missing: {target_backend}")
    if baseline_backend not in by_backend:
        raise ValueError(f"baseline backend result is missing: {baseline_backend}")
    target = by_backend[target_backend].metric(primary_metric)
    baseline = by_backend[baseline_backend].metric(primary_metric)
    if baseline <= 0:
        speedup = 0.0
    elif higher_is_better:
        speedup = target / baseline
    else:
        speedup = baseline / max(target, 1e-12)
    threshold = 1.0 - float(tolerance)
    passed = speedup >= threshold
    return BenchmarkComparison(
        target_backend=target_backend,
        baseline_backend=baseline_backend,
        primary_metric=primary_metric,
        target_value=target,
        baseline_value=baseline,
        higher_is_better=higher_is_better,
        speedup=speedup,
        passed=passed,
        tolerance=float(tolerance),
    )


def benchmark_result_from_train_payload(payload: Dict[str, Any]) -> BenchmarkResult:
    backend = str(payload.get("backend", "native"))
    last_metrics = (
        payload.get("last_metrics", {})
        if isinstance(payload.get("last_metrics"), dict)
        else {}
    )
    metrics = {
        "steps_per_second": float(payload.get("steps_per_second", 0.0) or 0.0),
        "step_time_ms": 1000.0
        / max(float(payload.get("steps_per_second", 0.0) or 0.0), 1e-9),
        "samples_per_second": float(last_metrics.get("samples_per_second", 0.0) or 0.0),
        "images_per_second": float(
            last_metrics.get("images_per_second", payload.get("images_per_second", 0.0))
            or 0.0
        ),
        "patch_tokens_per_second": float(
            last_metrics.get("patch_tokens_per_second", 0.0) or 0.0
        ),
        "tokens_per_second": float(last_metrics.get("tokens_per_second", 0.0) or 0.0),
        "image_text_pairs_per_second": float(
            last_metrics.get("image_text_pairs_per_second", 0.0) or 0.0
        ),
        "padding_ratio": float(last_metrics.get("padding_ratio", 0.0) or 0.0),
        "peak_memory_bytes": float(last_metrics.get("peak_memory_bytes", 0.0) or 0.0),
        "allocated_memory_bytes": float(
            last_metrics.get("allocated_memory_bytes", 0.0) or 0.0
        ),
    }
    for name in ("dataloader_wait_ms", *MetricContract().stable_metric_names()):
        if isinstance(last_metrics.get(name), (int, float)):
            metrics[name] = float(last_metrics[name])
    if (
        metrics["samples_per_second"] <= 0
        and last_metrics.get("batch_size")
        and metrics["steps_per_second"] > 0
    ):
        metrics["samples_per_second"] = (
            float(last_metrics["batch_size"]) * metrics["steps_per_second"]
        )
    return BenchmarkResult(backend=backend, metrics=metrics)
