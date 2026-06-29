# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Training and inference runtime entry points."""

from parascale.reporting.benchmark import (
    BenchmarkComparison,
    BenchmarkPlan,
    BenchmarkResult,
    BenchmarkScenario,
    benchmark_result_from_train_payload,
    build_benchmark_plan,
    compare_benchmark_results,
)
from parascale.workloads.tiny import build_tiny_torch_components

from .backends import (
    AscendNativeTrainingBackend,
    DeepSpeedTrainingBackend,
    FSDPTrainingBackend,
    NativeTrainingBackend,
    TrainingBackend,
    TrainingBackendRegistry,
    create_runtime_training_backend,
    default_training_backend_registry,
)
from .context import RuntimeContext, WorkloadDescriptor, build_runtime_context
from .factory import (
    ServingModelRegistry,
    TinyTorchServingAdapter,
    build_optimizer_for_model,
    build_serving_model_from_checkpoint,
    build_training_components,
    default_serving_model_registry,
    default_workload_registry,
    load_tiny_torch_serving_model,
)
from .inference import InferenceEngine
from .inference.engine import ServeEngine, ServeState
from .launcher import LaunchPlan, build_launch_plan
from .profiles import BenchmarkProfileStore
from .specs import (
    ClipContrastiveSpec,
    TinyTorchWorkloadSpec,
    VisionSyntheticSpec,
    VlmLoraSpec,
    YoloWorldSpec,
)
from .training import TrainEngine, TrainState
from .workloads import WorkloadRegistry

__all__ = [
    "TrainEngine",
    "TrainState",
    "RuntimeContext",
    "WorkloadDescriptor",
    "build_runtime_context",
    "LaunchPlan",
    "build_launch_plan",
    "BenchmarkPlan",
    "BenchmarkScenario",
    "BenchmarkResult",
    "BenchmarkComparison",
    "build_benchmark_plan",
    "benchmark_result_from_train_payload",
    "compare_benchmark_results",
    "ServeEngine",
    "InferenceEngine",
    "ServeState",
    "BenchmarkProfileStore",
    "TinyTorchWorkloadSpec",
    "VisionSyntheticSpec",
    "ClipContrastiveSpec",
    "VlmLoraSpec",
    "YoloWorldSpec",
    "TinyTorchServingAdapter",
    "ServingModelRegistry",
    "WorkloadRegistry",
    "build_serving_model_from_checkpoint",
    "build_optimizer_for_model",
    "build_training_components",
    "build_tiny_torch_components",
    "default_serving_model_registry",
    "default_workload_registry",
    "load_tiny_torch_serving_model",
    "TrainingBackend",
    "AscendNativeTrainingBackend",
    "NativeTrainingBackend",
    "FSDPTrainingBackend",
    "DeepSpeedTrainingBackend",
    "TrainingBackendRegistry",
    "create_runtime_training_backend",
    "default_training_backend_registry",
]
