# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Parallel planning primitives for the v1 runtime.

The old eager wrapper APIs have intentionally been removed. Parallelism is now
described as plans that are executed by runtime backends such as FSDP,
DeepSpeed, or future ParaScale-native backends.
"""

from .communication import (
    CompressionStats,
    GradientCompressor,
    IdentityCompressor,
    OneBitCompressor,
    TopKCompressor,
    build_gradient_compressor,
)
from .pipeline import LocalPipelineExecutor, PipelineStage, build_pipeline_stages
from .plan import ParallelDimension, ParallelPlan, build_parallel_plan
from .sequence import SequenceParallelAdapter, SequenceParallelConfig, SequenceShardSpec
from .tensor import (
    TensorParallelAdapter,
    TensorShardSpec,
    column_parallel_linear,
    row_parallel_linear,
)

__all__ = [
    "ParallelDimension",
    "ParallelPlan",
    "build_parallel_plan",
    "SequenceParallelAdapter",
    "SequenceParallelConfig",
    "SequenceShardSpec",
    "TensorParallelAdapter",
    "TensorShardSpec",
    "column_parallel_linear",
    "row_parallel_linear",
    "PipelineStage",
    "LocalPipelineExecutor",
    "build_pipeline_stages",
    "CompressionStats",
    "GradientCompressor",
    "IdentityCompressor",
    "TopKCompressor",
    "OneBitCompressor",
    "build_gradient_compressor",
]
