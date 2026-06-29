# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Auto parallel strategy APIs."""

from .hetero import (
    HeterogeneousParallelPlan,
    NodeGroup,
    build_heterogeneous_parallel_plan,
)
from .plan import BackendName, BatchPolicy, StrategyPlan
from .planner import build_strategy_plan
from .profiler import BatchRuntimeStats, RuntimeProfile, build_runtime_profile
from .tuner import (
    StrategyTuningResult,
    TuningDecision,
    apply_strategy_tuning,
    build_oom_retry_plan,
    tune_strategy_from_runtime,
)

__all__ = [
    "BackendName",
    "BatchPolicy",
    "StrategyPlan",
    "RuntimeProfile",
    "BatchRuntimeStats",
    "StrategyTuningResult",
    "TuningDecision",
    "NodeGroup",
    "HeterogeneousParallelPlan",
    "build_strategy_plan",
    "build_runtime_profile",
    "tune_strategy_from_runtime",
    "build_oom_retry_plan",
    "apply_strategy_tuning",
    "build_heterogeneous_parallel_plan",
]
