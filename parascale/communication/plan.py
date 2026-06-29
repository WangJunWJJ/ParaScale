# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:55
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Communication optimization plan for native distributed training."""

from __future__ import annotations

from parascale.contracts import CommunicationPlan

from .hooks import DdpHookPlan, recommend_ddp_hook


def build_communication_plan(
    *,
    backend: str,
    precision: str,
    task_type: str,
    model_family: str = "",
    gradient_accumulation_steps: int = 1,
    trainable_ratio: float | None = None,
    dataloader_wait_ms: float = 0.0,
) -> CommunicationPlan:
    backend = str(backend or "native")
    reasons: list[str] = []
    hook_plan: DdpHookPlan = recommend_ddp_hook(
        precision=precision,
        task_type=task_type,
        model_family=model_family,
        trainable_ratio=trainable_ratio,
    )
    if hook_plan.reason:
        reasons.append(hook_plan.reason)
    use_no_sync = int(gradient_accumulation_steps or 1) > 1 and backend == "native_ddp"
    if use_no_sync:
        reasons.append(
            "Gradient accumulation can use DDP no_sync for non-final micro-batches."
        )
    adapter_only_sync = bool(trainable_ratio is not None and trainable_ratio < 0.05)
    if adapter_only_sync:
        reasons.append(
            "Small trainable ratio suggests adapter-only gradient synchronization."
        )
    overlap_h2d = float(dataloader_wait_ms or 0.0) > 5.0
    if overlap_h2d:
        reasons.append(
            "High dataloader wait suggests overlapping H2D transfer with compute."
        )
    return CommunicationPlan(
        backend=backend,
        ddp_hook=hook_plan.hook,
        bucket_cap_mb=None,
        use_no_sync=use_no_sync,
        adapter_only_sync=adapter_only_sync,
        overlap_h2d=overlap_h2d,
        reasons=tuple(reasons),
        evidence={
            "precision": precision,
            "task_type": task_type,
            "model_family": model_family,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "trainable_ratio": trainable_ratio,
            "dataloader_wait_ms": dataloader_wait_ms,
        },
    )
