# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午4:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Position-normalized forward KL and supervised cross entropy."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import DistillationConfig


@dataclass
class DistillationLossOutput:
    total: torch.Tensor
    hard_loss: torch.Tensor
    soft_loss: torch.Tensor
    valid_count: int


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    config: DistillationConfig | None = None,
) -> DistillationLossOutput:
    """Return (1-alpha)*CE + alpha*T**2*KL(teacher || student).

    soft_loss includes T**2. Unused endpoint terms are scalar zeros. All
    inputs are validated even at endpoints; enabled is a runtime routing flag.
    Causal validity requires both neighboring mask entries and a valid target.
    Finite inputs can still overflow FP32 arithmetic; reject non-finite
    intermediate log probabilities or loss terms rather than return NaN/Inf.

    Args:
        student_logits: Floating [B,C] classification or [B,L,V] causal logits.
        teacher_logits: Same shape/device as student logits; always detached.
        labels: Integer [B] or [B,L] targets; optional only when alpha=1.
        attention_mask: Binary sample [B] or token [B,L] validity mask.
        config: Objective settings, or defaults when omitted.
    Returns:
        FP32 total/hard/temperature-scaled soft losses and valid position count.
    Raises:
        TypeError: An argument has an unsupported type or dtype.
        ValueError: Invalid shapes/devices/values, no valid positions, or
            non-finite FP32 computation.
    """
    config = DistillationConfig() if config is None else config
    if not isinstance(config, DistillationConfig):
        raise TypeError("config must be DistillationConfig")
    config.validate()
    for name, tensor in (
        ("student_logits", student_logits),
        ("teacher_logits", teacher_logits),
    ):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating tensor")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student and teacher logits must have identical shape and classes/vocabulary"
        )
    if student_logits.device != teacher_logits.device:
        raise ValueError("student and teacher logits must use the same device")
    causal = config.task == "causal_lm"
    if student_logits.ndim != (3 if causal else 2) or any(
        d == 0 for d in student_logits.shape
    ):
        raise ValueError(
            "logits require nonempty [B,L,V] for causal_lm or [B,C] for classification"
        )
    if causal and student_logits.shape[1] < 2:
        raise ValueError("causal_lm requires sequence length at least two")
    shape, device = student_logits.shape[:-1], student_logits.device
    valid = torch.ones(shape, dtype=torch.bool, device=device)
    if labels is None:
        if config.alpha != 1:
            raise ValueError("labels are required unless alpha=1")
    else:
        if not isinstance(labels, torch.Tensor) or labels.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("labels must be an integer tensor (not bool)")
        if labels.shape != shape or labels.device != device:
            raise ValueError("labels must match logits positions and device")
        valid = labels != config.ignore_index
        if ((labels[valid] < 0) | (labels[valid] >= student_logits.shape[-1])).any():
            raise ValueError("labels are outside the classes/vocabulary range")
    if attention_mask is not None:
        if (
            not isinstance(attention_mask, torch.Tensor)
            or attention_mask.shape != shape
            or attention_mask.device != device
        ):
            raise ValueError("attention_mask must match logits positions and device")
        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("attention_mask must contain only 0 or 1")
    if causal:
        student_logits, teacher_logits = student_logits[:, :-1], teacher_logits[:, :-1]
        valid = valid[:, 1:]
        if labels is not None:
            labels = labels[:, 1:]
        if attention_mask is not None:
            valid = valid & attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
    elif attention_mask is not None:
        valid = valid & attention_mask.bool()
    count = int(valid.sum().item())
    if not count:
        raise ValueError("no valid positions for distillation")
    s = student_logits[valid].float()
    t = teacher_logits.detach()[valid].float()
    if not torch.isfinite(s).all() or not torch.isfinite(t).all():
        raise ValueError("logits must be representable in FP32")
    hard = s.new_zeros(())
    soft = s.new_zeros(())
    if config.alpha < 1:
        hard = F.cross_entropy(s, labels[valid].long())
    if config.alpha > 0:
        log_s = F.log_softmax(s / config.temperature, dim=-1)
        log_t = F.log_softmax(t / config.temperature, dim=-1)
        if not torch.isfinite(log_s).all() or not torch.isfinite(log_t).all():
            raise ValueError(
                "distillation log probabilities are not finite in FP32; reduce logits magnitude or adjust temperature"
            )
        soft = (log_t.exp() * (log_t - log_s)).sum(-1).mean() * config.temperature**2
    total = (1 - config.alpha) * hard + config.alpha * soft
    for name, value in (("hard_loss", hard), ("soft_loss", soft), ("total", total)):
        if not torch.isfinite(value).all():
            raise ValueError(
                f"distillation {name} is not finite in FP32; reduce logits magnitude or adjust temperature"
            )
    return DistillationLossOutput(total, hard, soft, count)
