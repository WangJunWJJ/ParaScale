# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午4:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""A plain callable loss adapter, never a student model wrapper."""

from collections.abc import Mapping
from inspect import signature
from typing import Any

import torch

from .config import DistillationConfig
from .loss import distillation_loss


def _logits(output):
    value = (
        output
        if isinstance(output, torch.Tensor)
        else output.get("logits")
        if isinstance(output, Mapping)
        else getattr(output, "logits", None)
    )
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "model output must be a logits tensor, mapping['logits'], or object.logits"
        )
    return value


def _storage(tensor):
    return (
        (tensor.device, tensor.untyped_storage().data_ptr())
        if tensor.numel() and tensor.device.type != "meta"
        else None
    )


class OnlineDistillationLoss:
    """Mutates only teacher: freezes/clears grads, sets eval, places once.

    Pass an independent teacher. Neither teacher nor this callable is attached
    to student, so student checkpoints and optimizers remain unchanged.
    Classification uses optional batch['loss_mask'] for sample filtering;
    batch['attention_mask'] remains a teacher input. Causal LM uses the latter
    for adjacent-token loss validity as well.
    Native FP32 only; no distributed or mixed-precision integration is implied.
    """

    def __init__(
        self,
        student: torch.nn.Module,
        teacher: torch.nn.Module,
        config: DistillationConfig,
    ) -> None:
        """Freeze an independent teacher and snapshot objective settings.

        Args:
            student: Student module, inspected only for shared state.
            teacher: Teacher mutated by freezing, eval mode, and lazy placement.
            config: Validated settings copied independently of the caller.
        Returns:
            None.
        Raises:
            TypeError: Invalid module or configuration type.
            ValueError: Invalid settings or shared student/teacher storage.
        """
        if not isinstance(student, torch.nn.Module) or not isinstance(
            teacher, torch.nn.Module
        ):
            raise TypeError("student and teacher must be torch modules")
        if not isinstance(config, DistillationConfig):
            raise TypeError("config must be DistillationConfig")
        config.validate()
        student_tensors = [*student.parameters(), *student.buffers()]
        teacher_tensors = [*teacher.parameters(), *teacher.buffers()]
        identities = {id(p) for p in student_tensors}
        storages = {_storage(p) for p in student_tensors} - {None}
        if student is teacher or any(
            id(p) in identities or (_storage(p) is not None and _storage(p) in storages)
            for p in teacher_tensors
        ):
            raise ValueError(
                "student and teacher must not share parameter/buffer identity or storage aliases"
            )
        self.teacher = teacher
        self.config = DistillationConfig.from_dict(config.to_dict())
        self.last_metrics = {}
        self._device = None
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        teacher.eval()

    def __call__(self, output: Any, batch: Mapping[str, Any]) -> torch.Tensor:
        """Evaluate teacher and return the differentiable student loss.

        Args:
            output: Logits tensor, mapping['logits'], or object.logits.
            batch: Configured teacher inputs and optional labels. Classification
                uses loss_mask[B] for sample validity, not attention_mask[B,L].
                Causal LM uses attention_mask[B,L] for adjacent-token validity.
        Returns:
            Total FP32 loss; last_metrics contains detached Python numbers.
        Raises:
            TypeError: Invalid batch, output, labels, or logits types.
            ValueError: Incompatible teacher inputs/devices, invalid loss data,
                no valid positions, or non-finite FP32 computation.
        """
        if not isinstance(batch, Mapping):
            raise TypeError("distillation batch must be a mapping")
        student_logits = _logits(output)
        inputs = {
            key: batch[key]
            for key in self.config.input_keys
            if key in batch and key != "labels"
        }
        if not inputs:
            raise ValueError("batch has no matching teacher input_keys")
        try:
            signature(self.teacher.forward).bind(**inputs)
        except TypeError as exc:
            raise ValueError(
                f"batch inputs do not match teacher forward: {exc}"
            ) from exc
        if any(
            isinstance(value, torch.Tensor) and value.device != student_logits.device
            for value in inputs.values()
        ):
            raise ValueError("teacher inputs must match student logits device")
        if self._device is None:
            self.teacher.to(student_logits.device)
            self._device = student_logits.device
        elif self._device != student_logits.device:
            raise ValueError("student device changed after initial teacher placement")
        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = _logits(self.teacher(**inputs))
        result = distillation_loss(
            student_logits,
            teacher_logits,
            labels=batch.get("labels"),
            attention_mask=batch.get(
                "attention_mask" if self.config.task == "causal_lm" else "loss_mask"
            ),
            config=self.config,
        )
        self.last_metrics = {
            "total": result.total.detach().item(),
            "hard_loss": result.hard_loss.detach().item(),
            "soft_loss": result.soft_loss.detach().item(),
            "valid_count": result.valid_count,
        }
        return result.total
