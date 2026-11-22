# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Consumed-token and update state for same-topology stage checkpoints."""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from typing import Any

from parascale.data.text import TokenBatchCursor
from parascale.optimizers.schedule import SchedulerSpec

from .random_state import capture_rng_state, restore_rng_state


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _validate_host_rng(state: Any) -> None:
    """Check host RNG on private generators, never by changing global streams."""
    import numpy as np

    if not isinstance(state, dict) or set(state) != {"python", "numpy"}:
        raise ValueError("Stage checkpoint requires Python and NumPy RNG state")
    if not isinstance(state["python"], tuple) or len(state["python"]) != 3:
        raise ValueError("Invalid Python RNG state")
    cached_python = state["python"][2]
    if cached_python is not None and (
        type(cached_python) is not float or not math.isfinite(cached_python)
    ):
        raise ValueError("Invalid Python RNG Gaussian cache")
    random.Random().setstate(state["python"])
    algorithm, keys, position, has_gauss, cached = state["numpy"]
    if (
        algorithm != "MT19937"
        or not isinstance(keys, list)
        or len(keys) != 624
        or any(type(key) is not int or not 0 <= key < 2**32 for key in keys)
        or type(position) is not int
        or not 0 <= position <= 624
        or type(has_gauss) is not int
        or has_gauss not in (0, 1)
        or type(cached) is not float
        or not math.isfinite(cached)
    ):
        raise ValueError("Invalid NumPy RNG state")
    np.random.RandomState().set_state(
        (algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached)
    )


class StageTrainingCheckpoint:
    """Bind fixed-batch token training progress to an atomic stage checkpoint.

    Uses the existing engine's state/data_resume/scheduler fields; does not own
    execution. Only full fixed-size microbatches, one scheduler step per optimizer
    update, and counters starting at zero are admitted. Samples are rank-local
    consumed samples, not counts multiplied by replicated TP/PP ranks. Schedulers
    must be built from SchedulerSpec before optimizer restoration. No GradScaler,
    skipped updates, arbitrary lambdas or replay-skip dataloaders are supported.
    """

    def __init__(
        self,
        engine: Any,
        dataloader: TokenBatchCursor,
        optimizer: Any,
        *,
        microbatches: int,
        scheduler_spec: SchedulerSpec | None = None,
    ) -> None:
        if not isinstance(dataloader, TokenBatchCursor):
            raise ValueError("Stage recovery requires a TokenBatchCursor")
        if type(microbatches) is not int or microbatches < 1:
            raise ValueError("Stage microbatches must be a positive integer")
        self.engine, self.dataloader, self.optimizer = engine, dataloader, optimizer
        self.microbatches, self.scheduler_spec = microbatches, scheduler_spec
        self._scheduler = engine.scheduler

    def identity(self, optimizer: Any) -> dict[str, Any]:
        """Validate local binding and return common, rank-independent semantics."""
        from torch.optim.lr_scheduler import LambdaLR

        if optimizer is not self.optimizer:
            raise ValueError("Stage runtime is bound to a different optimizer")
        scheduler = self.engine.scheduler
        spec = self.scheduler_spec
        if (
            scheduler is not self._scheduler
            or (scheduler is None) != (spec is None)
            or (
                spec is not None
                and (
                    type(scheduler) is not LambdaLR
                    or scheduler.optimizer is not optimizer
                    or len(scheduler.lr_lambdas) != len(optimizer.param_groups)
                    or any(
                        type(function) is not partial
                        or function.func is not SchedulerSpec.factor
                        or function.args != (spec,)
                        or function.keywords
                        for function in scheduler.lr_lambdas
                    )
                    or _json(self.engine.scheduler_metadata)
                    != _json(spec.to_metadata())
                )
            )
        ):
            raise ValueError("Stage scheduler configuration/optimizer binding mismatch")
        data_identity = deepcopy(self.dataloader.identity)
        del data_identity["rank"]
        return {
            "kind": "stage_token_training",
            "version": 1,
            "microbatches": self.microbatches,
            "data": data_identity,
            "scheduler": spec.to_metadata() if spec is not None else None,
            "base_lrs": list(scheduler.base_lrs) if scheduler is not None else None,
        }

    def capture(self) -> dict[str, Any]:
        """Snapshot only committed progress; pending prefetch remains unconsumed."""
        state = self.engine.state
        return {
            "global_step": state.global_step,
            "consumed_micro_batches": state.consumed_micro_batches,
            "consumed_samples": state.consumed_samples,
            "data_state": self.engine.data_resume.capture(),
            "scheduler": deepcopy(self.engine.scheduler.state_dict())
            if self.engine.scheduler is not None
            else None,
            "rng": capture_rng_state(),
        }

    def validate(
        self, state: Any, *, step: int, optimizer_state: dict[str, Any]
    ) -> None:
        """Validate all progress and RNG before any model or optimizer mutation."""
        if not isinstance(state, dict) or set(state) != {
            "global_step",
            "consumed_micro_batches",
            "consumed_samples",
            "data_state",
            "scheduler",
            "rng",
        }:
            raise ValueError("Stage runtime state schema mismatch")
        expected = {
            "global_step": step,
            "consumed_micro_batches": step * self.microbatches,
            "consumed_samples": step * self.microbatches * self.dataloader.batch_size,
        }
        for key, value in expected.items():
            if type(state[key]) is not int or state[key] < 0 or state[key] != value:
                raise ValueError(f"Stage runtime counter mismatch: {key}")
        data = state["data_state"]
        if (
            not isinstance(data, dict)
            or set(data) != {"consumed_micro_batches", "resume_mode", "target", "state"}
            or type(data["consumed_micro_batches"]) is not int
            or data["consumed_micro_batches"] != expected["consumed_micro_batches"]
            or data["resume_mode"] != "state_dict"
            or data["target"] != "dataloader"
            or not isinstance(data["state"], dict)
            or set(data["state"]) != {"identity", "position"}
            or _json(data["state"]["identity"]) != _json(self.dataloader.identity)
            or data["state"]["position"] != expected["consumed_micro_batches"]
        ):
            raise ValueError("Stage consumed data cursor mismatch")
        self.engine.data_resume.validate_resume(self.dataloader, data)
        self._validate_scheduler(state["scheduler"], step, optimizer_state)
        _validate_host_rng(state["rng"])

    def _validate_scheduler(self, state: Any, step: int, optimizer_state: dict) -> None:
        scheduler, spec = self.engine.scheduler, self.scheduler_spec
        if scheduler is None:
            if state is not None:
                raise ValueError("Stage scheduler presence mismatch")
            return
        current = scheduler.state_dict()
        if not isinstance(state, dict) or set(state) != set(current):
            raise ValueError("Stage scheduler state schema mismatch")
        expected = deepcopy(current)
        expected.update(last_epoch=step, _step_count=step + 1)
        expected["_last_lr"] = [base * spec.factor(step) for base in scheduler.base_lrs]
        if _json(state) != _json(expected):
            raise ValueError("Stage scheduler progress/LR mismatch")
        groups = optimizer_state["param_groups"]
        if len(groups) != len(scheduler.base_lrs) or any(
            _json(group.get("lr")) != _json(lr)
            or _json(group.get("initial_lr")) != _json(base)
            for group, base, lr in zip(groups, scheduler.base_lrs, expected["_last_lr"])
        ):
            raise ValueError("Stage optimizer and scheduler LR mismatch")

    def restore(self, state: dict[str, Any]) -> None:
        """Apply prevalidated state, invalidate old iterators, restore host RNG last."""
        if self.engine.scheduler is not None:
            self.engine.scheduler.load_state_dict(deepcopy(state["scheduler"]))
        self.engine.data_resume.restore_manifest(
            SimpleNamespace(
                data_state=state["data_state"],
                consumed_samples=state["consumed_samples"],
            )
        )
        self.engine.state.global_step = state["global_step"]
        self.engine.state.last_metrics.clear()
        self.engine.state.metrics_history.clear()
        self.engine.state.metrics_history_dropped = 0
        restore_rng_state(state["rng"])
