# -*- coding: utf-8 -*-
# @Time : 2026/7/6 上午12:34
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Data consumption tracking and restoration for training resume."""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, Iterator, Optional, Tuple


class DataResumeController:
    """Track consumed batches and restore a dataloader cursor after checkpoint load."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.active_dataloader: Any = None
        self._resume_applied = False

    def prepare_iterator(self, dataloader: Any) -> Iterator[Any]:
        self.active_dataloader = dataloader
        data_state = dict(getattr(self.engine.state, "data_state", {}) or {})
        if self._resume_applied or not data_state:
            return iter(dataloader)

        mode = str(data_state.get("resume_mode", "replay_skip"))
        if mode == "state_dict":
            target_name = str(data_state.get("target", "dataloader"))
            target = self._resolve_target(dataloader, target_name)
            load_state_dict = getattr(target, "load_state_dict", None)
            if not callable(load_state_dict):
                raise RuntimeError(
                    "Checkpoint requires stateful data resume, but the current "
                    f"{target_name} does not implement load_state_dict()."
                )
            load_state_dict(dict(data_state.get("state", {})))
            self._resume_applied = True
            return iter(dataloader)

        if mode != "replay_skip":
            raise ValueError(f"unsupported data resume mode: {mode}")
        iterator = iter(dataloader)
        consumed = max(0, int(data_state.get("consumed_micro_batches", 0) or 0))
        if consumed:
            warnings.warn(
                "Dataloader has no state protocol; replaying and skipping "
                f"{consumed} consumed micro-batches. Exact resume requires a "
                "repeatable deterministic data source.",
                RuntimeWarning,
                stacklevel=2,
            )
        for skipped in range(consumed):
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "Dataloader exhausted while replaying checkpoint data position: "
                    f"required={consumed}, restored={skipped}."
                ) from exc
        self._resume_applied = True
        return iterator

    def record_consumption(self, metrics: Dict[str, Any]) -> None:
        micro_batches = max(
            1,
            int(metrics.get("gradient_accumulation_steps", 1) or 1),
        )
        self.engine.state.consumed_micro_batches += micro_batches
        self.engine.state.consumed_samples += max(
            0,
            int(metrics.get("batch_size", 0) or 0),
        )

    def capture(self) -> Dict[str, Any]:
        consumed = int(self.engine.state.consumed_micro_batches)
        target = self._stateful_target(self.active_dataloader)
        if target is not None:
            target_name, target_object = target
            state = target_object.state_dict()
            try:
                json.dumps(state)
            except (TypeError, ValueError):
                warnings.warn(
                    f"{target_name}.state_dict() is not JSON serializable; "
                    "falling back to replay-skip data resume.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                return {
                    "consumed_micro_batches": consumed,
                    "resume_mode": "state_dict",
                    "state": state,
                    "target": target_name,
                }
        return {
            "consumed_micro_batches": consumed,
            "resume_mode": "replay_skip",
        }

    def restore_manifest(self, manifest: Any) -> None:
        self.engine.state.consumed_micro_batches = int(
            getattr(manifest, "data_state", {}).get("consumed_micro_batches", 0) or 0
        )
        self.engine.state.consumed_samples = int(
            getattr(manifest, "consumed_samples", 0) or 0
        )
        self.engine.state.data_state = dict(getattr(manifest, "data_state", {}) or {})
        self._resume_applied = False

    @staticmethod
    def _stateful_target(dataloader: Any) -> Optional[Tuple[str, Any]]:
        if dataloader is None:
            return None
        for name, target in (
            ("dataloader", dataloader),
            ("batch_sampler", getattr(dataloader, "batch_sampler", None)),
            ("sampler", getattr(dataloader, "sampler", None)),
        ):
            if callable(getattr(target, "state_dict", None)):
                return name, target
        return None

    @staticmethod
    def _resolve_target(dataloader: Any, target_name: str) -> Any:
        if target_name == "dataloader":
            return dataloader
        if target_name in {"batch_sampler", "sampler"}:
            return getattr(dataloader, target_name, None)
        raise ValueError(f"unsupported data resume target: {target_name}")


__all__ = ["DataResumeController"]
