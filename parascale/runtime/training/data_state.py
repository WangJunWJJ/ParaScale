# -*- coding: utf-8 -*-
# @Time : 2026/7/6 上午12:34
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Data consumption tracking and restoration for training resume."""

from __future__ import annotations

import copy
import json
import warnings
from collections import deque
from contextlib import nullcontext
from typing import Any, Dict, Iterator, Optional, Tuple

from .random_state import preserve_rng_state


class DataResumeController:
    """Track consumed batches and restore a dataloader cursor after checkpoint load."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.active_dataloader: Any = None
        self._resume_applied = False
        self._tracking = False
        self._pending_states: deque = deque()
        self._committed_state: Optional[Dict[str, Any]] = None
        self._generation = 0
        self._source_iterator = None

    def prepare_iterator(self, dataloader: Any) -> Iterator[Any]:
        self.close_iterator()
        continuing = dataloader is self.active_dataloader
        restoring = not self._resume_applied and bool(
            getattr(self.engine.state, "data_state", {})
        )
        # DataLoader iterator creation consumes a base seed even with zero workers.
        with preserve_rng_state() if continuing or restoring else nullcontext():
            return self._prepare_iterator(dataloader)

    def _prepare_iterator(self, dataloader: Any) -> Iterator[Any]:
        target = self._stateful_target(dataloader)
        if (
            target is not None
            and target[0] != "dataloader"
            and int(getattr(dataloader, "num_workers", 0)) > 0
        ):
            raise ValueError(
                "Stateful multi-worker resume requires a loader-level state_dict "
                "protocol that accounts for worker prefetch; sampler state alone "
                "does not describe consumed batches."
            )
        if dataloader is self.active_dataloader and self._pending_states:
            if self._committed_state is None:
                raise RuntimeError(
                    "Cannot restart prefetched data without a stateful cursor"
                )
            previous_target = self._resolve_target(
                dataloader, self._committed_state["target"]
            )
            restore = getattr(previous_target, "load_state_dict", None)
            if not callable(restore):
                raise RuntimeError(
                    "Restarting prefetched data requires load_state_dict()"
                )
            restore(copy.deepcopy(self._committed_state["state"]))
        self._generation += 1
        self._pending_states.clear()
        self._tracking = target is not None
        self.active_dataloader = dataloader
        data_state = dict(getattr(self.engine.state, "data_state", {}) or {})
        if self._resume_applied or not data_state:
            iterator = self._start_iterator(dataloader)
        else:
            iterator = self._restore_iterator(dataloader, data_state)
        return (
            self._track_iterator(
                iterator, target, self._generation, iterator is not dataloader
            )
            if self._tracking
            else iterator
        )

    def _start_iterator(self, dataloader: Any) -> Iterator[Any]:
        self._committed_state = self._snapshot(self._stateful_target(dataloader))
        self._source_iterator = iter(dataloader)
        return self._source_iterator

    def close_iterator(self) -> None:
        """Join any source lookahead before restart or RNG restoration."""
        iterator, self._source_iterator = self._source_iterator, None
        close = getattr(iterator, "close", None)
        if iterator is not self.active_dataloader and callable(close):
            close()

    def _track_iterator(
        self, iterator: Iterator[Any], target: Any, generation: int, owned: bool
    ) -> Iterator[Any]:
        # Only metadata is retained, bounded by accumulation plus device lookahead.
        try:
            while True:
                if generation != self._generation:
                    raise RuntimeError(
                        "Data iterator was superseded by a new fit or checkpoint load"
                    )
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
                self._pending_states.append(self._snapshot(target))
                yield batch
        finally:
            close = getattr(iterator, "close", None)
            if owned and callable(close):
                close()

    def _restore_iterator(
        self, dataloader: Any, data_state: Dict[str, Any]
    ) -> Iterator[Any]:

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
            return self._start_iterator(dataloader)

        if mode != "replay_skip":
            raise ValueError(f"unsupported data resume mode: {mode}")
        iterator = self._start_iterator(dataloader)
        consumed = max(0, int(data_state.get("consumed_micro_batches", 0) or 0))
        limit = int(getattr(self.engine.config, "max_resume_replay_batches", 10000))
        if consumed > limit:
            raise RuntimeError(
                f"Data resume requires {consumed} replay batches, exceeding limit {limit}; "
                "use a stateful dataloader or explicitly increase max_resume_replay_batches."
            )
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
        self._committed_state = self._snapshot(self._stateful_target(dataloader))
        return iterator

    def record_consumption(self, metrics: Dict[str, Any]) -> None:
        micro_batches = max(
            1,
            int(metrics.get("gradient_accumulation_steps", 1) or 1),
        )
        if self._tracking:
            if micro_batches > len(self._pending_states):
                raise RuntimeError("Data consumption exceeds issued batches")
            for _ in range(micro_batches):
                self._committed_state = self._pending_states.popleft()
        self.engine.state.consumed_micro_batches += micro_batches
        self.engine.state.consumed_samples += max(
            0,
            int(metrics.get("batch_size", 0) or 0),
        )

    def capture(self) -> Dict[str, Any]:
        consumed = int(self.engine.state.consumed_micro_batches)
        if self.active_dataloader is None and self.engine.state.data_state:
            return {
                **copy.deepcopy(self.engine.state.data_state),
                "consumed_micro_batches": consumed,
            }
        snapshot = (
            self._committed_state
            if self._tracking
            else self._snapshot(self._stateful_target(self.active_dataloader))
        )
        if snapshot is not None:
            return {"consumed_micro_batches": consumed, **copy.deepcopy(snapshot)}
        return {
            "consumed_micro_batches": consumed,
            "resume_mode": "replay_skip",
        }

    @staticmethod
    def _snapshot(target: Any) -> Optional[Dict[str, Any]]:
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
                    "resume_mode": "state_dict",
                    "state": copy.deepcopy(state),
                    "target": target_name,
                }
        return None

    def restore_manifest(self, manifest: Any) -> None:
        self.close_iterator()
        self.engine.state.consumed_micro_batches = int(
            getattr(manifest, "data_state", {}).get("consumed_micro_batches", 0) or 0
        )
        self.engine.state.consumed_samples = int(
            getattr(manifest, "consumed_samples", 0) or 0
        )
        self.engine.state.data_state = copy.deepcopy(
            dict(getattr(manifest, "data_state", {}) or {})
        )
        self._resume_applied = False
        self._tracking = False
        self._pending_states.clear()
        self._committed_state = None
        self.active_dataloader = None
        self._generation += 1

    def validate_resume(self, dataloader: Any, data_state: Dict[str, Any]) -> None:
        """Preflight stateful data before any model/optimizer/RNG restoration."""
        mode = data_state.get("resume_mode", "replay_skip")
        if mode == "replay_skip":
            return
        if mode != "state_dict":
            raise ValueError(f"unsupported data resume mode: {mode}")
        target = self._resolve_target(
            dataloader, data_state.get("target", "dataloader")
        )
        validate = getattr(target, "validate_state_dict", None)
        if not callable(validate) or not callable(
            getattr(target, "load_state_dict", None)
        ):
            raise ValueError(
                "Data preflight requires validate_state_dict() and load_state_dict()"
            )
        validate(copy.deepcopy(data_state.get("state", {})))

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
