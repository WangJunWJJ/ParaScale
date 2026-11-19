# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Weights-only-safe process RNG snapshots, including optional accelerators."""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Any


def capture_rng_state(torch: Any = None) -> dict[str, Any]:
    state = {"python": random.getstate()}
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        algorithm, keys, position, has_gauss, cached = np.random.get_state()
        state["numpy"] = (
            algorithm,
            keys.tolist(),
            int(position),
            int(has_gauss),
            float(cached),
        )
    if torch is not None:
        state["torch_cpu"] = torch.get_rng_state()
        for name in ("cuda", "npu"):
            device = getattr(torch, name, None)
            if device is not None and device.is_available():
                state[f"torch_{name}"] = device.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], torch: Any = None) -> None:
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        import numpy as np

        algorithm, keys, position, has_gauss, cached = state["numpy"]
        np.random.set_state(
            (algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached)
        )
    if torch is not None:
        if state.get("torch_cpu") is not None:
            torch.set_rng_state(state["torch_cpu"])
        for name in ("cuda", "npu"):
            if state.get(f"torch_{name}") is not None:
                device = getattr(torch, name, None)
                if device is None or not device.is_available():
                    raise RuntimeError(
                        f"Checkpoint requires unavailable {name} RNG state"
                    )
                device.set_rng_state_all(state[f"torch_{name}"])


@contextmanager
def preserve_rng_state():
    """Keep iterator setup/replay from advancing restored model RNG streams."""
    try:
        import torch
    except ImportError:
        torch = None
    state = capture_rng_state(torch)
    try:
        yield
    finally:
        restore_rng_state(state, torch)
