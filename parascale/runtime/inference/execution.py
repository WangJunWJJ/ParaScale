# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午12:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Shared inference execution policy without a mandatory torch import."""

from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator


@contextmanager
def inference_execution(model: Any) -> Iterator[None]:
    """Keep models in eval mode and disable autograd when torch is loaded."""
    eval_fn = getattr(model, "eval", None)
    if callable(eval_fn):
        eval_fn()
    # A constructed torch model already loaded torch; duck models need not do so.
    torch = sys.modules.get("torch")
    with torch.no_grad() if torch is not None else nullcontext():
        yield
