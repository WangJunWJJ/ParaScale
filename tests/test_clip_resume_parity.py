# -*- coding: utf-8 -*-
# @Time : 2026/11/20 上午11:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Numerical diagnostics must distinguish equality from small state differences."""

import importlib
from pathlib import Path

import pytest


def test_comparison_measures_model_and_optimizer_differences(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "benchmarks" / "scripts"))
    compare = importlib.import_module("probe_clip_resume_parity").compare_state
    original = {"weights": torch.tensor([1.0, 2.0]), "state": [torch.tensor(3), None]}
    assert compare(original, original)["different_elements"] == 0
    changed = {"weights": torch.tensor([1.0, 2.5]), "state": [torch.tensor(3), None]}
    result = compare(original, changed)
    assert result["different_elements"] == 1
    assert result["max_abs"] == 0.5
    assert result["relative_l2"] == pytest.approx((0.25 / 14) ** 0.5)
    with pytest.raises(AssertionError, match="non-tensor"):
        compare({"step": 20}, {"step": 21})
