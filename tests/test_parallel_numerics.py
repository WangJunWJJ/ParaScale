# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Negative controls for distributed training numerical qualification."""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "kind,value",
    [("gradient", 0.1), ("exp_avg", 0.01), ("exp_avg_sq", 1e-8), ("update", 1e-5)],
)
def test_zeroed_training_state_cannot_pass_on_absolute_tolerance(kind, value):
    from tests.benchmarks.tools.parallel_numerics import compare_training_tensor

    expected = torch.full((8, 8), value)
    assert not compare_training_tensor(torch.zeros_like(expected), expected, kind)[
        "passed"
    ]


def test_exact_zero_and_exact_state_are_accepted():
    from tests.benchmarks.tools.parallel_numerics import compare_training_tensor

    for value in (0.0, 1e-8, 1.0):
        tensor = torch.full((4, 3), value)
        assert compare_training_tensor(tensor, tensor, "exp_avg_sq")["passed"]


def test_step_counter_must_match_exactly():
    from tests.benchmarks.tools.parallel_numerics import compare_training_tensor

    assert not compare_training_tensor(torch.tensor(4.0), torch.tensor(3.0), "step")[
        "passed"
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_comparisons_never_pass(value):
    from tests.benchmarks.tools.parallel_numerics import compare_training_tensor

    with pytest.raises(AssertionError, match="nonfinite"):
        compare_training_tensor(torch.tensor(value), torch.tensor(0.0), "gradient")


def test_dp_normalization_error_is_rejected():
    from tests.benchmarks.tools.parallel_numerics import compare_training_tensor

    expected = torch.randn(8, 8)
    assert not compare_training_tensor(expected / 2, expected, "gradient")["passed"]
