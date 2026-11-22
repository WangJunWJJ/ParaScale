# -*- coding: utf-8 -*-
# @Time : 2026/11/23 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Fixed-policy controls and predeclared paired throughput gates."""

from copy import deepcopy

import pytest


def test_bucket_comparison_changes_only_the_selected_policy():
    from tests.benchmarks.scripts.validate_clip_input_prefetch import comparison_config

    base = {"data": {"cpu_prefetch": False}, "parascale": {"precision": "fp32"}}
    original = deepcopy(base)
    for enabled in (False, True):
        result = comparison_config(base, "ddp-buckets", enabled)
        assert result["data"]["cpu_prefetch"] is True
        assert result["parascale"]["ddp_find_unused_parameters"] is enabled
        assert result["parascale"]["ddp_static_graph"] is False
        assert result["parascale"]["precision"] == "fp32"
    assert base == original


def test_prefetch_comparison_preserves_selected_bucket_policy():
    from tests.benchmarks.scripts.validate_clip_input_prefetch import comparison_config

    base = {"data": {}, "parascale": {"ddp_find_unused_parameters": True}}
    for enabled in (False, True):
        result = comparison_config(base, "cpu-prefetch", enabled)
        assert result["data"]["cpu_prefetch"] is enabled
        assert result["parascale"] == base["parascale"]


@pytest.mark.parametrize(
    "comparison,ratios,expected",
    [
        ("ddp-buckets", [0.94, 0.95, 0.96], True),
        ("ddp-buckets", [0.93, 0.94, 0.95], False),
        ("cpu-prefetch", [1.09, 1.1, 1.11], True),
        ("cpu-prefetch", [0.99, 1.1, 1.11], False),
    ],
)
def test_comparison_gates_are_fixed(comparison, ratios, expected):
    from tests.benchmarks.scripts.validate_clip_input_prefetch import comparison_passes

    assert comparison_passes(comparison, ratios) is expected


def test_comparison_rejects_unknown_policy_or_incomplete_pairs():
    from tests.benchmarks.scripts.validate_clip_input_prefetch import (
        comparison_config,
        comparison_passes,
    )

    with pytest.raises(ValueError):
        comparison_config({"data": {}, "parascale": {}}, "unknown", True)
    with pytest.raises(ValueError):
        comparison_passes("ddp-buckets", [1.0])
    for value in (float("nan"), float("inf"), 0.0):
        with pytest.raises(ValueError):
            comparison_passes("ddp-buckets", [1.0, 1.0, value])
