# -*- coding: utf-8 -*-
# @Time : 2026/11/21 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""The accelerated inference option remains independent of training config."""

import subprocess
import sys

import pytest

from parascale.quantization import InferenceQuantizationConfig


@pytest.mark.parametrize("group_size", [True, 0, 16, 96, 128.0])
def test_reject_invalid_groups(group_size):
    with pytest.raises(ValueError, match="group_size"):
        InferenceQuantizationConfig(group_size=group_size)


@pytest.mark.parametrize("excluded", ["lm_head", [""], [None]])
def test_reject_invalid_exclusions(excluded):
    with pytest.raises(ValueError, match="exclude_modules"):
        InferenceQuantizationConfig(exclude_modules=excluded)


def test_configuration_does_not_import_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from parascale.quantization import "
            "InferenceQuantizationConfig; InferenceQuantizationConfig(); "
            "assert 'torch' not in sys.modules; assert 'torchao' not in sys.modules",
        ],
        check=True,
    )


def test_mutable_exclusion_input_is_copied():
    excluded = ["lm_head"]
    config = InferenceQuantizationConfig(exclude_modules=excluded)
    excluded.append("decoder")
    assert config.exclude_modules == ("lm_head",)


def test_int8_backend_configuration():
    assert InferenceQuantizationConfig(backend="gemlite_int8").backend == "gemlite_int8"
    with pytest.raises(ValueError, match="backend"):
        InferenceQuantizationConfig(backend="unknown")
