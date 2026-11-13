# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午4:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
import subprocess
import sys

import pytest


@pytest.mark.parametrize("temperature", [1e-300, 1e-20, 2e19, 1e300])
def test_temperature_requires_fp32_safe_square(temperature):
    from parascale.distillation import DistillationConfig

    with pytest.raises(ValueError, match="temperature.*FP32"):
        DistillationConfig(temperature=temperature)


def test_config_roundtrip_and_lazy_import():
    from parascale.distillation import DistillationConfig

    config = DistillationConfig()
    assert not config.enabled
    assert config.task == "classification"
    assert DistillationConfig.from_dict(config.to_dict()) == config
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from parascale.distillation import DistillationConfig; assert 'torch' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": 1},
        {"task": "unknown"},
        {"temperature": True},
        {"temperature": 0},
        {"temperature": float("inf")},
        {"alpha": True},
        {"alpha": -0.1},
        {"alpha": 1.1},
        {"alpha": float("nan")},
        {"ignore_index": True},
        {"input_keys": "input_ids"},
        {"input_keys": []},
        {"input_keys": [""]},
        {"input_keys": ["labels"]},
        {"input_keys": ["x", "x"]},
    ],
)
def test_config_rejects_invalid(kwargs):
    from parascale.distillation import DistillationConfig

    with pytest.raises((TypeError, ValueError)):
        DistillationConfig(**kwargs)
