# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午4:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
import json
import subprocess
import sys
from fractions import Fraction

import pytest


@pytest.mark.parametrize("field", ["temperature", "alpha"])
def test_accepted_real_numbers_are_json_serializable(field):
    from parascale.distillation import DistillationConfig

    config = DistillationConfig(**{field: Fraction(1, 2)})
    payload = config.to_dict()
    assert type(payload[field]) is float
    assert DistillationConfig.from_dict(json.loads(json.dumps(payload))).to_dict() == payload


def test_numpy_real_numbers_are_json_serializable():
    from parascale.distillation import DistillationConfig

    numpy = pytest.importorskip("numpy")
    config = DistillationConfig(temperature=numpy.float32(2), alpha=numpy.float32(0.5))
    assert json.loads(json.dumps(config.to_dict()))["alpha"] == 0.5


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
