# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import pytest

from parascale.workloads.specs.ground_dino import GroundDinoSpec
from parascale.workloads.specs.tiny import TinyTorchWorkloadSpec
from parascale.workloads.specs.yolo import YoloWorldSpec


@pytest.mark.parametrize("spec", [TinyTorchWorkloadSpec, YoloWorldSpec, GroundDinoSpec])
def test_workload_window_counts_microbatches(spec):
    config = {"training": {"max_steps": 2, "gradient_accumulation_steps": 3},
              "model": {"path": "unused-model"}, "data": {"data_dir": "unused-data"}}
    assert spec.from_config(config).num_batches == 6
