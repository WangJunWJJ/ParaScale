# -*- coding: utf-8 -*-
# @Time : 2026/6/9 下午8:16
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

from parascale import ParaScaleConfig, StrategyPlan, build_parallel_plan


def test_parallel_plan_is_declarative_and_serializable():
    config = ParaScaleConfig(
        training_backend="fsdp",
        data_parallel_size=4,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )
    strategy = StrategyPlan(
        backend="fsdp", dp_size=4, tp_size=2, pp_size=1, zero_stage=0
    )

    plan = build_parallel_plan(config, strategy)
    payload = plan.to_dict()

    assert payload["backend"] == "fsdp"
    assert payload["dimensions"]["data"]["size"] == 4
    assert payload["dimensions"]["tensor"]["size"] == 2
    assert payload["sharding"] == "fsdp"
    assert "DataParallel" not in str(payload)
