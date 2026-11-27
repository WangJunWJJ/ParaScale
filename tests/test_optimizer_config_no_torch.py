# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import pytest


def test_optimizer_spec_defaults_to_adamw_without_torch():
    from parascale.optimizers.spec import OptimizerSpec

    spec = OptimizerSpec.from_config({"optimizer": {"lr": 0.002}})

    assert spec.type == "adamw"
    assert spec.lr == 0.002


@pytest.mark.parametrize("value", [None, "adamw", [], False])
def test_optimizer_spec_rejects_non_mapping_section(value):
    from parascale.optimizers.spec import OptimizerSpec

    with pytest.raises(ValueError, match="optimizer must be a mapping"):
        OptimizerSpec.from_config({"optimizer": value})


@pytest.mark.parametrize("field", ["lr", "eps", "weight_decay", "momentum", "dampening"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, None])
def test_optimizer_spec_rejects_invalid_numeric_fields(field, value):
    from parascale.optimizers.spec import OptimizerSpec

    kind = "four_bit_sgd" if field in {"momentum", "dampening"} else "adamw"
    with pytest.raises(ValueError, match=f"optimizer.{field}"):
        OptimizerSpec.from_config({"optimizer": {"type": kind, field: value}})


@pytest.mark.parametrize("value", [0, 2.5, True, "128", None])
def test_optimizer_spec_rejects_coerced_group_size(value):
    from parascale.optimizers.spec import OptimizerSpec

    with pytest.raises(ValueError, match="optimizer.group_size"):
        OptimizerSpec.from_config({"optimizer": {"type": "four_bit_adamw", "group_size": value}})


@pytest.mark.parametrize("field", ["nesterov", "compensate_quant_error"])
@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_optimizer_spec_requires_real_booleans(field, value):
    from parascale.optimizers.spec import OptimizerSpec

    with pytest.raises(ValueError, match=f"optimizer.{field}"):
        OptimizerSpec.from_config({"optimizer": {"type": "four_bit_sgd", field: value}})


@pytest.mark.parametrize("value", [None, 0.9, "09", [True, 0.99], [0.9, "bad"]])
def test_optimizer_spec_reports_invalid_betas(value):
    from parascale.optimizers.spec import OptimizerSpec

    with pytest.raises(ValueError, match="optimizer.betas"):
        OptimizerSpec.from_config({"optimizer": {"betas": value}})


@pytest.mark.parametrize("settings", [{"momentum": 0}, {"dampening": 0.1}])
def test_optimizer_spec_checks_nesterov_before_construction(settings):
    from parascale.optimizers.spec import OptimizerSpec

    with pytest.raises(ValueError, match="optimizer.nesterov"):
        OptimizerSpec.from_config({"optimizer": {"type": "four_bit_sgd", "nesterov": True, **settings}})


def test_optimizer_spec_preserves_zero_and_scientific_values():
    from parascale.optimizers.spec import OptimizerSpec

    spec = OptimizerSpec.from_config({"optimizer": {"lr": "1e-4", "eps": 0, "weight_decay": 0}})
    assert (spec.lr, spec.eps, spec.weight_decay) == (1e-4, 0, 0)


@pytest.mark.parametrize("optimizer_type", ["four_bit_adamw", "four_bit_sgd"])

def test_optimizer_spec_accepts_configured_four_bit_types(optimizer_type):
    from parascale.optimizers.spec import OptimizerSpec

    spec = OptimizerSpec.from_config(
        {
            "optimizer": {
                "type": optimizer_type,
                "lr": 0.01,
                "group_size": 64,
                "compensate_quant_error": True,
                "error_compensation_dtype": "fp32",
            }
        }
    )

    assert spec.type == optimizer_type
    assert spec.group_size == 64
    assert spec.to_metadata()["state_schema_version"] == 1

def test_optimizer_spec_rejects_fields_for_another_optimizer_type():
    from parascale.optimizers.spec import OptimizerSpec

    with pytest.raises(ValueError, match="momentum.*four_bit_adamw"):
        OptimizerSpec.from_config(
            {"optimizer": {"type": "four_bit_adamw", "momentum": 0.9}}
        )

def test_optimizer_spec_accepts_block_scaled_fp16_residuals():
    from parascale.optimizers.spec import OptimizerSpec

    spec = OptimizerSpec.from_config(
        {
            "optimizer": {
                "type": "four_bit_adamw",
                "error_compensation_dtype": "fp16",
                "error_compensation_mode": "block_scaled",
            }
        }
    )

    assert spec.error_compensation_mode == "block_scaled"
    assert spec.to_metadata()["error_compensation_mode"] == "block_scaled"


@pytest.mark.parametrize("backend", ["fsdp", "deepspeed"])

def test_four_bit_optimizer_spec_rejects_sharded_backends(backend):
    from parascale.optimizers.spec import OptimizerSpec

    spec = OptimizerSpec.from_config(
        {"optimizer": {"type": "four_bit_adamw"}}
    )

    with pytest.raises(ValueError, match=backend):
        spec.validate_backend(backend, zero_stage=0)

def test_four_bit_optimizer_spec_rejects_native_zero_stage_one():
    from parascale.optimizers.spec import OptimizerSpec

    spec = OptimizerSpec.from_config(
        {"optimizer": {"type": "four_bit_sgd"}}
    )

    with pytest.raises(ValueError, match="zero_stage=1"):
        spec.validate_backend("native_ddp", zero_stage=1)

def test_optimizer_parameter_selection_filters_frozen_params_without_torch():
    from parascale.workloads.registry import trainable_parameter_stats

    class Param:
        def __init__(self, count, requires_grad):
            self.requires_grad = requires_grad
            self.count = count

        def numel(self):
            return self.count

    class Model:
        def parameters(self):
            return [
                Param(10, False),
                Param(5, True),
                Param(15, True),
            ]

    selected, stats = trainable_parameter_stats(Model())

    assert len(selected) == 2
    assert stats["trainable_params"] == 20
    assert stats["total_params"] == 30
    assert stats["trainable_ratio"] == 20 / 30

def test_optimizer_parameter_selection_rejects_all_frozen_without_torch():
    from parascale.workloads.registry import trainable_parameter_stats

    class Param:
        requires_grad = False

        def numel(self):
            return 10

    class Model:
        def parameters(self):
            return [Param()]

    try:
        trainable_parameter_stats(Model())
    except RuntimeError as exc:
        assert "no trainable parameters" in str(exc)
    else:
        raise AssertionError("optimizer construction must reject all-frozen models")

def test_vlm_lora_uses_shared_trainable_parameter_selection_without_torch():
    from pathlib import Path

    source = Path("parascale/workloads/vlm_lora.py").read_text(encoding="utf-8")

    assert "from .optimizer import build_adamw_optimizer_for_model" in source
    assert "trainable_parameters = [" not in source
    assert "optim.AdamW(trainable_parameters" not in source
