# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Dependency-free admission and checkpoint identity for optional optimizer."""

from types import SimpleNamespace

import pytest

from parascale.optimizers.spec import OptimizerSpec
from parascale.runtime.training.checkpointing import CheckpointController


def candidate():
    return OptimizerSpec.from_config({"optimizer": {"type": "torchao_adamw4bit"}})


def test_torchao_spec_has_explicit_fixed_algorithm_identity():
    spec = candidate()
    assert spec.is_four_bit
    metadata = spec.to_metadata()
    assert metadata["implementation"] == "torchao.optim.AdamW4bit"
    assert metadata["torch_version"] == "2.9.1"
    assert metadata["torchao_version"] == "0.15.0"
    assert metadata["group_size"] == 128
    assert metadata["amsgrad"] is False
    assert metadata["bf16_stochastic_round"] is False
    assert metadata["compensate_quant_error"] is None


@pytest.mark.parametrize("backend", ["native", "native_ddp"])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_torchao_spec_admits_qualified_execution(backend, precision):
    candidate().validate_backend(backend, precision=precision)


@pytest.mark.parametrize("backend", ["native_parallel", "fsdp", "deepspeed"])
def test_torchao_spec_rejects_unqualified_backends(backend):
    with pytest.raises(ValueError, match=backend):
        candidate().validate_backend(backend)


def test_torchao_spec_rejects_zero_and_fp16():
    with pytest.raises(ValueError, match="zero_stage"):
        candidate().validate_backend("native_ddp", zero_stage=1)
    with pytest.raises(ValueError, match="precision"):
        candidate().validate_backend("native", precision="fp16")


@pytest.mark.parametrize("field", ["group_size", "compensate_quant_error", "error_compensation_dtype"])
def test_torchao_spec_rejects_legacy_quantization_options(field):
    with pytest.raises(ValueError, match=field):
        OptimizerSpec.from_config({"optimizer": {"type": "torchao_adamw4bit", field: 128}})


@pytest.mark.parametrize("key", ["implementation", "torch_version", "torchao_version", "amsgrad", "bf16_stochastic_round"])
def test_torchao_checkpoint_requires_exact_algorithm_identity(key):
    metadata = candidate().to_metadata()
    saved = dict(metadata)
    saved.pop(key)
    optimizer = SimpleNamespace(_parascale_optimizer_metadata=metadata)
    manifest = SimpleNamespace(metadata={"optimizer": saved})
    with pytest.raises(ValueError, match="optimizer metadata mismatch"):
        CheckpointController._validate_optimizer_metadata(manifest, optimizer)


def test_torchao_checkpoint_rejects_missing_identity():
    optimizer = SimpleNamespace(_parascale_optimizer_metadata=candidate().to_metadata())
    with pytest.raises(ValueError, match="optimizer metadata"):
        CheckpointController._validate_optimizer_metadata(SimpleNamespace(metadata={}), optimizer)


def test_torchao_checkpoint_accepts_matching_identity():
    metadata = candidate().to_metadata()
    CheckpointController._validate_optimizer_metadata(
        SimpleNamespace(metadata={"optimizer": dict(metadata)}),
        SimpleNamespace(_parascale_optimizer_metadata=metadata),
    )
