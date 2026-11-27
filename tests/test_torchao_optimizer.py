# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Optional optimizer dependency and execution guards."""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_missing_dependency_is_actionable(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    from parascale.optimizers import torchao_adapter as adapter

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(adapter, "version", missing)
    with pytest.raises(RuntimeError, match="optimizer-4bit"):
        adapter.load_optimizer_class()


def test_version_mismatch_is_rejected_before_import(monkeypatch):
    from parascale.optimizers import torchao_adapter as adapter

    monkeypatch.setattr(adapter, "version", lambda name: "0.0.0")
    with pytest.raises(RuntimeError, match="2.9.1"):
        adapter.load_optimizer_class()


def test_native_execution_rejects_cpu_candidate():
    from parascale.config import ParaScaleConfig
    from parascale.runtime.backends.native import NativeTrainingBackend

    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(8))])
    optimizer._parascale_optimizer_metadata = {"type": "torchao_adamw4bit"}
    with pytest.raises(ValueError, match="CUDA"):
        NativeTrainingBackend(optimizer=optimizer, config=ParaScaleConfig()).setup_optimizer(optimizer)


def test_model_export_dependency_preparation_checks_identity():
    from parascale.optimizers.torchao_adapter import prepare_checkpoint

    with pytest.raises(ValueError, match="identity"):
        prepare_checkpoint({"type": "torchao_adamw4bit"})
    prepare_checkpoint({"type": "adamw"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_official_optimizer_schedule_storage_and_restore():
    pytest.importorskip("torchao")
    from parascale.optimizers.factory import build_optimizer
    from parascale.optimizers.schedule import SchedulerSpec
    from parascale.optimizers.torchao_adapter import state_inventory, validate_execution

    model = torch.nn.ParameterList([
        torch.nn.Parameter(torch.linspace(-1, 1, size, device="cuda"))
        for size in (4096, 2048, 4097)
    ])
    config = {"optimizer": {"type": "torchao_adamw4bit", "lr": 1e-4}}
    optimizer = build_optimizer(model, config)
    validate_execution(optimizer, SimpleNamespace(training_backend="native", zero_stage=0, precision="bf16"))
    assert not optimizer.state
    scheduler = SchedulerSpec(type="cosine", total_steps=4).build(optimizer)
    for parameter in model:
        parameter.grad = torch.ones_like(parameter) * 0.01
    optimizer.step()
    scheduler.step()
    inventory = state_inventory(optimizer)
    assert inventory["state_bytes_by_device"]["cuda:0"] == 53640
    assert inventory["quantized_params"] == 4096
    assert inventory["dense_params"] == 6145
    json.dumps(inventory, allow_nan=False)
    import io

    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    buffer.seek(0)
    restored = build_optimizer(model, config)
    restored.load_state_dict(torch.load(buffer, weights_only=True, map_location="cpu"))
    assert state_inventory(restored) == inventory
    assert isinstance(restored.param_groups[0]["lr"], torch.Tensor)
