# -*- coding: utf-8 -*-
# @Time : 2026/11/13 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Torch-free entrypoint checks and optional native training integration."""

import json

import pytest

from parascale.cli import main
from parascale.config import ParaScaleConfig
from parascale.runtime.train_runner import run_train_from_config


def test_quantization_configuration_roundtrip_without_torch_dependency():
    from parascale.quantization.base import QuantizationConfig

    config = ParaScaleConfig(
        quantization={
            "enabled": True,
            "bits": 4,
            "quantize_activations": True,
            "exclude_modules": ["lm_head", "decoder.output"],
        }
    )
    assert isinstance(config.quantization, QuantizationConfig)
    restored = ParaScaleConfig.from_dict(config.to_layered_dict())
    assert restored.quantization.to_dict() == config.quantization.to_dict()
    with pytest.raises(ValueError, match="bits"):
        ParaScaleConfig(quantization={"bits": 3})


@pytest.mark.parametrize(
    "field", ["enabled", "per_channel", "quantize_activations", "fuse_modules"]
)
def test_quantization_rejects_string_booleans(field):
    with pytest.raises(ValueError, match="boolean"):
        ParaScaleConfig(quantization={field: "false"})


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"quantization": {"enabled": True, "mode": "ptq"}}, "quantize"),
        (
            {"training_backend": "fsdp", "quantization": {"enabled": True}},
            "native",
        ),
        (
            {"precision": "bf16", "quantization": {"enabled": True}},
            "fp32",
        ),
        (
            {"adapter_only_checkpoint": True, "quantization": {"enabled": True}},
            "full checkpoints",
        ),
    ],
)
def test_train_rejects_unsupported_quantization_before_build(settings, message):
    with pytest.raises(ValueError, match=message):
        run_train_from_config({"parascale": settings})


def test_quantize_cli_requires_ptq_config(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"parascale": {}}), encoding="utf-8")
    assert (
        main(["quantize", "--config", str(config), "--output", str(tmp_path / "q.pt")])
        == 2
    )
    assert "enabled" in capsys.readouterr().err


def _tiny_config(tmp_path, mode):
    return {
        "parascale": {
            "training_backend": "native",
            "quantization": {"enabled": True, "mode": mode, "bits": 4},
        },
        "training": {
            "workload": "torch_tiny_mlp",
            "max_steps": 2,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    }


def test_native_qat_exports_and_resumes(tmp_path):
    torch = pytest.importorskip("torch")
    config = _tiny_config(tmp_path, "qat")
    result = run_train_from_config(config)
    assert result["global_step"] == 2
    assert result["quantization"]["mode"] == "qat"
    artifact = torch.load(result["quantization"]["checkpoint"], weights_only=True)
    assert artifact
    resumed = run_train_from_config(config, resume_step=2)
    assert resumed["global_step"] == 4
    config["parascale"]["quantization"]["bits"] = 8
    with pytest.raises(
        RuntimeError, match="quantization numerical configuration mismatch"
    ):
        run_train_from_config(config, resume_step=2)


@pytest.mark.parametrize("workload", ["torch_tiny_mlp", "clip_contrastive"])
def test_ptq_cli_exports_loadable_workload(tmp_path, workload):
    torch = pytest.importorskip("torch")
    from parascale.commands.quantization import run_ptq_from_config
    from parascale.quantization import load_quantized_model
    from parascale.workloads import build_training_components

    config = _tiny_config(tmp_path, "ptq")
    config["training"]["workload"] = workload
    output = tmp_path / "ptq.pt"
    result = run_ptq_from_config(config, str(output))
    assert result["execution"] == "dequantize_float_reference"
    template, _, batches, _ = build_training_components(config)
    loaded, _, _ = load_quantized_model(str(output), model=template)
    batch = next(iter(batches))
    with torch.no_grad():
        assert torch.isfinite(loaded(**batch)).all()
