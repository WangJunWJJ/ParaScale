# -*- coding: utf-8 -*-
# @Time : 2026/6/25 下午3:56
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Example layout checks that do not require torch."""

from __future__ import annotations

import json
from pathlib import Path

from parascale.config import ParaScaleConfig

EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "examples"


def _load_example_config(path: str) -> dict:
    config_path = EXAMPLE_ROOT / path / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def test_examples_are_split_by_hardware_without_runtime_artifacts():
    expected = [
        "gpu/example_001_clip_tiny_native",
        "gpu/example_002_vision_synthetic_native",
        "ascend/example_001_tiny_ascend_native",
        "ascend/example_002_tiny_native_ddp_hccl",
    ]

    for relative in expected:
        example_dir = EXAMPLE_ROOT / relative
        assert (example_dir / "README.md").is_file()
        assert (example_dir / "config.json").is_file()
        assert not (example_dir / "runs").exists()
        assert not (example_dir / "checkpoints").exists()


def test_gpu_examples_use_unified_runtime_with_cuda_hints_only_in_config():
    clip_config = _load_example_config("gpu/example_001_clip_tiny_native")
    vision_config = _load_example_config("gpu/example_002_vision_synthetic_native")

    for config in [clip_config, vision_config]:
        parascale = ParaScaleConfig.from_dict(config["parascale"])
        assert parascale.training_backend == "native"
        assert parascale.device_prefetch is False
        assert config["runtime"]["accelerator"] == "cuda"
        assert config["runtime"]["communication_backend"] == "nccl"


def test_ascend_examples_use_unified_runtime_with_npu_hccl_hints():
    single_config = _load_example_config("ascend/example_001_tiny_ascend_native")
    ddp_config = _load_example_config("ascend/example_002_tiny_native_ddp_hccl")

    single_parascale = ParaScaleConfig.from_dict(single_config["parascale"])
    ddp_parascale = ParaScaleConfig.from_dict(ddp_config["parascale"])

    assert single_parascale.training_backend == "ascend_native"
    assert ddp_parascale.training_backend == "native_ddp"
    assert ddp_parascale.data_parallel_size == 2
    assert single_config["runtime"]["accelerator"] == "npu"
    assert ddp_config["runtime"]["accelerator"] == "npu"
    assert single_config["runtime"]["communication_backend"] == "hccl"
    assert ddp_config["runtime"]["communication_backend"] == "hccl"


def test_every_example_config_has_a_thin_run_script():
    config_paths = sorted(EXAMPLE_ROOT.glob("*/*/config.json"))

    assert len(config_paths) == 10
    for config_path in config_paths:
        example_dir = config_path.parent
        script_path = example_dir / "run.sh"
        assert script_path.is_file(), f"missing run script: {script_path}"

        script = script_path.read_text(encoding="utf-8")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_command = "infer" if "inference" in config else "train"

        assert script.startswith("#!/usr/bin/env bash\n")
        assert "set -euo pipefail" in script
        assert 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' in script
        assert f"-m parascale.cli {expected_command}" in script
        assert '"$SCRIPT_DIR/config.json"' in script
        assert '"$@"' in script
        assert "workload" not in script


def test_distributed_ascend_example_uses_torchrun():
    script_path = (
        EXAMPLE_ROOT
        / "ascend/example_002_tiny_native_ddp_hccl/run.sh"
    )
    script = script_path.read_text(encoding="utf-8")

    assert '"${TORCHRUN:-torchrun}"' in script
    assert '--nproc_per_node="${NPROC_PER_NODE:-2}"' in script
