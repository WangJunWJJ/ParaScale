# -*- coding: utf-8 -*-
# @Time : 2026/11/19 下午4:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded offline CLIP candidates, not GPU or production certification."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from parascale.configuration import normalize_execution_config, validate_config_schema
from parascale.workloads.capability import describe_workload
from parascale.workloads.specs.clip import ClipContrastiveSpec

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "configs" / "production" / "clip_datacomp"
CASES = [
    (1, "native_1gpu.json"),
    (2, "native_ddp_2gpu.json"),
    (4, "native_ddp_4gpu.json"),
]


@pytest.mark.parametrize("world_size,filename", CASES)
def test_candidate_execution_contract(world_size, filename):
    path = CANDIDATES / filename
    assert path.is_file(), f"Missing production candidate: {filename}"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert validate_config_schema(raw)["ok"]
    normalized, config = normalize_execution_config(raw)
    assert describe_workload(normalized).workload == "clip_contrastive"
    spec = ClipContrastiveSpec.from_config(normalized)
    backend = "native" if world_size == 1 else "native_ddp"
    assert config.training_backend == backend
    assert config.data_parallel_size == world_size
    assert raw["hardware_profile"] == {
        "num_gpus": world_size,
        "gpus_per_node": world_size,
    }
    assert raw["distributed"] == {"nnodes": 1, "nproc_per_node": world_size}
    assert config.precision == "fp32"
    assert config.ddp_comm_hook == "none"
    assert not config.quantization.enabled
    assert config.batch_size == spec.batch_size == 8
    assert config.gradient_accumulation_steps == 1
    assert config.learning_rate == spec.lr == 1e-6
    assert raw["optimizer"] == {"type": "adamw", "lr": 1e-6}
    assert spec.model_type == "hf_clip"
    assert spec.pretrained_model_name_or_path == "/models/clip"
    assert spec.data_type == "datacomp_wds"
    assert spec.data_dir == "/datasets/datacomp"
    assert spec.streaming is True
    assert spec.num_workers == 0
    assert spec.num_batches == raw["training"]["max_steps"] == 1000
    assert raw["training"]["checkpoint_interval"] == 100
    run_dir = f"runs/production_candidate/{world_size}gpu"
    assert raw["runtime"] == {"device": "cuda", "run_dir": run_dir}
    assert raw["training"]["checkpoint_dir"] == f"{run_dir}/checkpoints"
    assert config.checkpoint_save_path == f"{run_dir}/checkpoints"
    assert config.checkpoint_save_interval == 100
    # An explicit allowlist catches silently ignored workload/runtime options too.
    allowed = {
        "parascale": {
            "task_type",
            "model_family",
            "target_scale",
            "training_backend",
            "precision",
            "data_parallel_size",
            "ddp_comm_hook",
            "checkpoint_save_path",
            "checkpoint_save_interval",
        },
        "task": {"type", "workload"},
        "model": {"type", "pretrained_model_name_or_path"},
        "data": {"type", "data_dir", "streaming", "num_workers", "text_length"},
        "optimizer": {"type", "lr"},
        "hardware_profile": {"num_gpus", "gpus_per_node"},
        "distributed": {"nnodes", "nproc_per_node"},
        "runtime": {"device", "run_dir"},
        "training": {
            "workload",
            "batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "checkpoint_interval",
            "checkpoint_dir",
            "seed",
        },
    }
    assert set(raw) == {"schema_version", *allowed}
    for section, keys in allowed.items():
        assert set(raw[section]) <= keys


def test_candidates_parse_without_importing_torch():
    code = """
import json
import sys
from pathlib import Path
from parascale.configuration import normalize_execution_config
from parascale.workloads.specs.clip import ClipContrastiveSpec
paths = sorted(Path(sys.argv[1]).glob('*.json'))
assert len(paths) == 3
for path in paths:
    data, _ = normalize_execution_config(json.loads(path.read_text(encoding='utf-8')))
    assert ClipContrastiveSpec.from_config(data).model_type == 'hf_clip'
assert 'torch' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code, str(CANDIDATES)], cwd=ROOT, check=True
    )


def test_readme_records_offline_candidate_limits():
    path = CANDIDATES / "README.md"
    assert path.is_file(), "Missing candidate operating instructions"
    readme = path.read_text(encoding="utf-8")
    for required in (
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "/models/clip",
        "/datasets/datacomp",
        "2.5.0a0+872d972e41.nv24.08",
        "4.51.3",
        "A6000",
        "parascale-detection:a6000-ultralytics83161",
        "sha256:043573ecd623390f83f2a7dadac4d1ae55c178fb07e3b6122dd8606e6b323493",
        "pip freeze",
        "docker image inspect",
        "rank-local negatives",
        "弱扩展",
        "global batch",
        "global-negative CLIP",
        "24h",
        "remote156",
        "超时",
        "待验收",
        "历史候选",
        "非生产认证",
        "非公开可部署 digest",
        "sha256sum",
        "--resume-step",
        "--nproc_per_node=2",
        "--nproc_per_node=4",
    ):
        assert required in readme
    for _, filename in CASES:
        assert filename in readme
