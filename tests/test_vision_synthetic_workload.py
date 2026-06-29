# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from parascale.cli import run_benchmark_from_config, run_train_from_config
from parascale.workloads.vision import (
    VisionSyntheticSpec,
    build_vision_synthetic_components,
)


def _vision_config(tmp_path):
    return {
        "parascale": {
            "task_type": "vision",
            "model_family": "vit",
            "training_backend": "native",
            "batch_size": 4,
            "max_patch_tokens_per_batch": 64,
            "checkpoint_save_path": str(tmp_path),
            "checkpoint_save_interval": 1,
        },
        "model": {
            "type": "tiny_patch_classifier",
            "image_size": 32,
            "channels": 3,
            "patch_size": 16,
            "hidden_dim": 16,
            "num_classes": 5,
        },
        "data": {
            "type": "vision_synthetic",
            "num_samples": 12,
            "batch_size": 4,
            "image_size": 32,
        },
        "optimizer": {
            "type": "adamw",
            "lr": 0.01,
        },
        "model_profile": {
            "total_params": 50_000,
            "total_memory": 200_000,
            "num_layers": 3,
            "model_type": "vision_transformer",
        },
        "hardware_profile": {
            "num_gpus": 1,
            "gpus_per_node": 1,
            "gpu_memory": 1_000_000_000,
            "available_memory": 900_000_000,
        },
        "training": {
            "workload": "vision_synthetic",
            "max_steps": 2,
            "checkpoint_dir": str(tmp_path),
            "checkpoint_interval": 1,
        },
    }


def test_vision_synthetic_factory_builds_patch_batches():
    spec = VisionSyntheticSpec(
        image_size=32,
        patch_size=16,
        hidden_dim=16,
        num_classes=5,
        num_samples=8,
        batch_size=4,
        num_batches=1,
        max_patch_tokens_per_batch=16,
        device="cpu",
    )

    model, optimizer, dataloader, loss_fn = build_vision_synthetic_components(spec)
    batch = next(iter(dataloader))
    output = model(pixel_values=batch["pixel_values"])
    loss = loss_fn(output, batch)

    assert output.shape == (4, 5)
    assert batch["num_images"] == 4
    assert batch["patch_tokens"] == 16
    assert loss.item() > 0
    assert optimizer is not None


def test_cli_vision_synthetic_train_outputs_throughput_metrics(tmp_path):
    payload = run_train_from_config(_vision_config(tmp_path))
    metrics = payload["last_metrics"]

    assert payload["vision_synthetic"] is True
    assert payload["capability_level"] == "local_native_vision_synthetic"
    assert payload["global_step"] == 2
    assert metrics["images"] == 4
    assert metrics["patch_tokens"] == 16
    assert metrics["images_per_second"] > 0
    assert metrics["patch_tokens_per_second"] > 0
    assert (Path(payload["checkpoint"]).parent / "backend_state.pt").is_file()


def test_cli_vision_synthetic_benchmark_preserves_train_metrics(tmp_path):
    config = _vision_config(tmp_path)
    config["training"]["benchmark_steps"] = 2

    payload = run_benchmark_from_config(config)

    assert payload["mode"] == "benchmark"
    assert payload["train"]["vision_synthetic"] is True
    assert payload["train"]["last_metrics"]["patch_tokens_per_second"] > 0
