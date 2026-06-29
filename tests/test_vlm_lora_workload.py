# -*- coding: utf-8 -*-
# @Time : 2026/6/15 下午4:20
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import pytest

torch = pytest.importorskip("torch")

from parascale.cli import run_benchmark_from_config, run_train_from_config
from parascale.workloads.vlm_lora import VlmLoraSpec, build_vlm_lora_components


def test_vlm_lora_factory_runs_one_forward_step():
    spec = VlmLoraSpec(
        data_type="synthetic_image_text",
        image_size=32,
        patch_size=16,
        vocab_size=256,
        text_length=12,
        embed_dim=32,
        lora_rank=4,
        lora_alpha=8.0,
        batch_size=4,
        num_batches=1,
        num_samples=4,
        device="cpu",
    )

    model, optimizer, dataloader, loss_fn = build_vlm_lora_components(spec)
    batch = next(iter(dataloader))
    logits = model(**batch)
    loss = loss_fn(logits, batch)

    assert logits.shape == (4, 256)
    assert batch["num_pairs"] == 4
    assert batch["tokens"] > 0
    assert loss.item() > 0
    assert model.trainable_parameters < model.total_parameters
    assert optimizer is not None


def test_cli_vlm_lora_benchmark_outputs_multimodal_metrics(tmp_path):
    config = {
        "parascale": {
            "task_type": "multimodal",
            "model_family": "vlm_lora",
            "training_backend": "native",
            "batch_size": 2,
            "checkpoint_save_path": str(tmp_path),
            "checkpoint_save_interval": 999999,
        },
        "task": {
            "type": "multimodal",
            "workload": "vlm_lora",
            "modalities": ["image", "text"],
            "objective": "image_conditioned_text_adapter",
            "lora_rank": 4,
            "lora_alpha": 8.0,
        },
        "model": {
            "type": "tiny_vlm_lora",
            "image_size": 32,
            "patch_size": 16,
            "vocab_size": 256,
            "text_length": 12,
            "embed_dim": 32,
        },
        "data": {
            "type": "synthetic_image_text",
            "num_samples": 4,
            "batch_size": 2,
            "image_size": 32,
            "text_length": 12,
        },
        "lora": {"rank": 4, "alpha": 8.0, "dropout": 0.0},
        "optimizer": {"type": "adamw", "lr": 0.001},
        "model_profile": {
            "total_params": 100_000,
            "trainable_params": 10_000,
            "total_memory": 400_000,
            "num_layers": 4,
            "model_type": "tiny_vlm_lora",
        },
        "hardware_profile": {
            "num_gpus": 1,
            "gpus_per_node": 1,
            "gpu_memory": 1_000_000_000,
            "available_memory": 900_000_000,
        },
        "training": {
            "workload": "vlm_lora",
            "max_steps": 2,
            "benchmark_steps": 2,
            "warmup_steps": 0,
            "checkpoint_dir": str(tmp_path),
            "checkpoint_interval": 999999,
            "skip_final_checkpoint": True,
        },
    }

    payload = run_benchmark_from_config(config)

    assert payload["benchmark_type"] == "vlm_lora_train"
    assert payload["train"]["vlm_lora"] is True
    assert payload["metrics"]["end_to_end_image_text_pairs_per_second"] > 0
    assert payload["metrics"]["adapter_params"] > 0
    assert 0 < payload["metrics"]["trainable_ratio"] < 1


def test_vlm_lora_adapter_only_checkpoint_saves_adapter_state(tmp_path):
    config = {
        "parascale": {
            "task_type": "multimodal",
            "model_family": "vlm_lora",
            "training_backend": "native",
            "batch_size": 2,
            "checkpoint_save_path": str(tmp_path),
            "checkpoint_save_interval": 999999,
            "adapter_only_checkpoint": True,
        },
        "task": {
            "type": "multimodal",
            "workload": "vlm_lora",
            "modalities": ["image", "text"],
            "objective": "image_conditioned_text_adapter",
            "lora_rank": 4,
            "lora_alpha": 8.0,
        },
        "model": {
            "type": "tiny_vlm_lora",
            "image_size": 32,
            "patch_size": 16,
            "vocab_size": 256,
            "text_length": 12,
            "embed_dim": 32,
        },
        "data": {
            "type": "synthetic_image_text",
            "num_samples": 4,
            "batch_size": 2,
            "image_size": 32,
            "text_length": 12,
        },
        "lora": {"rank": 4, "alpha": 8.0, "dropout": 0.0},
        "optimizer": {"type": "adamw", "lr": 0.001},
        "model_profile": {
            "total_params": 100_000,
            "trainable_params": 10_000,
            "total_memory": 400_000,
            "num_layers": 4,
            "model_type": "tiny_vlm_lora",
        },
        "hardware_profile": {
            "num_gpus": 1,
            "gpus_per_node": 1,
            "gpu_memory": 1_000_000_000,
            "available_memory": 900_000_000,
        },
        "training": {
            "workload": "vlm_lora",
            "max_steps": 1,
            "checkpoint_dir": str(tmp_path),
            "checkpoint_interval": 999999,
            "skip_final_checkpoint": False,
        },
    }

    payload = run_train_from_config(config)
    checkpoint = payload["checkpoint"]
    state_path = tmp_path / "step-00000001" / "backend_state.pt"
    saved = torch.load(state_path, map_location="cpu", weights_only=True)
    backend_state = saved["backend_state"]

    assert checkpoint is not None
    assert backend_state["adapter_only_checkpoint"] is True
    assert backend_state["model_state_dict"] is None
    assert backend_state["adapter_state_dict"]
