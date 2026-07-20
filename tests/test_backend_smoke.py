# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.optim as optim

from parascale.checkpoint import CheckpointManager
from parascale.cli import (
    run_benchmark_from_config,
    run_serve_from_config,
    run_train_from_config,
)
from parascale.config import ParaScaleConfig
from parascale.runtime.backends import create_runtime_training_backend
from parascale.runtime.training import TrainEngine


@pytest.mark.parametrize(
    ("optimizer_type", "expected_name"),
    [
        ("four_bit_adamw", "FourBitAdamW"),
        ("four_bit_sgd", "FourBitSGD"),
    ],
)
def test_configured_optimizer_factory_selects_four_bit_type(
    optimizer_type, expected_name
):
    from parascale.workloads import build_optimizer_for_model

    optimizer = build_optimizer_for_model(
        nn.Linear(4, 2),
        {"optimizer": {"type": optimizer_type, "lr": 0.01}},
    )

    assert optimizer.__class__.__name__ == expected_name
    assert optimizer._parascale_optimizer_metadata["type"] == optimizer_type


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, input_ids=None, **kwargs):
        x = input_ids if input_ids is not None else kwargs["x"]
        return self.fc(x.float())


def test_native_backend_smoke():
    model = TinyModel()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    config = ParaScaleConfig(training_backend="native")

    backend = create_runtime_training_backend(model, optimizer, config)
    wrapped_model, wrapped_optimizer = backend.setup()

    assert wrapped_model is model
    assert wrapped_optimizer is optimizer


def test_train_engine_native_fit_runs_model_optimizer_loss_smoke():
    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    config = ParaScaleConfig(training_backend="native")
    engine = TrainEngine(
        config=config,
        model_profile={
            "total_params": 10_000,
            "total_memory": 40_000,
            "num_layers": 1,
            "model_type": "mlp",
        },
        hardware_profile={
            "num_gpus": 1,
            "gpu_memory": 1_000_000_000,
            "available_memory": 900_000_000,
            "gpus_per_node": 1,
        },
    )

    def loss_fn(output, batch):
        return output.sum()

    state = engine.fit(
        [{"input_ids": torch.ones(2, 4)}, {"input_ids": torch.ones(2, 4)}],
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )

    assert state.global_step == 2
    assert "loss" in state.last_metrics


def test_train_engine_rebuilds_optimizer_after_backend_setup():
    model = TinyModel()
    initial_optimizer = optim.SGD(model.parameters(), lr=1e-2)
    config = ParaScaleConfig(training_backend="native")
    engine = TrainEngine(
        config=config,
        model_profile={
            "total_params": 10_000,
            "total_memory": 40_000,
            "num_layers": 1,
            "model_type": "mlp",
        },
        hardware_profile={
            "num_gpus": 1,
            "gpu_memory": 1_000_000_000,
            "available_memory": 900_000_000,
            "gpus_per_node": 1,
        },
    )
    rebuilt = {}

    def optimizer_builder(wrapped_model):
        rebuilt["model"] = wrapped_model
        return optim.AdamW(wrapped_model.parameters(), lr=1e-3)

    def loss_fn(output, batch):
        return output.sum()

    engine.fit(
        [{"input_ids": torch.ones(2, 4)}],
        model=model,
        optimizer=initial_optimizer,
        optimizer_builder=optimizer_builder,
        loss_fn=loss_fn,
    )

    assert rebuilt["model"] is engine.training_backend.model
    assert engine.training_backend.optimizer is not initial_optimizer
    assert isinstance(engine.training_backend.optimizer, optim.AdamW)


def test_train_engine_load_checkpoint_restores_backend_payload(tmp_path):
    config = ParaScaleConfig(training_backend="native")
    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    engine = TrainEngine(
        config=config,
        model_profile={
            "total_params": 10_000,
            "total_memory": 40_000,
            "num_layers": 1,
            "model_type": "mlp",
        },
        hardware_profile={
            "num_gpus": 1,
            "gpu_memory": 1_000_000_000,
            "available_memory": 900_000_000,
            "gpus_per_node": 1,
        },
    )

    def loss_fn(output, batch):
        return output.sum()

    engine.fit(
        [{"input_ids": torch.ones(2, 4)}],
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )
    manager = CheckpointManager(str(tmp_path))
    engine.save_checkpoint(manager)
    expected = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    restored_model = TinyModel()
    restored_optimizer = optim.SGD(restored_model.parameters(), lr=1e-2)
    restored = TrainEngine(
        config=config,
        model_profile={
            "total_params": 10_000,
            "total_memory": 40_000,
            "num_layers": 1,
            "model_type": "mlp",
        },
        hardware_profile={
            "num_gpus": 1,
            "gpu_memory": 1_000_000_000,
            "available_memory": 900_000_000,
            "gpus_per_node": 1,
        },
    )
    manifest = restored.load_checkpoint(
        manager, 1, model=restored_model, optimizer=restored_optimizer
    )

    assert manifest.metadata["backend_state_loaded"] is True
    assert restored.state.global_step == 1
    for name, value in restored_model.state_dict().items():
        assert torch.allclose(value, expected[name])


def test_train_engine_checkpoint_restores_scheduler_and_rng_state(tmp_path):
    config = ParaScaleConfig(training_backend="native")
    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    engine = TrainEngine(
        config=config,
        model_profile={
            "total_params": 10_000,
            "total_memory": 40_000,
            "num_layers": 1,
            "model_type": "mlp",
        },
        hardware_profile={
            "num_gpus": 1,
            "gpu_memory": 1_000_000_000,
            "available_memory": 900_000_000,
            "gpus_per_node": 1,
        },
    )

    def loss_fn(output, batch):
        return output.sum()

    engine.fit(
        [{"input_ids": torch.ones(2, 4)}],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
    )
    assert scheduler.last_epoch == 1

    torch.manual_seed(1234)
    manager = CheckpointManager(str(tmp_path))
    engine.save_checkpoint(manager, scheduler=scheduler)
    expected_next_random = torch.rand(3)

    restored_model = TinyModel()
    restored_optimizer = optim.SGD(restored_model.parameters(), lr=1e-2)
    restored_scheduler = optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=1, gamma=0.5
    )
    torch.manual_seed(9999)
    restored = TrainEngine(
        config=config,
        model_profile={
            "total_params": 10_000,
            "total_memory": 40_000,
            "num_layers": 1,
            "model_type": "mlp",
        },
        hardware_profile={
            "num_gpus": 1,
            "gpu_memory": 1_000_000_000,
            "available_memory": 900_000_000,
            "gpus_per_node": 1,
        },
    )
    manifest = restored.load_checkpoint(
        manager,
        1,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert manifest.metadata["scheduler_state_written"] is True
    assert restored_scheduler.last_epoch == 1
    assert torch.allclose(torch.rand(3), expected_next_random)


def test_cli_native_synthetic_train_checkpoint_resume_and_serve(tmp_path):
    config = {
        "parascale": {
            "training_backend": "native",
            "checkpoint_save_path": str(tmp_path),
            "checkpoint_save_interval": 1,
        },
        "model_profile": {
            "total_params": 10_000,
            "total_memory": 40_000,
            "num_layers": 1,
            "model_type": "mlp",
        },
        "hardware_profile": {
            "num_gpus": 1,
            "gpu_memory": 1_000_000_000,
            "available_memory": 900_000_000,
            "gpus_per_node": 1,
        },
        "training": {
            "workload": "synthetic_regression",
            "max_steps": 2,
            "checkpoint_dir": str(tmp_path),
            "checkpoint_interval": 1,
        },
    }

    trained = run_train_from_config(config)
    resumed = run_train_from_config(config, resume_step=trained["global_step"])
    served = run_serve_from_config(
        {"serving": {"mock": True, "requests": ["hello"]}},
        checkpoint=trained["checkpoint"],
    )

    assert trained["dry_run"] is False
    assert trained["global_step"] == 2
    assert (Path(trained["checkpoint"]).parent / "backend_state.pt").is_file()
    assert resumed["resumed_from"]["global_step"] == 2
    assert resumed["resumed_from"]["metadata"]["backend_state_loaded"] is True
    assert resumed["global_step"] == 4
    assert served["manifest"]["global_step"] == 2
    assert served["result"]["mode"] == "mock"


def test_cli_synthetic_benchmark_outputs_metrics(tmp_path):
    config = {
        "parascale": {
            "training_backend": "native",
            "checkpoint_save_path": str(tmp_path),
            "checkpoint_save_interval": 1,
        },
        "model_profile": {
            "total_params": 10_000,
            "total_memory": 40_000,
            "num_layers": 1,
            "model_type": "mlp",
        },
        "hardware_profile": {
            "num_gpus": 1,
            "gpu_memory": 1_000_000_000,
            "available_memory": 900_000_000,
            "gpus_per_node": 1,
        },
        "training": {
            "workload": "synthetic_regression",
            "benchmark_steps": 2,
            "checkpoint_dir": str(tmp_path),
            "checkpoint_interval": 1,
        },
    }

    payload = run_benchmark_from_config(config)

    assert payload["mode"] == "benchmark"
    assert payload["dry_run"] is False
    assert payload["metrics"]["steps_per_second"] > 0
    assert payload["metrics"]["step_time_ms"] > 0


def test_cli_benchmark_can_validate_resume(tmp_path):
    config = {
        "parascale": {
            "training_backend": "native",
            "checkpoint_save_path": str(tmp_path),
            "checkpoint_save_interval": 1,
        },
        "model_profile": {
            "total_params": 10_000,
            "total_memory": 40_000,
            "num_layers": 1,
            "model_type": "mlp",
        },
        "hardware_profile": {
            "num_gpus": 1,
            "gpu_memory": 1_000_000_000,
            "available_memory": 900_000_000,
            "gpus_per_node": 1,
        },
        "training": {
            "workload": "synthetic_regression",
            "benchmark_steps": 2,
            "checkpoint_dir": str(tmp_path),
            "checkpoint_interval": 1,
            "validate_resume": True,
            "resume_validation_steps": 1,
        },
    }

    payload = run_benchmark_from_config(config)

    assert payload["validation"]["checkpoint"]["ok"] is True
    assert payload["validation"]["resume"]["ok"] is True
    assert payload["validation"]["resume"]["backend_state_loaded"] is True
    assert payload["validation"]["resume"]["global_step"] == 3


def test_cli_native_tiny_torch_workload_runs_real_factory(tmp_path):
    config = {
        "parascale": {
            "training_backend": "native",
            "checkpoint_save_path": str(tmp_path),
            "checkpoint_save_interval": 1,
        },
        "model": {
            "type": "tiny_mlp",
            "input_dim": 4,
            "hidden_dim": 8,
            "output_dim": 2,
        },
        "data": {
            "type": "tensor_random",
            "batch_size": 2,
        },
        "optimizer": {
            "type": "adamw",
            "lr": 0.001,
        },
        "model_profile": {
            "total_params": 10_000,
            "total_memory": 40_000,
            "num_layers": 2,
            "model_type": "mlp",
        },
        "hardware_profile": {
            "num_gpus": 1,
            "gpu_memory": 1_000_000_000,
            "available_memory": 900_000_000,
            "gpus_per_node": 1,
        },
        "training": {
            "workload": "torch_tiny_mlp",
            "max_steps": 2,
            "checkpoint_dir": str(tmp_path),
            "checkpoint_interval": 1,
        },
    }

    payload = run_train_from_config(config)

    assert payload["dry_run"] is False
    assert payload["synthetic"] is False
    assert payload["runtime_status"] == "real_local"
    assert payload["capability_level"] == "local_native_real_torch"
    assert payload["global_step"] == 2
    assert payload["train_device"] in {"cpu", "cuda:0", "npu:0"}
    assert (Path(payload["checkpoint"]).parent / "backend_state.pt").is_file()


def test_cli_serve_can_run_tiny_torch_checkpoint_without_mock(tmp_path):
    config = {
        "parascale": {
            "training_backend": "native",
            "checkpoint_save_path": str(tmp_path),
            "checkpoint_save_interval": 1,
        },
        "model": {
            "type": "tiny_mlp",
            "input_dim": 4,
            "hidden_dim": 8,
            "output_dim": 2,
        },
        "data": {
            "type": "tensor_random",
            "batch_size": 2,
        },
        "optimizer": {
            "type": "adamw",
            "lr": 0.001,
        },
        "model_profile": {
            "total_params": 10_000,
            "total_memory": 40_000,
            "num_layers": 2,
            "model_type": "mlp",
        },
        "hardware_profile": {
            "num_gpus": 1,
            "gpu_memory": 1_000_000_000,
            "available_memory": 900_000_000,
            "gpus_per_node": 1,
        },
        "training": {
            "workload": "torch_tiny_mlp",
            "max_steps": 1,
            "checkpoint_dir": str(tmp_path),
            "checkpoint_interval": 1,
        },
    }
    trained = run_train_from_config(config)
    served = run_serve_from_config(
        {
            "model": config["model"],
            "serving": {
                "workload": "torch_tiny_mlp",
                "requests": [[1.0, 0.0, 0.0, 1.0]],
            },
        },
        checkpoint=trained["checkpoint"],
    )

    assert served["dry_run"] is False
    assert served["mock"] is False
    assert served["runtime_status"] == "real_local"
    assert served["result"]["mode"] == "model"
    assert len(served["result"]["outputs"]) == 1
    assert len(served["result"]["outputs"][0]) == 2


def test_fsdp_backend_import_boundary():
    if importlib.util.find_spec("torch.distributed.fsdp") is None:
        pytest.skip("FSDP is not available in this PyTorch build")

    config = ParaScaleConfig(training_backend="fsdp")
    model = TinyModel()
    backend = create_runtime_training_backend(
        model, optim.AdamW(model.parameters()), config
    )

    assert backend.name == "fsdp"


def test_deepspeed_backend_import_boundary():
    if importlib.util.find_spec("deepspeed") is None:
        pytest.skip("DeepSpeed is not installed")

    config = ParaScaleConfig(training_backend="deepspeed", zero_stage=2)
    model = TinyModel()
    backend = create_runtime_training_backend(
        model, optim.AdamW(model.parameters()), config
    )

    assert backend.name == "deepspeed"
