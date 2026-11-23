# -*- coding: utf-8 -*-
# @Time : 2026/11/21 下午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Native optimizer conversion and collective checkpoint contracts."""

import copy
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from parascale.runtime.backends.native import NativeTrainingBackend
from parascale.runtime.training.checkpointing import CheckpointController


def test_zero_conversion_preserves_groups_state_and_updates(monkeypatch):
    parameters = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
    original = torch.optim.SGD(
        [
            {"params": [parameters[0]], "lr": 0.1, "weight_decay": 0.01},
            {"params": [parameters[1]], "lr": 0.001, "weight_decay": 0.0},
        ],
        lr=0.05,
        momentum=0.9,
    )
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    original.step()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        "parascale.optimizers.create_native_zero_redundancy_optimizer",
        lambda params, cls, stage, **kwargs: cls(params, **kwargs),
    )
    original._parascale_optimizer_metadata = {"kind": "test"}
    converted = NativeTrainingBackend(
        config=SimpleNamespace(zero_stage=1)
    ).setup_optimizer(original)
    assert [(g["lr"], g["weight_decay"]) for g in converted.param_groups] == [
        (0.1, 0.01),
        (0.001, 0.0),
    ]
    assert len(converted.state) == 2
    for parameter in parameters:
        torch.testing.assert_close(
            converted.state[parameter]["momentum_buffer"],
            original.state[parameter]["momentum_buffer"],
        )
    assert (
        converted._parascale_optimizer_metadata
        == original._parascale_optimizer_metadata
    )
    assert converted.param_groups[0] is not original.param_groups[0]


def test_checkpoint_preparation_runs_on_worker_before_rank0_save(tmp_path, monkeypatch):
    optimizer = SimpleNamespace(consolidate_state_dict=Mock())
    backend = NativeTrainingBackend(optimizer=optimizer)
    engine = SimpleNamespace(
        config=SimpleNamespace(training_backend="native"),
        training_backend=backend,
        state=SimpleNamespace(global_step=1),
        _distributed_rank=lambda: 1,
        _distributed_world_size=lambda: 2,
    )
    controller = CheckpointController(engine)
    monkeypatch.setattr(controller, "_collective_call", lambda fn, **kwargs: fn())
    monkeypatch.setattr(controller, "_save_runtime_state", lambda *args: False)
    result = controller.save(SimpleNamespace(payload_path=lambda *args: tmp_path))
    assert result["rank"] == 1
    optimizer.consolidate_state_dict.assert_called_once_with(to=0)


def test_ordinary_optimizer_checkpoint_preparation_is_noop():
    NativeTrainingBackend(optimizer=object()).prepare_checkpoint()


def _zero_worker(rank, root, optimizer_kind, warm_steps):
    from parascale.checkpoint import CheckpointManager
    from parascale.config import ParaScaleConfig
    from parascale.runtime.backends.native import NativeDdpTrainingBackend
    from parascale.runtime.training.engine import TrainEngine

    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    torch.distributed.init_process_group(
        "gloo",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(3)
        model = torch.nn.Linear(2, 2)
        reference = copy.deepcopy(model)

        def make_optimizer(module):
            optimizer_class = (
                torch.optim.SGD if optimizer_kind == "sgd" else torch.optim.AdamW
            )
            options = {"momentum": 0.9} if optimizer_kind == "sgd" else {}
            return optimizer_class(
                [
                    {"params": [module.weight], "lr": 0.1, "weight_decay": 0.01},
                    {"params": [module.bias], "lr": 0.001, "weight_decay": 0.0},
                ],
                lr=0.05,
                **options,
            )

        def step(module, opt):
            opt.zero_grad(set_to_none=True)
            module(torch.ones(3, 2)).square().mean().backward()
            opt.step()

        original_optimizer = make_optimizer(model)
        reference_optimizer = make_optimizer(reference)
        for _ in range(warm_steps):
            step(model, original_optimizer)
            step(reference, reference_optimizer)
        config = ParaScaleConfig(
            training_backend="native_ddp",
            data_parallel_size=2,
            zero_stage=1,
            ddp_comm_hook="none",
        )
        backend = NativeDdpTrainingBackend(model, original_optimizer, config)
        model, optimizer = backend.setup()

        for _ in range(3):
            step(model, optimizer)
            step(reference, reference_optimizer)
        engine = TrainEngine(config, training_backend=backend)
        engine.state.global_step = 3
        manager = CheckpointManager(str(Path(root) / "checkpoints"))
        engine.save_checkpoint(manager)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(100)
        optimizer.optim.state.clear()
        engine.load_checkpoint(manager, 3)
        step(model, optimizer)
        step(reference, reference_optimizer)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)
        assert [group["lr"] for group in optimizer.param_groups] == [0.1, 0.001]
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    "optimizer_kind,warm_steps", [("sgd", 0), ("sgd", 2), ("adamw", 2)]
)
def test_real_two_rank_zero_checkpoint_continuation(
    tmp_path, monkeypatch, optimizer_kind, warm_steps
):
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    ctx = torch.multiprocessing.get_context("spawn")
    workers = [
        ctx.Process(
            target=_zero_worker, args=(r, str(tmp_path), optimizer_kind, warm_steps)
        )
        for r in range(2)
    ]
    deadline = time.monotonic() + 100
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(max(0, deadline - time.monotonic()))
        assert [worker.exitcode for worker in workers] == [0, 0]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            if worker.pid is not None:
                worker.join(10)
