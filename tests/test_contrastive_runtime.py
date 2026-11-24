# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午11:36
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Real CPU/Gloo TrainEngine updates against independent global-batch losses."""

import json
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest

from parascale.config import ParaScaleConfig
from parascale.runtime.backends import create_runtime_training_backend
from parascale.runtime.training import TrainEngine
from parascale.workloads.contrastive_loss import ContrastiveLoss

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

from tests.test_contrastive_objectives import _PairModel, _reference  # noqa: E402


class _TrackedLoss(ContrastiveLoss):
    def __init__(self, family):
        super().__init__(family)
        self.bindings = []
        self.local_losses = []

    def bind_data_parallel(self, group):
        self.bindings.append(group)
        super().bind_data_parallel(group)

    def __call__(self, output, batch):
        loss = super().__call__(output, batch)
        self.local_losses.append(loss.detach().clone())
        return loss


def _oracle(initial, family, global_batches, accumulation):
    reference = _PairModel(family).double()
    reference.load_state_dict(initial)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.07, momentum=0.9)
    losses, gradients, updates = [], [], []
    for start in range(0, len(global_batches), accumulation):
        window = global_batches[start : start + accumulation]
        optimizer.zero_grad()
        loss = torch.stack(
            [_reference(reference(images, texts), family) for images, texts in window]
        ).mean()
        if accumulation > 1:
            concatenated = _reference(
                reference(
                    torch.cat([images for images, _ in window]),
                    torch.cat([texts for _, texts in window]),
                ),
                family,
            )
            assert not torch.isclose(loss, concatenated, atol=1e-6, rtol=1e-6)
        loss.backward()
        losses.append(loss.detach().clone())
        gradients.append(
            {
                name: value.grad.detach().clone()
                for name, value in reference.named_parameters()
            }
        )
        optimizer.step()
        updates.append(
            {
                name: value.detach().clone()
                for name, value in reference.named_parameters()
            }
        )
    return losses, gradients, updates


def _check_binding_scope(engine, family):
    # Use the real initialized WORLD group, never monkeypatched process globals.
    assert dist.is_initialized() and dist.get_world_size() == 2
    for axis in (
        "model_parallel_size",
        "tensor_parallel_size",
        "pipeline_parallel_size",
    ):
        objective = _TrackedLoss(family)
        original = getattr(engine.config, axis)
        setattr(engine.config, axis, 2)
        try:
            with pytest.raises(ValueError, match="unsupported|parallel"):
                engine._bind_loss_data_parallel(objective)
            assert objective.bindings == []
        finally:
            setattr(engine.config, axis, original)

    local = TrainEngine(ParaScaleConfig(training_backend="native"))
    local.training_backend = create_runtime_training_backend(config=local.config)
    objective = _TrackedLoss(family)
    local._bind_loss_data_parallel(objective)
    assert objective.bindings == [None]
    assert objective.group is None


def _runtime_worker(rank, rendezvous, result_dir, family, accumulation):
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "2",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "OMP_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    torch.set_num_threads(1)
    assert not dist.is_initialized()
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=35),
    )
    try:
        assert dist.get_backend() == "gloo"
        sizes, steps = (1, 3), 2
        torch.manual_seed(17)
        model = _PairModel(family).double()
        initial = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        generator = torch.Generator().manual_seed(73)
        global_batches = [
            tuple(
                torch.randn(2, sum(sizes), 3, dtype=torch.float64, generator=generator)
            )
            for _ in range(steps * accumulation)
        ]
        expected_losses, expected_grads, expected_updates = _oracle(
            initial, family, global_batches, accumulation
        )
        start, stop = sum(sizes[:rank]), sum(sizes[: rank + 1])
        batches = [
            {
                "images": images[start:stop].clone(),
                "texts": texts[start:stop].clone(),
                "num_pairs": sizes[rank],
            }
            for images, texts in global_batches
        ]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.07, momentum=0.9)
        engine = TrainEngine(
            ParaScaleConfig(
                training_backend="native_ddp",
                data_parallel_size=2,
                gradient_accumulation_steps=accumulation,
                precision="fp32",
                ddp_comm_hook="none",
            )
        )
        objective = _TrackedLoss(family)
        observed_grads, observed_updates = [], []
        original_step = engine.step

        def capture_step(optimizer=None):
            assert all(value.grad is not None for value in model.parameters())
            observed_grads.append(
                {
                    name: value.grad.detach().clone()
                    for name, value in model.named_parameters()
                }
            )
            # Backend.step performs optimizer.step and zero_grad. Capture first.
            updated = original_step(optimizer)
            observed_updates.append(
                {
                    name: value.detach().clone()
                    for name, value in model.named_parameters()
                }
            )
            return updated

        engine.step = capture_step
        state = engine.fit(
            batches,
            max_steps=steps,
            model=model,
            optimizer=optimizer,
            loss_fn=objective,
        )
        assert isinstance(
            engine.training_backend.model, torch.nn.parallel.DistributedDataParallel
        )
        assert all(value.device.type == "cpu" for value in model.parameters())
        assert objective.bindings == [dist.group.WORLD]
        assert objective.group is dist.group.WORLD
        assert state.global_step == steps
        assert state.consumed_micro_batches == steps * accumulation
        assert len(observed_grads) == len(observed_updates) == steps
        assert len(objective.local_losses) == steps * accumulation
        for step in range(steps):
            window = objective.local_losses[
                step * accumulation : (step + 1) * accumulation
            ]
            global_mean = torch.stack(window).mean() * sizes[rank]
            dist.all_reduce(global_mean)
            global_mean /= sum(sizes)
            torch.testing.assert_close(
                global_mean, expected_losses[step], atol=1e-9, rtol=1e-7
            )
            for name in expected_grads[step]:
                # Runtime denominator collectives use FP32 even for this double oracle.
                torch.testing.assert_close(
                    observed_grads[step][name],
                    expected_grads[step][name],
                    atol=1e-7,
                    rtol=1e-6,
                )
                torch.testing.assert_close(
                    observed_updates[step][name],
                    expected_updates[step][name],
                    atol=1e-7,
                    rtol=1e-6,
                )
        assert any(
            not torch.equal(value, initial[name])
            for name, value in model.state_dict().items()
        )
        _check_binding_scope(engine, family)
        (Path(result_dir) / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "family": family,
                    "accumulation": accumulation,
                    "steps": state.global_step,
                    "local_batch": sizes[rank],
                    "gradients_and_updates_match": True,
                    "binding_scope_checked": True,
                }
            ),
            encoding="utf-8",
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("family,accumulation", [("clip", 1), ("siglip", 2)])
def test_fit_contrastive_unequal_batches_matches_global_oracle(
    tmp_path, monkeypatch, family, accumulation
):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    context = torch.multiprocessing.spawn(
        _runtime_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path), family, accumulation),
        nprocs=2,
        join=False,
    )
    try:
        deadline = time.monotonic() + 90
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("TrainEngine Gloo workers exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        assert all(not process.is_alive() for process in context.processes)
    for rank, size in enumerate((1, 3)):
        report = json.loads(
            (tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8")
        )
        assert report == {
            "rank": rank,
            "family": family,
            "accumulation": accumulation,
            "steps": 2,
            "local_batch": size,
            "gradients_and_updates_match": True,
            "binding_scope_checked": True,
        }
