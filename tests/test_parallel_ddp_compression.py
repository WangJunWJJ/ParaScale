# -*- coding: utf-8 -*-
# @Time : 2026/11/25 下午10:58
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Pinned CUDA parity and checkpoint contracts for stage BF16 communication."""

import copy
import math
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
dist = torch.distributed


def _worker(rank, world, microbatches, folder):
    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
        bf16_compress_hook,
    )

    from parascale.parallel.distributed import DistributedParallelConfig
    from parascale.parallel.execution import DistributedTrainer
    from tests.test_stage_dcp_checkpoint import RuntimeState, _equal

    root = Path(folder)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=world,
        device_id=device,
        timeout=timedelta(seconds=90),
    )
    try:
        torch.manual_seed(37)
        config = DistributedParallelConfig(dp_size=world, microbatches=microbatches)
        example = torch.zeros(2, 8, device=device)
        loss_fn = torch.nn.MSELoss()
        context = dict(
            model="mlp", loss="mse", tokenizer="none", data="fixed", precision="fp32"
        )

        def model():
            return torch.nn.Sequential(
                torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4)
            ).to(device)

        def trainer(module, policy="bf16_compress"):
            return DistributedTrainer(
                module,
                config,
                example,
                loss_fn,
                checkpoint_context=context,
                ddp_comm_hook=policy,
            )

        def optimizer(module):
            return torch.optim.AdamW(module.parameters(), lr=0.01, foreach=False)

        def snapshot(module, opt, runtime):
            state = module.state_dict()
            return copy.deepcopy(
                (
                    state,
                    state._metadata,
                    opt.state_dict(),
                    runtime.capture(),
                    torch.get_rng_state(),
                    torch.cuda.get_rng_state(device),
                )
            )

        stage_model = model()
        reference_model = copy.deepcopy(stage_model)
        stage = trainer(stage_model)
        stage_opt = optimizer(stage_model)
        group = stage.mesh["dp"].get_group()
        reference = torch.nn.parallel.DistributedDataParallel(
            reference_model, device_ids=[rank], process_group=group
        )
        reference.register_comm_hook(state=group, hook=bf16_compress_hook)
        reference_opt = optimizer(reference_model)
        generator = torch.Generator(device=device).manual_seed(101 + rank)
        inputs = torch.randn(2 * microbatches, 8, generator=generator, device=device)
        targets = torch.randn(2 * microbatches, 4, generator=generator, device=device)

        loss = stage.forward_backward(inputs, targets).loss
        reference_losses = []
        for index, (batch, target) in enumerate(
            zip(inputs.chunk(microbatches), targets.chunk(microbatches))
        ):
            sync = reference.no_sync() if index + 1 < microbatches else nullcontext()
            with sync:
                part = loss_fn(reference(batch), target) / microbatches
                part.backward()
            reference_losses.append(part.detach())
        reference_loss = torch.stack(reference_losses).sum()
        dist.all_reduce(reference_loss, group=group)
        assert math.isfinite(loss)
        assert loss == pytest.approx(reference_loss.item() / world, rel=1e-6)

        for actual, expected in zip(
            stage_model.parameters(), reference_model.parameters()
        ):
            assert torch.isfinite(actual.grad).all()
            _equal(actual.grad, expected.grad)
            replicas = [torch.empty_like(actual.grad) for _ in range(world)]
            dist.all_gather(replicas, actual.grad, group=group)
            for replica in replicas:
                _equal(actual.grad, replica)
        stage_opt.step()
        reference_opt.step()
        _equal(stage_model.state_dict(), reference_model.state_dict())
        _equal(stage_opt.state_dict(), reference_opt.state_dict())

        runtime = RuntimeState(rank)
        runtime.progress = 1
        saved = snapshot(stage_model, stage_opt, runtime)
        for fmt in ("rank", "dcp"):
            stage.save_checkpoint(
                root / fmt, stage_opt, step=1, runtime=runtime, checkpoint_format=fmt
            )

        continuations = []
        for fmt in ("rank", "dcp"):
            restored = trainer(model())
            restored_opt = optimizer(restored.module)
            restored_runtime = RuntimeState(rank)
            torch.rand(1, device=device)
            assert (
                restored.load_checkpoint(
                    root / fmt, restored_opt, runtime=restored_runtime
                )
                == 1
            )
            _equal(snapshot(restored.module, restored_opt, restored_runtime), saved)
            next_loss = restored.forward_backward(inputs, targets).loss
            restored_opt.step()
            restored_runtime.progress += 1
            continuations.append(
                (next_loss, snapshot(restored.module, restored_opt, restored_runtime))
            )

        _equal(*continuations)
        incompatible = trainer(model(), policy="none")
        incompatible_opt = optimizer(incompatible.module)
        incompatible_runtime = RuntimeState(rank)
        before = snapshot(incompatible.module, incompatible_opt, incompatible_runtime)
        for fmt in ("rank", "dcp"):
            with pytest.raises(RuntimeError, match="DDP communication policy mismatch"):
                incompatible.load_checkpoint(
                    root / fmt, incompatible_opt, runtime=incompatible_runtime
                )
            _equal(
                snapshot(incompatible.module, incompatible_opt, incompatible_runtime),
                before,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("microbatches", [1, 2])
def test_stage_bf16_hook_matches_native_ddp_and_recovers(tmp_path, world, microbatches):
    if torch.__version__.split("+")[0] != "2.9.1":
        pytest.skip("Requires torch==2.9.1")
    if (
        not dist.is_available()
        or not dist.is_nccl_available()
        or torch.cuda.device_count() < world
    ):
        pytest.skip(f"Requires NCCL and {world} CUDA devices")
    if torch.cuda.nccl.version() <= (2, 9, 6):
        pytest.skip("Requires NCCL newer than 2.9.6")
    for rank in range(world):
        with torch.cuda.device(rank):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                pytest.skip("Requires native CUDA BF16 support on every device")
    torch.multiprocessing.spawn(
        _worker, args=(world, microbatches, str(tmp_path)), nprocs=world, join=True
    )
