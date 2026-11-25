# -*- coding: utf-8 -*-
# @Time : 2026/11/25 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""DCP storage fidelity and coordinated distributed recovery contracts."""

import copy
import json
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from parascale.parallel import distributed_checkpoint as checkpoint
from parascale.parallel.distributed import DistributedParallelConfig


def test_dcp_template_preserves_model_metadata_without_tensor_storage():
    from parascale.parallel.checkpoint_dcp import _copy_tree

    model = torch.nn.Linear(3, 2)
    state = model.state_dict()
    state._metadata[""]["version"] = 71
    source = {
        "model": state,
        "optimizer": {
            "state": {0: {"step": torch.tensor(2.0)}},
            "param_groups": [{"params": [0], "betas": (0.9, 0.999)}],
        },
    }
    template = _copy_tree(source, "meta")
    assert isinstance(template["model"], OrderedDict)
    assert template["model"]._metadata == state._metadata
    assert template["model"]._metadata is not state._metadata
    assert template["model"]["weight"].device.type == "meta"
    target = _copy_tree(template, "cpu")
    assert target["model"]["weight"].shape == state["weight"].shape
    assert target["optimizer"]["param_groups"][0]["betas"] == (0.9, 0.999)
    assert target["model"]._metadata[""]["version"] == 71


def test_dcp_replica_comparison_preserves_signed_zero_bits():
    from parascale.parallel.checkpoint_dcp import _prepare_replica

    source = torch.tensor([0.0, -0.0])
    target = torch.tensor([0.0, 0.0])
    assert torch.equal(source, target)
    scratch = torch.empty(8, dtype=torch.uint8)
    comparison = torch.empty(8, dtype=torch.bool)
    _, incoming, _ = _prepare_replica(source, "cpu", scratch, comparison, True)
    local, received, mask = _prepare_replica(target, "cpu", scratch, comparison, False)
    assert local.dtype == incoming.dtype == torch.uint8
    torch.eq(local, received, out=mask)
    assert not mask.all().item()


@pytest.mark.parametrize("value", [object(), {True: 1}, float("nan"), {"x": {1, 2}}])
def test_dcp_template_rejects_unsupported_state(value):
    from parascale.parallel.checkpoint_dcp import _copy_tree

    with pytest.raises(ValueError, match="DCP"):
        _copy_tree(value, "meta")


def test_dcp_template_rejects_nested_metadata_tensor():
    from parascale.parallel.checkpoint_dcp import _copy_tree

    state = torch.nn.Linear(2, 2).state_dict()
    state._metadata[""]["custom"] = [{"tensor": torch.tensor(123)}]
    with pytest.raises(ValueError, match="metadata.*tensor"):
        _copy_tree(state, "meta")


@pytest.mark.parametrize("shape,dtype", [((2,), torch.int64), ((1,), torch.float32)])
def test_dcp_index_rejects_incompatible_cpu_target(tmp_path, shape, dtype):
    import torch.distributed.checkpoint as dcp

    from parascale.parallel.checkpoint_dcp import _validate_tensor_index

    source = {"stage_0": {"model": {"nested.weight": torch.tensor([1.75, 2.25])}}}
    index = dcp.save(source, checkpoint_id=tmp_path, no_dist=True)
    target = torch.full(shape, -7, dtype=dtype)
    with pytest.raises(ValueError, match="DCP index tensor mismatch"):
        _validate_tensor_index({"stage_0": {"model": {"nested.weight": target}}}, index)
    assert torch.all(target == -7)


class VersionedLinear(torch.nn.Linear):
    _version = 71

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args):
        if local_metadata.get("version") != self._version:
            raise ValueError("Lost module version metadata")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, *args)


class RuntimeState:
    def __init__(self, rank):
        self.rank = rank
        self.progress = 2

    def identity(self, optimizer):
        return {"kind": "dcp-test"}

    def capture(self):
        return {"rank": self.rank, "progress": self.progress}

    def validate(self, state, *, step, optimizer_state):
        if state != {"rank": self.rank, "progress": step}:
            raise ValueError("Bad rank-local progress")

    def restore(self, state):
        self.progress = state["progress"]


def _equal(left, right):
    if isinstance(right, torch.Tensor):
        assert (left.dtype, left.shape, left.device) == (
            right.dtype,
            right.shape,
            right.device,
        )
        assert torch.equal(
            left.reshape(-1).contiguous().view(torch.uint8),
            right.reshape(-1).contiguous().view(torch.uint8),
        )
    elif isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in right:
            _equal(left[key], right[key])
    elif isinstance(right, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for actual, expected in zip(left, right):
            _equal(actual, expected)
    else:
        assert type(left) is type(right) and left == right


def _worker(rank, folder, backend):
    root = Path(folder)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        torch.set_num_threads(1)
        torch.manual_seed(42 + rank)
        model = torch.nn.Sequential(VersionedLinear(4, 4), torch.nn.Dropout(0.1)).to(
            device
        )
        ddp = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank] if device.type == "cuda" else None
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        trainer = SimpleNamespace(
            module=model,
            config=DistributedParallelConfig(dp_size=2),
            rank=rank,
            device=device,
            checkpoint_context={
                "model": {"kind": "linear"},
                "loss": "sum",
                "data": "fixed",
                "tokenizer": "none",
                "precision": "fp32",
            },
        )
        runtime = RuntimeState(rank)

        def update():
            optimizer.zero_grad(set_to_none=True)
            ddp(torch.randn(3, 4, device=device)).square().mean().backward()
            optimizer.step()

        def rng_state():
            return (
                torch.get_rng_state(),
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            )

        update()
        update()
        path = root / "step-00000002"
        checkpoint.save_checkpoint(
            trainer, path, optimizer, step=2, runtime=runtime, checkpoint_format="dcp"
        )
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["schema_version"] == 6
        assert manifest["storage"]["torch_version"] == str(torch.__version__)
        update()
        expected = copy.deepcopy(
            (model.state_dict(), optimizer.state_dict(), rng_state())
        )
        runtime.progress = 0
        assert (
            checkpoint.load_checkpoint(trainer, path, optimizer, runtime=runtime) == 2
        )
        assert runtime.progress == 2
        update()
        _equal(model.state_dict(), expected[0])
        _equal(optimizer.state_dict(), expected[1])
        _equal(rng_state(), expected[2])
        assert model.state_dict()._metadata["0"]["version"] == 71

        for fault in ("version", "data", "metadata", "rank"):
            dist.barrier()
            target = path / "manifest.json"
            if fault != "version":
                entry = next(
                    row
                    for row in manifest["files"]
                    if (
                        row["path"].endswith(".distcp")
                        if fault == "data"
                        else row["path"] == "shared/.metadata"
                        if fault == "metadata"
                        else row["path"] == "rank-00001.pt"
                    )
                )
                target = path / entry["path"]
            saved = target.read_bytes() if rank == 0 else None
            if rank == 0:
                if fault == "version":
                    bad = copy.deepcopy(manifest)
                    bad["storage"]["torch_version"] = "0.0.0"
                    target.write_text(json.dumps(bad))
                else:
                    target.write_bytes(b"corrupt")
            dist.barrier()
            before = copy.deepcopy(
                (model.state_dict(), optimizer.state_dict(), rng_state())
            )
            try:
                checkpoint.load_checkpoint(trainer, path, optimizer, runtime=runtime)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"Accepted {fault} corruption")
            _equal(model.state_dict(), before[0])
            _equal(optimizer.state_dict(), before[1])
            _equal(rng_state(), before[2])
            dist.barrier()
            if rank == 0:
                target.write_bytes(saved)
            dist.barrier()

        for fault in ("tensor", "metadata"):
            runtime.progress = 3
            if rank == 1:
                if fault == "tensor":
                    with torch.no_grad():
                        next(model.parameters()).view(-1)[0].add_(1)
                else:
                    original_lr = optimizer.param_groups[0]["lr"]
                    optimizer.param_groups[0]["lr"] = original_lr + 1
            rejected = root / f"rejected-{fault}"
            try:
                checkpoint.save_checkpoint(
                    trainer,
                    rejected,
                    optimizer,
                    step=3,
                    runtime=runtime,
                    checkpoint_format="dcp",
                )
            except RuntimeError as error:
                assert "replica" in str(error).lower()
            else:
                raise AssertionError("Accepted divergent replicas")
            assert not (rejected / "manifest.json").exists()
            if rank == 1:
                if fault == "tensor":
                    with torch.no_grad():
                        next(model.parameters()).copy_(expected[0]["0.weight"])
                else:
                    optimizer.param_groups[0]["lr"] = original_lr
            dist.barrier()

        from parascale.parallel import checkpoint_dcp

        def fail_preparation(*args):
            raise RuntimeError("injected replica preparation failure")

        rejected = root / "rejected-prepare"
        with pytest.MonkeyPatch.context() as patch:
            if rank == 1:
                patch.setattr(
                    checkpoint_dcp, "_prepare_replica", fail_preparation, raising=False
                )
            with pytest.raises(
                RuntimeError, match="rank 1:.*injected replica preparation"
            ):
                checkpoint.save_checkpoint(
                    trainer,
                    rejected,
                    optimizer,
                    step=3,
                    runtime=runtime,
                    checkpoint_format="dcp",
                )
        assert not (rejected / "manifest.json").exists()
        dist.barrier()
        (root / f"rank-{rank}-passed").touch()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_real_dcp_resume_and_failure_contracts(tmp_path, backend):
    if not dist.is_available() or (backend == "gloo" and not dist.is_gloo_available()):
        pytest.skip("Gloo unavailable")
    if backend == "nccl" and (
        not dist.is_nccl_available() or torch.cuda.device_count() < 2
    ):
        pytest.skip("Two CUDA devices and NCCL required")
    mp.spawn(_worker, args=(str(tmp_path), backend), nprocs=2, join=True)
    assert all((tmp_path / f"rank-{rank}-passed").is_file() for rank in range(2))


def _uneven_pp_worker(rank, folder):
    from torch.distributed.device_mesh import init_device_mesh

    from parascale.parallel import checkpoint_dcp

    dist.init_process_group(
        "gloo",
        init_method=(Path(folder) / "rendezvous").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=20),
    )
    try:
        torch.set_num_threads(1)
        config = DistributedParallelConfig(dp_size=2, pp_size=2)
        _, stage, _ = config.coordinates(rank)
        trainer = SimpleNamespace(
            config=config,
            device=torch.device("cpu"),
            mesh=init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "pp")),
        )
        torch.manual_seed(42 + stage)
        model = torch.nn.Sequential(*(torch.nn.Linear(2, 2) for _ in range(1 + stage)))
        common = {"model": model.state_dict()}
        assert len(common["model"]) == (2, 4)[stage]
        checkpoint_dcp._verify_replicas(trainer, common)

        prepare = checkpoint_dcp._prepare_replica
        calls = 0

        def fail_on_third(*args):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("injected failure during padded iteration")
            return prepare(*args)

        with pytest.MonkeyPatch.context() as patch:
            if rank == 3:
                patch.setattr(checkpoint_dcp, "_prepare_replica", fail_on_third)
            with pytest.raises(
                RuntimeError, match="rank 3:.*injected failure during padded iteration"
            ):
                checkpoint_dcp._verify_replicas(trainer, common)
        if rank == 3:
            assert calls == 3
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_dcp_replica_phases_pad_uneven_pipeline_stages(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_uneven_pp_worker, args=(str(tmp_path),), nprocs=4, join=True)


def _fresh_format_worker(rank, folder, backend):
    root = Path(folder)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=90),
    )
    try:
        torch.set_num_threads(1)
        torch.manual_seed(42 + rank)

        def fresh():
            model = torch.nn.Sequential(
                VersionedLinear(4, 4), torch.nn.Dropout(0.1)
            ).to(device)
            ddp = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[rank] if device.type == "cuda" else None
            )
            trainer = SimpleNamespace(
                module=model,
                config=DistributedParallelConfig(dp_size=4),
                rank=rank,
                device=device,
                checkpoint_context={
                    "model": {"kind": "linear"},
                    "loss": "sum",
                    "data": "fixed",
                    "tokenizer": "none",
                    "precision": "fp32",
                },
            )
            return trainer, ddp, torch.optim.AdamW(model.parameters(), lr=0.01)

        def update(ddp, optimizer, batch):
            optimizer.zero_grad(set_to_none=True)
            loss = ddp(batch).square().mean()
            loss.backward()
            optimizer.step()
            return loss.detach()

        def snapshot(trainer, optimizer, runtime):
            state = trainer.module.state_dict()
            return copy.deepcopy(
                (
                    state,
                    state._metadata,
                    optimizer.state_dict(),
                    runtime.capture(),
                    torch.get_rng_state(),
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                )
            )

        trainer, warm, optimizer = fresh()
        runtime = RuntimeState(rank)
        batches = [torch.randn(3, 4, device=device) for _ in range(3)]
        for batch in batches[:2]:
            update(warm, optimizer, batch)
        saved = snapshot(trainer, optimizer, runtime)
        for fmt in ("rank", "dcp"):
            checkpoint.save_checkpoint(
                trainer,
                root / fmt,
                optimizer,
                step=2,
                runtime=runtime,
                checkpoint_format=fmt,
            )

        results = []
        for fmt in ("rank", "dcp"):
            restored, cold, restored_optimizer = fresh()
            restored_runtime = RuntimeState(rank)
            restored_runtime.progress = 0
            assert (
                checkpoint.load_checkpoint(
                    restored, root / fmt, restored_optimizer, runtime=restored_runtime
                )
                == 2
            )
            _equal(snapshot(restored, restored_optimizer, restored_runtime), saved)
            loss = update(cold, restored_optimizer, batches[2])
            restored_runtime.progress += 1
            results.append(
                (loss, snapshot(restored, restored_optimizer, restored_runtime))
            )

        _equal(*results)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_dp4_fresh_ddp_resume_is_exact_across_rank_and_dcp(tmp_path, backend):
    if not dist.is_available() or (backend == "gloo" and not dist.is_gloo_available()):
        pytest.skip("Gloo unavailable")
    if backend == "nccl" and (
        not dist.is_nccl_available() or torch.cuda.device_count() < 4
    ):
        pytest.skip("Four CUDA devices and NCCL required")
    mp.spawn(_fresh_format_worker, args=(str(tmp_path), backend), nprocs=4, join=True)
