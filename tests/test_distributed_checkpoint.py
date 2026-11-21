# -*- coding: utf-8 -*-
# @Time : 2026/11/21 下午6:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""CPU checkpoint regression tests with explicitly injected collectives."""

import errno
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from parascale.parallel import distributed_checkpoint as checkpoint
from parascale.parallel.distributed import DistributedParallelConfig


@pytest.fixture
def two_ranks(monkeypatch):
    local = threading.local()
    barrier = threading.Barrier(2, timeout=10)
    values = {}

    def gather(output, value):
        index = local.index
        local.index += 1
        values[index, local.rank] = value
        barrier.wait()
        output[:] = [values[index, rank] for rank in range(2)]
        barrier.wait()

    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: local.rank)

    def run(operation):
        values.clear()

        def worker(rank):
            local.rank, local.index = rank, 0
            try:
                return operation(rank)
            except Exception as error:
                return error

        with ThreadPoolExecutor(max_workers=2) as executor:
            return list(executor.map(worker, range(2)))

    return run


def _rank_trainers():
    trainers = [
        SimpleNamespace(
            module=torch.nn.Linear(2, 2),
            rank=rank,
            config=DistributedParallelConfig(dp_size=2),
            device=torch.device("cpu"),
        )
        for rank in range(2)
    ]
    return trainers, [torch.optim.SGD(t.module.parameters(), lr=0.1) for t in trainers]


@pytest.mark.parametrize("mismatch", ["step", "directory", "topology"])
def test_save_identity_mismatch_rejected_before_writing(tmp_path, two_ranks, mismatch):
    trainers, optimizers = _rank_trainers()
    if mismatch == "topology":
        trainers[1].config = DistributedParallelConfig(dp_size=2, microbatches=4)
    outcomes = two_ranks(
        lambda rank: checkpoint.save_checkpoint(
            trainers[rank],
            tmp_path / (str(rank) if mismatch == "directory" else "shared"),
            optimizers[rank],
            step=7 + (rank if mismatch == "step" else 0),
        )
    )
    assert all(
        isinstance(result, RuntimeError) and "identity" in str(result)
        for result in outcomes
    )
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("manifest.json"))


def test_load_mixed_checkpoints_rejected_before_model_mutation(tmp_path, two_ranks):
    trainers, optimizers = _rank_trainers()
    for step in (7, 8):
        assert two_ranks(
            lambda rank: checkpoint.save_checkpoint(
                trainers[rank], tmp_path / str(step), optimizers[rank], step=step
            )
        ) == [None, None]
    for trainer in trainers:
        trainer.module.load_state_dict = Mock(side_effect=AssertionError("mutation"))
    outcomes = two_ranks(
        lambda rank: checkpoint.load_checkpoint(
            trainers[rank], tmp_path / str(7 + rank), optimizers[rank]
        )
    )
    assert all(
        isinstance(result, RuntimeError) and "identity" in str(result)
        for result in outcomes
    )
    for trainer in trainers:
        trainer.module.load_state_dict.assert_not_called()


@pytest.fixture
def runtime(monkeypatch):
    trainer = SimpleNamespace(
        module=torch.nn.Linear(2, 2),
        rank=0,
        config=DistributedParallelConfig(),
        device=torch.device("cpu"),
    )
    optimizer = torch.optim.Adam(trainer.module.parameters())
    calls = []

    def gather(output, value):
        calls.append(value)
        output[:] = [value]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "barrier", Mock(side_effect=AssertionError("barrier")))
    monkeypatch.setattr(
        dist, "all_reduce", Mock(side_effect=AssertionError("all_reduce"))
    )
    monkeypatch.setattr(
        torch.cuda, "get_rng_state", Mock(side_effect=AssertionError("CUDA"))
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", Mock(side_effect=AssertionError("CUDA"))
    )
    return trainer, optimizer, calls


def write_checkpoint(
    directory, trainer, optimizer, *, metadata_edit=None, payload_edit=None
):
    payload = dict(
        model=trainer.module.state_dict(),
        optimizer=optimizer.state_dict(),
        cpu_rng=torch.get_rng_state(),
        cuda_rng=None,
        step=7,
    )
    if payload_edit:
        payload_edit(payload)
    path = directory / "rank-00000.pt"
    torch.save(payload, path)
    metadata = dict(
        schema_version=1,
        topology=asdict(trainer.config),
        step=7,
        files=[
            dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        ],
    )
    if metadata_edit:
        metadata_edit(metadata)
    (directory / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_cpu_roundtrip_restores_model_optimizer_rng(tmp_path, runtime):
    trainer, optimizer, _ = runtime
    trainer.module(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    expected = {k: v.clone() for k, v in trainer.module.state_dict().items()}
    rng = torch.get_rng_state().clone()
    checkpoint.save_checkpoint(trainer, tmp_path, optimizer, step=7)
    with torch.no_grad():
        for parameter in trainer.module.parameters():
            parameter.zero_()
    optimizer.state.clear()
    torch.manual_seed(98)
    assert checkpoint.load_checkpoint(trainer, tmp_path, optimizer) == 7
    for key, value in trainer.module.state_dict().items():
        torch.testing.assert_close(value, expected[key])
    assert len(optimizer.state) == 2
    assert all(value["step"].item() == 1 for value in optimizer.state.values())
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("step", [-1, True, 1.5])
def test_invalid_step_is_coordinated_before_state_dict(
    tmp_path, runtime, monkeypatch, step
):
    trainer, optimizer, calls = runtime
    state = Mock(side_effect=AssertionError("state_dict must not run"))
    monkeypatch.setattr(trainer.module, "state_dict", state)
    with pytest.raises(RuntimeError, match="step"):
        checkpoint.save_checkpoint(trainer, tmp_path, optimizer, step=step)
    assert calls and "step" in calls[-1]
    state.assert_not_called()


def test_committed_checkpoint_is_immutable(tmp_path, runtime, monkeypatch):
    trainer, optimizer, calls = runtime
    write_checkpoint(tmp_path, trainer, optimizer)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    state = Mock()
    monkeypatch.setattr(trainer.module, "state_dict", state)
    with pytest.raises(RuntimeError, match="committed"):
        checkpoint.save_checkpoint(trainer, tmp_path, optimizer, step=8)
    assert calls
    state.assert_not_called()
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m.update(schema_version=True),
        lambda m: m.update(schema_version=2),
        lambda m: m.update(extra=1),
        lambda m: m.pop("step"),
        lambda m: m.update(step=True),
        lambda m: m["topology"].update(microbatches=9),
        lambda m: m["topology"].update(dp_size=True),
        lambda m: m.update(files=[]),
        lambda m: m["files"][0].update(path="../rank-00000.pt"),
        lambda m: m["files"][0].update(extra=1),
        lambda m: m["files"][0].update(sha256="bad"),
    ],
)
def test_manifest_rejected_before_mutation(tmp_path, runtime, monkeypatch, edit):
    trainer, optimizer, calls = runtime
    write_checkpoint(tmp_path, trainer, optimizer, metadata_edit=edit)
    load = Mock()
    monkeypatch.setattr(trainer.module, "load_state_dict", load)
    with pytest.raises(RuntimeError):
        checkpoint.load_checkpoint(trainer, tmp_path, optimizer)
    assert calls and calls[-1] is not None
    load.assert_not_called()


@pytest.mark.parametrize(
    "edit",
    [
        lambda p: p.update(step=8),
        lambda p: p.pop("cpu_rng"),
        lambda p: p.update(extra=1),
        lambda p: p["model"].pop("bias"),
        lambda p: p["model"].update(weight=torch.ones(3, 2)),
        lambda p: p["optimizer"].update(param_groups=[]),
        lambda p: p.update(cpu_rng=torch.tensor([1])),
        lambda p: p.update(cuda_rng=torch.ones(4)),
    ],
)
def test_payload_rejected_before_mutation(tmp_path, runtime, monkeypatch, edit):
    trainer, optimizer, calls = runtime
    write_checkpoint(tmp_path, trainer, optimizer, payload_edit=edit)
    load = Mock()
    monkeypatch.setattr(trainer.module, "load_state_dict", load)
    with pytest.raises(RuntimeError):
        checkpoint.load_checkpoint(trainer, tmp_path, optimizer)
    assert calls and calls[-1] is not None
    load.assert_not_called()


@pytest.mark.parametrize("damage", ["missing", "checksum", "deserialize", "manifest"])
def test_damaged_files_are_coordinated(tmp_path, runtime, monkeypatch, damage):
    trainer, optimizer, calls = runtime
    write_checkpoint(tmp_path, trainer, optimizer)
    path = tmp_path / "rank-00000.pt"
    if damage == "missing":
        path.unlink()
    elif damage == "manifest":
        (tmp_path / "manifest.json").write_text("{", encoding="utf-8")
    else:
        path.write_bytes(b"not a torch checkpoint")
        if damage == "deserialize":
            manifest = tmp_path / "manifest.json"
            metadata = json.loads(manifest.read_text())
            metadata["files"][0]["sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            manifest.write_text(json.dumps(metadata))
    load = Mock()
    monkeypatch.setattr(trainer.module, "load_state_dict", load)
    with pytest.raises(RuntimeError):
        checkpoint.load_checkpoint(trainer, tmp_path, optimizer)
    assert calls and calls[-1] is not None
    load.assert_not_called()


@pytest.mark.parametrize("operation", ["save", "load"])
def test_peer_preflight_failure_stops_local_state_access(
    tmp_path, runtime, monkeypatch, operation
):
    trainer, optimizer, _ = runtime
    write_checkpoint(tmp_path, trainer, optimizer)
    if operation == "save":
        (tmp_path / "manifest.json").unlink()
    state = Mock(side_effect=AssertionError("state access after peer failure"))
    monkeypatch.setattr(trainer.module, "state_dict", state)
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda out, value: out.__setitem__(slice(None), [None, "peer disk failure"]),
    )
    with pytest.raises(RuntimeError, match="peer disk failure"):
        if operation == "save":
            checkpoint.save_checkpoint(trainer, tmp_path, optimizer, step=7)
        else:
            checkpoint.load_checkpoint(trainer, tmp_path, optimizer)
    state.assert_not_called()


@pytest.mark.parametrize("failure", ["state", "write", "publish"])
def test_save_failure_is_coordinated_without_manifest(
    tmp_path, runtime, monkeypatch, failure
):
    trainer, optimizer, calls = runtime
    if failure == "state":
        monkeypatch.setattr(
            trainer.module, "state_dict", Mock(side_effect=RuntimeError("injected"))
        )
    elif failure == "write":
        monkeypatch.setattr(torch, "save", Mock(side_effect=OSError("injected")))
    else:
        replace = checkpoint.os.replace

        def fail_manifest(source, target):
            if target.name == "manifest.json":
                raise OSError("injected")
            replace(source, target)

        monkeypatch.setattr(checkpoint.os, "replace", fail_manifest)
    with pytest.raises(RuntimeError, match="injected"):
        checkpoint.save_checkpoint(trainer, tmp_path, optimizer, step=7)
    assert calls and "injected" in calls[-1]
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("phase", [1, 2, 3, 4])
def test_peer_load_failure_prevents_all_mutation(tmp_path, runtime, monkeypatch, phase):
    trainer, optimizer, calls = runtime
    trainer.config = DistributedParallelConfig(dp_size=2)
    write_checkpoint(
        tmp_path,
        trainer,
        optimizer,
        metadata_edit=lambda m: m["files"].append(
            dict(path="rank-00001.pt", sha256="0" * 64)
        ),
    )
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def gather(output, value):
        if isinstance(value, dict) and "directory" in value:
            output[:] = [value, value]
            return
        calls.append(value)
        output[:] = [value, "peer failure" if len(calls) == phase else None]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    model_load, optimizer_load, rng = Mock(), Mock(), Mock()
    monkeypatch.setattr(trainer.module, "load_state_dict", model_load)
    monkeypatch.setattr(optimizer, "load_state_dict", optimizer_load)
    monkeypatch.setattr(torch, "set_rng_state", rng)
    with pytest.raises(RuntimeError, match="rank 1: peer failure"):
        checkpoint.load_checkpoint(trainer, tmp_path, optimizer)
    assert len(calls) == phase
    model_load.assert_not_called()
    optimizer_load.assert_not_called()
    rng.assert_not_called()


@pytest.mark.parametrize("phase", [1, 2, 3, 4])
def test_peer_save_failure_prevents_publication(tmp_path, runtime, monkeypatch, phase):
    trainer, optimizer, calls = runtime
    trainer.config = DistributedParallelConfig(dp_size=2)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def gather(output, value):
        if isinstance(value, dict) and "directory" in value:
            output[:] = [value, value]
            return
        calls.append(value)
        output[:] = [value, "peer failure" if len(calls) == phase else None]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="rank 1: peer failure"):
        checkpoint.save_checkpoint(trainer, tmp_path, optimizer, step=7)
    assert len(calls) == phase
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("target", ["model", "optimizer", "rng"])
def test_application_error_is_reported_without_success(
    tmp_path, runtime, monkeypatch, target
):
    trainer, optimizer, calls = runtime
    write_checkpoint(tmp_path, trainer, optimizer)
    failure = Mock(side_effect=RuntimeError("apply failure"))
    if target == "model":
        monkeypatch.setattr(trainer.module, "load_state_dict", failure)
    elif target == "optimizer":
        monkeypatch.setattr(optimizer, "load_state_dict", failure)
    else:
        monkeypatch.setattr(torch, "set_rng_state", failure)
    with pytest.raises(RuntimeError, match="apply failure"):
        checkpoint.load_checkpoint(trainer, tmp_path, optimizer)
    assert "apply failure" in calls[-1]


def test_directory_creation_error_is_coordinated(tmp_path, runtime, monkeypatch):
    trainer, optimizer, calls = runtime
    directory = tmp_path / "not-a-directory"
    directory.write_bytes(b"existing")
    state = Mock()
    monkeypatch.setattr(trainer.module, "state_dict", state)
    with pytest.raises(RuntimeError, match="directory creation"):
        checkpoint.save_checkpoint(trainer, directory, optimizer, step=7)
    state.assert_not_called()
    assert calls[-1] is not None


def test_save_syncs_directory_after_atomic_renames(tmp_path, runtime, monkeypatch):
    trainer, optimizer, _ = runtime
    seen = []

    def sync(directory):
        seen.append(
            (
                directory,
                (tmp_path / "rank-00000.pt").exists(),
                (tmp_path / "manifest.json").exists(),
            )
        )

    monkeypatch.setattr(checkpoint, "_fsync_directory", sync)
    checkpoint.save_checkpoint(trainer, tmp_path, optimizer, step=7)
    assert seen == [
        (tmp_path.parent, False, False),
        (tmp_path, True, False),
        (tmp_path, True, True),
    ]


@pytest.mark.parametrize("error", [None, errno.EINVAL, errno.EIO])
def test_directory_fsync_closes_descriptor_and_only_ignores_unsupported(
    tmp_path, monkeypatch, error
):
    fake_os = SimpleNamespace(
        name="posix",
        O_RDONLY=0,
        O_DIRECTORY=65536,
        open=Mock(return_value=42),
        close=Mock(),
        fsync=Mock(),
    )
    if error:
        fake_os.fsync.side_effect = OSError(error, "injected")
    monkeypatch.setattr(checkpoint, "os", fake_os)
    if error == errno.EIO:
        with pytest.raises(OSError):
            checkpoint._fsync_directory(tmp_path)
    else:
        checkpoint._fsync_directory(tmp_path)
    fake_os.open.assert_called_once_with(tmp_path, 65536)
    fake_os.fsync.assert_called_once_with(42)
    fake_os.close.assert_called_once_with(42)


def test_nonzero_rank_loads_its_own_file(tmp_path, runtime, monkeypatch):
    trainer, optimizer, _ = runtime
    trainer.rank = 1
    trainer.config = DistributedParallelConfig(dp_size=2)
    write_checkpoint(tmp_path, trainer, optimizer)
    (tmp_path / "rank-00000.pt").rename(tmp_path / "rank-00001.pt")
    manifest = tmp_path / "manifest.json"
    metadata = json.loads(manifest.read_text())
    entry = metadata["files"][0]
    metadata["files"] = [
        dict(path="rank-00000.pt", sha256="0" * 64),
        dict(path="rank-00001.pt", sha256=entry["sha256"]),
    ]
    manifest.write_text(json.dumps(metadata))
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda out, value: out.__setitem__(slice(None), [value, value]),
    )
    assert checkpoint.load_checkpoint(trainer, tmp_path, optimizer) == 7


def test_nonzero_rank_receives_publication_failure(tmp_path, runtime, monkeypatch):
    trainer, optimizer, calls = runtime
    trainer.rank = 1
    trainer.config = DistributedParallelConfig(dp_size=2)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)

    def gather(output, value):
        if isinstance(value, dict) and "directory" in value:
            output[:] = [value, value]
            return
        calls.append(value)
        if isinstance(value, dict):
            output[:] = [dict(path="rank-00000.pt", sha256="0" * 64), value]
        else:
            output[:] = ["publish failure" if len(calls) == 6 else None, value]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="rank 0: publish failure"):
        checkpoint.save_checkpoint(trainer, tmp_path, optimizer, step=7)
    assert len(calls) == 6
    assert (tmp_path / "rank-00001.pt").exists()
    assert not (tmp_path / "manifest.json").exists()
