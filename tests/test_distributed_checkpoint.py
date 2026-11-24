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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from parascale.parallel import distributed_checkpoint as checkpoint
from parascale.parallel.distributed import DistributedParallelConfig


class RuntimeParticipant:
    """Local-only runtime used to exercise the checkpoint transaction boundary."""

    def __init__(self, optimizer, *, identity="test-runtime-v1"):
        self.optimizer = optimizer
        self.name = identity
        self.state = {"progress": 7}
        self.restored = False

    def identity(self, optimizer):
        assert optimizer is self.optimizer
        return {"kind": self.name}

    def capture(self):
        return dict(self.state)

    def validate(self, state, *, step, optimizer_state):
        if state != {"progress": step}:
            raise ValueError("runtime progress mismatch")

    def restore(self, state):
        self.state = dict(state)
        self.restored = True


def test_runtime_checkpoint_roundtrip_and_rng_application_order(
    tmp_path, runtime, monkeypatch
):
    trainer, optimizer, _ = runtime
    participant = RuntimeParticipant(optimizer)
    checkpoint.save_checkpoint(
        trainer, tmp_path, optimizer, step=7, runtime=participant
    )
    metadata = json.loads((tmp_path / "manifest.json").read_text())
    assert metadata["schema_version"] == 5
    assert metadata["runtime"] == participant.identity(optimizer)
    participant.state = {"progress": 0}
    original = torch.set_rng_state

    def restore_rng(value):
        assert participant.restored
        original(value)

    monkeypatch.setattr(torch, "set_rng_state", restore_rng)
    assert (
        checkpoint.load_checkpoint(trainer, tmp_path, optimizer, runtime=participant)
        == 7
    )
    assert participant.state == {"progress": 7}


@pytest.mark.parametrize(
    "saved,requested", [(False, True), (True, False), (True, True)]
)
def test_runtime_presence_and_identity_rejected_before_payload_reads(
    tmp_path, runtime, monkeypatch, saved, requested
):
    trainer, optimizer, _ = runtime
    participant = RuntimeParticipant(optimizer)
    checkpoint.save_checkpoint(
        trainer, tmp_path, optimizer, step=7, runtime=participant if saved else None
    )
    participant.name = "changed-runtime"
    loader = Mock(side_effect=AssertionError("payload read"))
    monkeypatch.setattr(torch, "load", loader)
    with pytest.raises(RuntimeError, match="runtime|schema"):
        checkpoint.load_checkpoint(
            trainer, tmp_path, optimizer, runtime=participant if requested else None
        )
    loader.assert_not_called()


def test_runtime_validation_failure_prevents_all_rank_application(tmp_path, two_ranks):
    trainers, optimizers = _rank_trainers()
    participants = [RuntimeParticipant(optimizer) for optimizer in optimizers]
    assert two_ranks(
        lambda rank: checkpoint.save_checkpoint(
            trainers[rank],
            tmp_path,
            optimizers[rank],
            step=7,
            runtime=participants[rank],
        )
    ) == [None, None]
    participants[1].validate = Mock(side_effect=ValueError("bad cursor"))
    for trainer, optimizer, participant in zip(trainers, optimizers, participants):
        trainer.module.load_state_dict = Mock(
            side_effect=AssertionError("model mutation")
        )
        optimizer.load_state_dict = Mock(
            side_effect=AssertionError("optimizer mutation")
        )
        participant.restore = Mock(side_effect=AssertionError("runtime mutation"))
    outcomes = two_ranks(
        lambda rank: checkpoint.load_checkpoint(
            trainers[rank], tmp_path, optimizers[rank], runtime=participants[rank]
        )
    )
    assert all(
        isinstance(error, RuntimeError) and "bad cursor" in str(error)
        for error in outcomes
    )
    for trainer, optimizer, participant in zip(trainers, optimizers, participants):
        trainer.module.load_state_dict.assert_not_called()
        optimizer.load_state_dict.assert_not_called()
        participant.restore.assert_not_called()


def test_runtime_capture_failure_never_publishes_manifest(tmp_path, two_ranks):
    trainers, optimizers = _rank_trainers()
    participants = [RuntimeParticipant(optimizer) for optimizer in optimizers]
    participants[1].state = {"progress": 8}
    outcomes = two_ranks(
        lambda rank: checkpoint.save_checkpoint(
            trainers[rank],
            tmp_path,
            optimizers[rank],
            step=7,
            runtime=participants[rank],
        )
    )
    assert all(
        isinstance(error, RuntimeError) and "runtime progress" in str(error)
        for error in outcomes
    )
    assert not list(tmp_path.rglob("manifest.json"))


def training_context():
    return {
        "model": {"kind": "linear", "width": 2},
        "loss": {"kind": "sum"},
        "data": {"kind": "deterministic-test", "revision": 1},
        "tokenizer": "none",
        "precision": "fp32",
    }


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
            checkpoint_context=training_context(),
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


@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
def test_pipeline_checkpoint_binds_schedule_before_payload_reads(
    tmp_path, two_ranks, monkeypatch, schedule
):
    trainers, optimizers = _rank_trainers()
    for trainer in trainers:
        trainer.config = DistributedParallelConfig(
            pp_size=2, pipeline_schedule=schedule
        )
    assert two_ranks(
        lambda rank: checkpoint.save_checkpoint(
            trainers[rank], tmp_path, optimizers[rank], step=7
        )
    ) == [None, None]
    metadata = json.loads((tmp_path / "manifest.json").read_text())
    assert metadata["schema_version"] == 4
    assert metadata["topology"]["pipeline_schedule"] == schedule
    assert two_ranks(
        lambda rank: checkpoint.load_checkpoint(
            trainers[rank], tmp_path, optimizers[rank]
        )
    ) == [7, 7]
    trainers[1].config = DistributedParallelConfig(
        pp_size=2, pipeline_schedule="1f1b" if schedule == "gpipe" else "gpipe"
    )
    loader = Mock(side_effect=AssertionError("payload read before schedule validation"))
    monkeypatch.setattr(torch, "load", loader)
    outcomes = two_ranks(
        lambda rank: checkpoint.load_checkpoint(
            trainers[rank], tmp_path, optimizers[rank]
        )
    )
    assert all(
        isinstance(error, RuntimeError) and "topology/schema" in str(error)
        for error in outcomes
    )
    loader.assert_not_called()


@pytest.fixture
def runtime(monkeypatch):
    trainer = SimpleNamespace(
        module=torch.nn.Linear(2, 2),
        rank=0,
        config=DistributedParallelConfig(),
        device=torch.device("cpu"),
        checkpoint_context=training_context(),
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
        optimizer_identity=checkpoint._optimizer_identity(trainer.module, optimizer),
        cpu_rng=torch.get_rng_state(),
        cuda_rng=None,
        step=7,
    )
    if payload_edit:
        payload_edit(payload)
    path = directory / "rank-00000.pt"
    torch.save(payload, path)
    metadata = dict(
        schema_version=4,
        context=trainer.checkpoint_context,
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


@pytest.mark.parametrize("field", ["model", "loss", "data", "tokenizer", "precision"])
def test_training_context_mismatch_rejected_before_payload_read(
    tmp_path, runtime, monkeypatch, field
):
    trainer, optimizer, _ = runtime
    checkpoint.save_checkpoint(trainer, tmp_path, optimizer, step=7)
    trainer.checkpoint_context[field] = "bf16" if field == "precision" else "changed"
    from parascale.parallel.checkpoint_context import canonical_checkpoint_context

    canonical_checkpoint_context(trainer.checkpoint_context)
    read = Mock(side_effect=AssertionError("payload read"))
    monkeypatch.setattr(torch, "load", read)
    with pytest.raises(RuntimeError, match="context"):
        checkpoint.load_checkpoint(trainer, tmp_path, optimizer)
    read.assert_not_called()


@pytest.mark.parametrize("invalid", [None, {}, {"model": "incomplete"}])
def test_missing_training_context_rejected_before_directory(tmp_path, runtime, invalid):
    trainer, optimizer, _ = runtime
    trainer.checkpoint_context = invalid
    with pytest.raises(RuntimeError, match="context"):
        checkpoint.save_checkpoint(trainer, tmp_path / "new", optimizer, step=7)
    assert not (tmp_path / "new").exists()


def test_peer_training_context_mismatch_prevents_publication(tmp_path, two_ranks):
    trainers, optimizers = _rank_trainers()
    trainers[1].checkpoint_context["data"]["revision"] = 2
    outcomes = two_ranks(
        lambda rank: checkpoint.save_checkpoint(
            trainers[rank], tmp_path / "new", optimizers[rank], step=7
        )
    )
    assert all(isinstance(result, RuntimeError) for result in outcomes)
    assert not (tmp_path / "new").exists()


def test_peer_training_context_load_failure_precedes_all_reads(
    tmp_path, two_ranks, monkeypatch
):
    trainers, optimizers = _rank_trainers()
    assert two_ranks(
        lambda rank: checkpoint.save_checkpoint(
            trainers[rank], tmp_path, optimizers[rank], step=7
        )
    ) == [None, None]
    trainers[1].checkpoint_context["model"]["width"] = 3
    read = Mock(side_effect=AssertionError("payload read"))
    monkeypatch.setattr(torch, "load", read)
    outcomes = two_ranks(
        lambda rank: checkpoint.load_checkpoint(
            trainers[rank], tmp_path, optimizers[rank]
        )
    )
    assert all(
        isinstance(result, RuntimeError) and "context" in str(result)
        for result in outcomes
    )
    read.assert_not_called()


def test_duplicate_manifest_context_key_rejected(tmp_path, runtime, monkeypatch):
    trainer, optimizer, _ = runtime
    checkpoint.save_checkpoint(trainer, tmp_path, optimizer, step=7)
    path = tmp_path / "manifest.json"
    path.write_text(path.read_text().replace('"width": 2', '"width": 1, "width": 2'))
    read = Mock()
    monkeypatch.setattr(torch, "load", read)
    with pytest.raises(RuntimeError, match="duplicate key"):
        checkpoint.load_checkpoint(trainer, tmp_path, optimizer)
    read.assert_not_called()


def test_load_agreement_distinguishes_boolean_and_integer_context(
    tmp_path, two_ranks, monkeypatch
):
    trainers, optimizers = _rank_trainers()
    assert two_ranks(
        lambda rank: checkpoint.save_checkpoint(
            trainers[rank], tmp_path, optimizers[rank], step=7
        )
    ) == [None, None]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    trainers[0].checkpoint_context["model"]["width"] = 1
    trainers[1].checkpoint_context["model"]["width"] = True
    real_read = Path.read_text

    def divergent_read(path, *args, **kwargs):
        if path.name == "manifest.json":
            return json.dumps(
                manifest | {"context": trainers[dist.get_rank()].checkpoint_context}
            )
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", divergent_read)
    load = Mock()
    monkeypatch.setattr(torch, "load", load)
    outcomes = two_ranks(
        lambda rank: checkpoint.load_checkpoint(
            trainers[rank], tmp_path, optimizers[rank]
        )
    )
    assert all(
        isinstance(result, RuntimeError) and "identity" in str(result)
        for result in outcomes
    )
    load.assert_not_called()


@pytest.mark.parametrize("change", ["type", "order", "groups"])
def test_optimizer_identity_rejected_before_any_mutation(
    tmp_path, runtime, monkeypatch, change
):
    trainer, _, _ = runtime
    trainer.module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    parameters = list(trainer.module.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": parameters[:2]}, {"params": parameters[2:]}], lr=0.01
    )
    trainer.module(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    checkpoint.save_checkpoint(trainer, tmp_path, optimizer, step=7)
    if change == "type":
        replacement = torch.optim.Adam(optimizer.param_groups, lr=0.01)
    elif change == "order":
        replacement = torch.optim.AdamW(
            [{"params": parameters[2:]}, {"params": parameters[:2]}], lr=0.01
        )
    else:
        replacement = torch.optim.AdamW(
            [{"params": parameters[::2]}, {"params": parameters[1::2]}], lr=0.01
        )
    model_load, optimizer_load, rng = Mock(), Mock(), Mock()
    monkeypatch.setattr(trainer.module, "load_state_dict", model_load)
    monkeypatch.setattr(replacement, "load_state_dict", optimizer_load)
    monkeypatch.setattr(torch, "set_rng_state", rng)
    with pytest.raises(RuntimeError, match="optimizer identity"):
        checkpoint.load_checkpoint(trainer, tmp_path, replacement)
    model_load.assert_not_called()
    optimizer_load.assert_not_called()
    rng.assert_not_called()


@pytest.mark.parametrize("invalid", ["external", "duplicate"])
def test_optimizer_invalid_ownership_rejected_before_writing(
    tmp_path, runtime, invalid
):
    trainer, optimizer, _ = runtime
    if invalid == "external":
        optimizer.param_groups[0]["params"].append(torch.nn.Parameter(torch.ones(2)))
    else:
        optimizer.param_groups[0]["params"].append(trainer.module.weight)
    with pytest.raises(RuntimeError, match="optimizer"):
        checkpoint.save_checkpoint(trainer, tmp_path, optimizer, step=7)
    assert not list(tmp_path.iterdir())


def test_peer_optimizer_identity_mismatch_prevents_all_mutation(
    tmp_path, two_ranks, monkeypatch
):
    trainers, optimizers = _rank_trainers()
    assert two_ranks(
        lambda rank: checkpoint.save_checkpoint(
            trainers[rank], tmp_path, optimizers[rank], step=7
        )
    ) == [None, None]
    optimizers[1] = torch.optim.Adam(trainers[1].module.parameters())
    rng = Mock()
    monkeypatch.setattr(torch, "set_rng_state", rng)
    for trainer, optimizer in zip(trainers, optimizers):
        trainer.module.state_dict = Mock(side_effect=AssertionError("state inspection"))
        trainer.module.load_state_dict = Mock()
        optimizer.load_state_dict = Mock()
    outcomes = two_ranks(
        lambda rank: checkpoint.load_checkpoint(
            trainers[rank], tmp_path, optimizers[rank]
        )
    )
    assert all(
        isinstance(result, RuntimeError) and "optimizer identity" in str(result)
        for result in outcomes
    )
    for trainer, optimizer in zip(trainers, optimizers):
        trainer.module.state_dict.assert_not_called()
        trainer.module.load_state_dict.assert_not_called()
        optimizer.load_state_dict.assert_not_called()
    rng.assert_not_called()


@pytest.mark.parametrize("invalid", ["external", "duplicate", "slots"])
def test_peer_optimizer_save_failure_precedes_model_capture(
    tmp_path, two_ranks, invalid
):
    trainers, optimizers = _rank_trainers()
    parameters = optimizers[1].param_groups[0]["params"]
    if invalid == "slots":
        state = optimizers[1].state_dict()
        state["param_groups"][0]["params"] = [0, 0]
        optimizers[1].state_dict = Mock(return_value=state)
    else:
        parameters.append(
            torch.nn.Parameter(torch.ones(2))
            if invalid == "external"
            else parameters[0]
        )
    for trainer in trainers:
        trainer.module.state_dict = Mock(side_effect=AssertionError("model capture"))
    directory = tmp_path / "not-created"
    outcomes = two_ranks(
        lambda rank: checkpoint.save_checkpoint(
            trainers[rank], directory, optimizers[rank], step=7
        )
    )
    assert all(
        isinstance(result, RuntimeError)
        and "rank 1" in str(result)
        and "optimizer" in str(result)
        for result in outcomes
    )
    for trainer in trainers:
        trainer.module.state_dict.assert_not_called()
    assert not directory.exists()


def test_optimizer_subset_and_tied_parameter_roundtrip(tmp_path, runtime):
    trainer, _, _ = runtime
    trainer.module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    trainer.module[1].weight = trainer.module[0].weight
    selected = [trainer.module[1].weight, trainer.module[1].bias]
    optimizer = torch.optim.AdamW(selected, lr=0.03)
    trainer.module(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    checkpoint.save_checkpoint(trainer, tmp_path, optimizer, step=7)
    replacement = torch.optim.AdamW(selected, lr=0.7)
    assert checkpoint.load_checkpoint(trainer, tmp_path, replacement) == 7
    assert replacement.param_groups[0]["lr"] == 0.03
    for parameter in selected:
        for name, value in optimizer.state[parameter].items():
            torch.testing.assert_close(
                replacement.state[parameter][name], value, rtol=0, atol=0
            )
    identity = torch.load(tmp_path / "rank-00000.pt", weights_only=False)[
        "optimizer_identity"
    ]
    assert identity["parameter_groups"] == [["0.weight", "1.bias"]]


@pytest.mark.parametrize(
    "identity", [None, {}, {"type": "bad", "parameter_groups": []}]
)
def test_malformed_optimizer_identity_rejected_before_mutation(
    tmp_path, runtime, monkeypatch, identity
):
    trainer, optimizer, _ = runtime
    write_checkpoint(
        tmp_path,
        trainer,
        optimizer,
        payload_edit=lambda payload: payload.update(optimizer_identity=identity),
    )
    load = Mock()
    monkeypatch.setattr(trainer.module, "load_state_dict", load)
    with pytest.raises(RuntimeError, match="optimizer identity"):
        checkpoint.load_checkpoint(trainer, tmp_path, optimizer)
    load.assert_not_called()


@pytest.mark.parametrize("damage", ["duplicate", "boolean", "orphan_state"])
def test_optimizer_serialized_ids_rejected_before_mutation(
    tmp_path, runtime, monkeypatch, damage
):
    trainer, optimizer, _ = runtime

    def edit(payload):
        state = payload["optimizer"]
        if damage == "duplicate":
            state["param_groups"][0]["params"] = [0, 0]
        elif damage == "boolean":
            state["param_groups"][0]["params"] = [False, 1]
        else:
            state["state"][99] = {}

    write_checkpoint(tmp_path, trainer, optimizer, payload_edit=edit)
    load = Mock()
    monkeypatch.setattr(trainer.module, "load_state_dict", load)
    with pytest.raises(RuntimeError, match="optimizer"):
        checkpoint.load_checkpoint(trainer, tmp_path, optimizer)
    load.assert_not_called()


def test_optimizer_warm_restart_preserves_next_update(tmp_path, runtime):
    trainer, _, _ = runtime
    optimizer = torch.optim.AdamW(trainer.module.parameters(), lr=0.02)

    def update(module, opt):
        opt.zero_grad(set_to_none=True)
        module(torch.tensor([[1.0, 2.0], [3.0, 4.0]])).square().sum().backward()
        opt.step()

    update(trainer.module, optimizer)
    checkpoint.save_checkpoint(trainer, tmp_path, optimizer, step=1)
    update(trainer.module, optimizer)
    expected = {
        name: value.clone() for name, value in trainer.module.state_dict().items()
    }
    restarted = SimpleNamespace(**vars(trainer))
    restarted.module = torch.nn.Linear(2, 2)
    restored = torch.optim.AdamW(restarted.module.parameters(), lr=0.9)
    assert checkpoint.load_checkpoint(restarted, tmp_path, restored) == 1
    update(restarted.module, restored)
    for name, value in restarted.module.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    for before, after in zip(optimizer.state.values(), restored.state.values()):
        for name in before:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


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
        lambda m: m.update(schema_version=1),
        lambda m: m.update(schema_version=2),
        lambda m: m.update(schema_version=3),
        lambda m: m["topology"].pop("pipeline_schedule"),
        lambda m: m["topology"].update(pipeline_schedule=True),
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
