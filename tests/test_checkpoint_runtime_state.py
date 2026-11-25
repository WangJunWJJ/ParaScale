# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Rank-local runtime checkpoint and replay regression tests."""

import copy
import random
import sys
from types import SimpleNamespace

import pytest

from parascale.checkpoint import CheckpointManager
from parascale.runtime.training.checkpointing import CheckpointController
from parascale.runtime.training.data_state import DataResumeController


class FakeTorch:
    def __init__(self):
        self.cpu = [1]
        self.npu_state = [[2]]
        self.cuda = SimpleNamespace(is_available=lambda: False)
        self.npu = SimpleNamespace(
            is_available=lambda: True,
            get_rng_state_all=lambda: copy.deepcopy(self.npu_state),
            set_rng_state_all=lambda state: setattr(self, "npu_state", state),
        )
        self.payloads = {}

    def get_rng_state(self):
        return self.cpu[:]

    def set_rng_state(self, state):
        self.cpu = state[:]

    def save(self, payload, path):
        token = str(len(self.payloads))
        self.payloads[token] = copy.deepcopy(payload)
        path.write(token.encode())

    def load(self, path, *, map_location, weights_only):
        assert map_location == "cpu" and weights_only is True
        return copy.deepcopy(self.payloads[path.read_text()])


def engine(rank=0, world_size=1):
    precision = SimpleNamespace(value={"amp_scaler": {"scale": 128.0}})
    precision.state_dict = lambda: copy.deepcopy(precision.value)
    precision.load_state_dict = lambda state: setattr(precision, "value", state)
    result = SimpleNamespace(
        precision=precision,
        config=SimpleNamespace(training_backend="native"),
        state=SimpleNamespace(
            global_step=3,
            last_metrics={},
            consumed_samples=6,
            consumed_micro_batches=3,
            data_state={},
        ),
        training_backend=SimpleNamespace(
            name="native", state_dict=lambda: {}, load_state_dict=lambda state: None
        ),
        _distributed_rank=lambda: rank,
        _distributed_world_size=lambda: world_size,
        _distributed_barrier=lambda: None,
        plan=lambda: SimpleNamespace(to_dict=lambda: {}),
    )
    result.data_resume = DataResumeController(result)
    return result


def test_full_rng_roundtrip_fake_npu():
    torch = FakeTorch()
    random.seed(12)
    state = CheckpointController._capture_rng_state(torch)
    expected = random.random()
    torch.cpu = [9]
    torch.npu_state = [[9]]
    CheckpointController._restore_rng_state(torch, state)
    assert random.random() == expected
    assert torch.cpu == [1]
    assert torch.npu_state == [[2]]


def test_rank_sidecars_and_precision_roundtrip(tmp_path, monkeypatch):
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    manager = CheckpointManager(str(tmp_path))
    worker = engine(1, 2)
    worker.data_resume.active_dataloader = SimpleNamespace(
        state_dict=lambda: {"rank": 1, "position": 6}
    )
    torch.cpu = [11]
    CheckpointController(worker).save(manager)
    assert manager.payload_path(3, "rank-00001/runtime_state.pt").is_file()
    assert not manager.payload_path(3, "backend_state.pt").exists()
    leader = engine(0, 2)
    torch.cpu = [22]
    CheckpointController(leader).save(manager)
    manifest = manager.read_manifest(3)
    entries = [entry for entry in manifest.files if entry["role"] == "runtime_state"]
    assert {entry["rank"] for entry in entries} == {0, 1}
    assert all("sha256" in entry for entry in entries)
    worker.precision.value = {}
    CheckpointController(worker).load(manager, 3)
    assert torch.cpu == [11]
    assert worker.precision.value == {"amp_scaler": {"scale": 128.0}}
    assert worker.state.data_state["state"] == {"rank": 1, "position": 6}


@pytest.mark.parametrize(
    "key,value",
    [
        ("precision", "bf16"),
        ("gradient_accumulation_steps", 2),
        ("grad_clip_norm", 1.0),
        ("zero_stage", 1),
        ("ddp_find_unused_parameters", True),
        ("ddp_static_graph", True),
        ("ddp_gradient_as_bucket_view", False),
        ("ddp_bucket_cap_mb", 64),
        ("ddp_comm_hook", "fp16_compress"),
        ("training_backend", "native"),
    ],
)
def test_native_policy_drift_rejected_before_loading_state(
    tmp_path, monkeypatch, key, value
):
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    manager = CheckpointManager(str(tmp_path))
    target = engine()
    target.config.training_backend = "native_ddp"
    target.training_backend.name = "native_ddp"
    controller = CheckpointController(target)
    controller.save(manager)
    setattr(target.config, key, value)
    target.training_backend.load_state_dict = lambda _: pytest.fail(
        "must not load backend"
    )
    torch.load = lambda *_args, **_kwargs: pytest.fail("must not load tensors or RNG")
    target.state.global_step = 99
    with pytest.raises(ValueError, match="training policy mismatch"):
        controller.load(manager, 3)
    assert target.state.global_step == 99


def test_native_checkpoint_requires_recorded_policy(tmp_path, monkeypatch):
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    manager = CheckpointManager(str(tmp_path))
    controller = CheckpointController(engine())
    controller.save(manager)
    manifest = manager.read_manifest(3)
    assert manifest.metadata.pop("native_training_policy")["version"] == 1
    manager.write_manifest(manifest)
    with pytest.raises(ValueError, match="missing native training policy"):
        controller.load(manager, 3)


def test_native_policy_allows_retention_and_metrics_changes(tmp_path, monkeypatch):
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    manager = CheckpointManager(str(tmp_path))
    target = engine()
    controller = CheckpointController(target)
    controller.save(manager)
    target.config.checkpoint_keep_last = 5
    target.config.metrics_history_limit = 10
    assert controller.load(manager, 3).global_step == 3


def test_native_policy_uses_effective_hook_and_static_graph():
    from parascale.runtime.backends.native import NativeDdpTrainingBackend

    config = SimpleNamespace(
        precision="bf16",
        ddp_comm_hook="auto",
        task_type="vision",
        model_family="clip",
        gradient_accumulation_steps=2,
        ddp_static_graph=True,
    )
    backend = NativeDdpTrainingBackend(config=config)
    automatic = backend.checkpoint_policy()
    config.ddp_comm_hook = automatic["ddp"]["comm_hook"]
    config.ddp_static_graph = False
    assert backend.checkpoint_policy() == automatic


@pytest.mark.parametrize("name", ["fp16_compress", "bf16_compress"])
def test_native_registration_preserves_official_hook_and_default_group(name):
    pytest.importorskip("torch")
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

    from parascale.communication.hooks import get_ddp_comm_hook
    from parascale.runtime.backends.native import NativeDdpTrainingBackend

    hook = getattr(default_hooks, f"{name}_hook")
    assert get_ddp_comm_hook(name) is hook
    calls = []
    model = SimpleNamespace(register_comm_hook=lambda **kwargs: calls.append(kwargs))
    backend = NativeDdpTrainingBackend(config=SimpleNamespace(ddp_comm_hook=name))
    backend._register_comm_hook(model)
    assert calls == [{"state": None, "hook": hook}]
    assert backend.checkpoint_policy()["ddp"]["comm_hook"] == name


@pytest.mark.parametrize("action", ["save", "load"])
def test_active_native_policy_drift_rejected_before_io(tmp_path, monkeypatch, action):
    from parascale.runtime.backends.native import NativeDdpTrainingBackend

    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    target = engine()
    target.config.training_backend = "native_ddp"
    target.training_backend = NativeDdpTrainingBackend(config=target.config)
    target.training_backend.setup()
    manager = CheckpointManager(str(tmp_path))
    controller = CheckpointController(target)
    controller.save(manager)
    target.config.ddp_find_unused_parameters = True
    monkeypatch.setattr(
        controller, "_save_runtime_state", lambda *_: pytest.fail("must not write")
    )
    torch.load = lambda *_args, **_kwargs: pytest.fail("must not read state")
    with pytest.raises(ValueError, match="policy changed after backend setup"):
        getattr(controller, action)(manager, 3)


@pytest.mark.parametrize("action", ["save", "load"])
@pytest.mark.parametrize("drift", ["replacement_config", "non_native_backend"])
def test_engine_policy_and_active_backend_must_agree(
    tmp_path, monkeypatch, action, drift
):
    from parascale.runtime.backends.native import NativeTrainingBackend

    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    target = engine()
    target.training_backend = NativeTrainingBackend(config=target.config)
    target.training_backend.setup()
    manager = CheckpointManager(str(tmp_path))
    controller = CheckpointController(target)
    controller.save(manager)
    if drift == "replacement_config":
        target.config = copy.deepcopy(target.config)
        target.config.grad_clip_norm = 1.0
    else:
        target.training_backend = SimpleNamespace(name="fsdp")
    monkeypatch.setattr(
        controller, "_save_runtime_state", lambda *_: pytest.fail("must not write")
    )
    torch.load = lambda *_args, **_kwargs: pytest.fail("must not read state")
    with pytest.raises(ValueError, match="training policy mismatch"):
        getattr(controller, action)(manager, 3)


@pytest.mark.parametrize("exhausted", [False, True])
def test_replay_preserves_random_even_on_failure(exhausted):
    target = engine()
    target.state.data_state = {
        "resume_mode": "replay_skip",
        "consumed_micro_batches": 2,
    }

    class Source:
        def __iter__(self):
            random.random()
            return self

        def __next__(self):
            random.random()
            if exhausted:
                raise StopIteration
            return 1

    random.seed(31)
    before = random.getstate()
    with pytest.warns(RuntimeWarning, match="deterministic"):
        if exhausted:
            with pytest.raises(RuntimeError, match="exhausted"):
                target.data_resume.prepare_iterator(Source())
        else:
            target.data_resume.prepare_iterator(Source())
    assert random.getstate() == before


def test_real_cpu_weights_only_roundtrip_and_dropout(tmp_path):
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    manager = CheckpointManager(str(tmp_path))
    target = engine()
    torch.manual_seed(51)
    numpy.random.seed(51)
    random.seed(51)
    CheckpointController(target).save(manager)
    expected = (
        random.random(),
        numpy.random.random(),
        torch.nn.functional.dropout(torch.ones(64), training=True),
    )
    target.precision.value = {}
    CheckpointController(target).load(manager, 3)
    assert random.random() == expected[0]
    assert numpy.random.random() == expected[1]
    assert torch.equal(
        torch.nn.functional.dropout(torch.ones(64), training=True), expected[2]
    )
    assert target.precision.value["amp_scaler"]["scale"] == 128.0


@pytest.mark.parametrize("corruption", ["rank", "missing_entry", "duplicate_entry"])
def test_runtime_identity_validated_before_backend_load(
    tmp_path, monkeypatch, corruption
):
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    manager = CheckpointManager(str(tmp_path))
    target = engine()
    CheckpointController(target).save(manager)
    if corruption == "rank":
        torch.payloads[
            manager.payload_path(3, "rank-00000/runtime_state.pt").read_text()
        ]["rank"] = 9
    else:
        manifest = manager.read_manifest(3)
        entry = next(
            entry for entry in manifest.files if entry["role"] == "runtime_state"
        )
        if corruption == "missing_entry":
            manifest.files.remove(entry)
        else:
            manifest.files.append(entry)
        manager.write_manifest(manifest)
    target.training_backend.load_state_dict = lambda state: pytest.fail(
        "backend must not load"
    )
    with pytest.raises(ValueError, match="runtime_state"):
        CheckpointController(target).load(manager, 3)


def test_missing_precision_methods_are_supported(tmp_path, monkeypatch):
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    manager = CheckpointManager(str(tmp_path))
    target = engine()
    target.precision = object()
    CheckpointController(target).save(manager)
    CheckpointController(target).load(manager, 3)


def test_runtime_failure_is_synchronized_before_raising(tmp_path, monkeypatch):
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    target = engine()
    calls = []
    target._distributed_barrier = lambda: calls.append("sync")

    def fail():
        raise ValueError("bad precision")

    target.precision.state_dict = fail
    with pytest.raises(ValueError, match="bad precision"):
        CheckpointController(target).save(CheckpointManager(str(tmp_path)))
    assert calls == ["sync", "sync"]  # Policy agreement, then failed sidecar phase.


def test_peer_failure_stops_checkpoint_phase(monkeypatch):
    distributed = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 2,
        all_gather_object=lambda errors, local: errors.__setitem__(1, "sidecar failed"),
    )
    torch = FakeTorch()
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    with pytest.raises(RuntimeError, match="another rank.*sidecar failed"):
        CheckpointController(engine())._collective_call(lambda: None)


def test_rng_without_numpy_and_with_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)
    torch = FakeTorch()
    torch.cuda = torch.npu
    state = CheckpointController._capture_rng_state(torch)
    assert "numpy" not in state
    assert state["torch_cuda"] == [[2]]
    torch.npu_state = [[9]]
    CheckpointController._restore_rng_state(torch, state)
    assert torch.npu_state == [[2]]


def test_real_cpu_continuous_vs_resumed_scaler_training(tmp_path):
    torch = pytest.importorskip("torch")
    from parascale.runtime.training.precision import PrecisionController

    target = engine()
    target.config.precision = "fp16"
    target.amp_scaler = torch.amp.GradScaler("cpu", init_scale=32, growth_interval=1)
    target.precision = PrecisionController(target)
    model = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(8, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    target.training_backend.state_dict = lambda: {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    def restore(state):
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])

    target.training_backend.load_state_dict = restore

    def update():
        optimizer.zero_grad()
        loss = model(torch.ones(4, 8)).square().mean()
        target.amp_scaler.scale(loss).backward()
        target.amp_scaler.step(optimizer)
        target.amp_scaler.update()

    update()
    manager = CheckpointManager(str(tmp_path))
    CheckpointController(target).save(manager)
    update()
    expected = copy.deepcopy(model.state_dict())
    expected_scaler = copy.deepcopy(target.amp_scaler.state_dict())
    CheckpointController(target).load(manager, 3)
    target.precision.setup_scaler()
    target.precision.setup_scaler()
    update()
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in expected.items()
    )
    assert target.amp_scaler.state_dict() == expected_scaler


@pytest.mark.parametrize("payload", ["backend_state.pt", "rank-00000/runtime_state.pt"])
def test_real_serialization_failure_preserves_payload(tmp_path, monkeypatch, payload):
    torch = pytest.importorskip("torch")
    manager = CheckpointManager(str(tmp_path))
    controller = CheckpointController(engine())
    controller.save(manager)
    path = manager.payload_path(3, payload)
    before = path.read_bytes()

    def fail_save(value, stream):
        stream.write(b"partial serialization")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        if payload == "backend_state.pt":
            controller._save_portable_backend_state(manager, 3, None)
        else:
            controller._save_runtime_state(manager, 3)
    assert path.read_bytes() == before
    assert manager.validate(3).ok
    assert not list(path.parent.glob(".*.tmp"))


def test_native_load_failure_is_synchronized_before_state_advances(
    tmp_path, monkeypatch
):
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    target = engine()
    manager = CheckpointManager(str(tmp_path))
    controller = CheckpointController(target)
    controller.save(manager)
    target.state.global_step = 0
    barriers = []
    target._distributed_barrier = lambda: barriers.append("sync")

    def fail_load(state):
        raise OSError("local load failed")

    target.training_backend.load_state_dict = fail_load
    with pytest.raises(OSError, match="local load failed"):
        controller.load(manager, 3)
    assert barriers == ["sync", "sync"]
    assert target.state.global_step == 0
