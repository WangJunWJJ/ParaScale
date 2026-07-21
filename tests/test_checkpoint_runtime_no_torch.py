# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import tempfile
from pathlib import Path

import pytest

from parascale import (
    CheckpointManager,
    CheckpointManifest,
    ParaScaleConfig,
    TrainEngine,
)


class _BackendSpecificCheckpoint:
    def __init__(self, name):
        self.name = name
        self.saved = None
        self.loaded = None

    def save_checkpoint(
        self, checkpoint_manager, step=None, client_state=None, **kwargs
    ):
        self.saved = {"step": step, "client_state": dict(client_state or {})}
        role = "fsdp_state" if self.name == "fsdp" else "deepspeed_checkpoint"
        path = "fsdp_state.pt" if self.name == "fsdp" else "deepspeed"
        payload = checkpoint_manager.payload_path(int(step), path)
        if self.name == "fsdp":
            payload.parent.mkdir(parents=True, exist_ok=True)
            payload.write_bytes(b"fsdp-state")
        else:
            payload.mkdir(parents=True, exist_ok=True)
        return {
            "files": [{"path": path, "role": role, "format": self.name}],
            "metadata": {"backend_checkpoint": self.name},
        }

    def load_checkpoint(self, checkpoint_manager, step=None, **kwargs):
        self.loaded = {"step": step}
        return {"last_metrics": {"loss": 0.25}, "scheduler_state_dict": {"epoch": 3}}

class _CheckpointScheduler:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"epoch": 2}

    def load_state_dict(self, state):
        self.loaded = dict(state)

def test_train_engine_delegates_backend_specific_checkpoints_without_torch():
    root = (
        Path(tempfile.gettempdir())
        / "parascale-test-runs"
        / "backend_specific_checkpoint"
    )

    for backend_name, expected_role in [
        ("fsdp", "fsdp_state"),
        ("deepspeed", "deepspeed_checkpoint"),
    ]:
        manager = CheckpointManager(str(root / backend_name))
        backend = _BackendSpecificCheckpoint(backend_name)
        scheduler = _CheckpointScheduler()
        train = TrainEngine(
            config=ParaScaleConfig(training_backend=backend_name),
            model_profile={
                "total_params": 1,
                "total_memory": 1,
                "num_layers": 1,
                "model_type": "toy",
            },
            hardware_profile={
                "num_gpus": 1,
                "gpu_memory": 1,
                "available_memory": 1,
                "gpus_per_node": 1,
            },
            training_backend=backend,
        )
        train.state.global_step = 4
        train.state.last_metrics = {"loss": 0.5}

        train.save_checkpoint(manager, scheduler=scheduler)
        manifest = manager.read_manifest(4)

        assert backend.saved["step"] == 4
        assert backend.saved["client_state"]["global_step"] == 4
        assert backend.saved["client_state"]["scheduler_state_dict"] == {"epoch": 2}
        assert manifest.backend == backend_name
        assert manifest.files[0]["role"] == expected_role
        assert manifest.metadata["backend_specific_checkpoint"] is True
        assert manifest.metadata["backend_checkpoint"] == backend_name

        restored = train.load_checkpoint(manager, 4, scheduler=scheduler)

        assert backend.loaded == {"step": 4}
        assert scheduler.loaded == {"epoch": 3}
        assert restored.metadata["backend_state_loaded"] is True

def test_checkpoint_resume_passes_manifest_fsdp_state_dict_type():
    class Backend:
        name = "fsdp"

        def __init__(self):
            self.loaded = None

        def load_checkpoint(self, _manager, step=None, **kwargs):
            self.loaded = {"step": step, **kwargs}
            return {}

    root = Path(tempfile.gettempdir()) / "parascale-test-runs" / "manifest_format"
    manager = CheckpointManager(str(root))
    payload = manager.payload_path(5, "rank-00000/fsdp_state.pt")
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(b"sharded-state")
    manager.write_manifest(
        CheckpointManifest(
            step=5,
            backend="fsdp",
            files=[
                {
                    "path": "rank-00000/fsdp_state.pt",
                    "role": "fsdp_state",
                    "state_dict_type": "sharded",
                    "rank": 0,
                }
            ],
            metadata={"rank_count": 1},
        )
    )
    backend = Backend()
    train = TrainEngine(
        config=ParaScaleConfig(
            training_backend="fsdp",
            fsdp_state_dict_type="full",
        ),
        training_backend=backend,
    )

    train.load_checkpoint(manager, 5)

    assert backend.loaded == {"step": 5, "state_dict_type": "sharded"}

def test_fsdp_load_checkpoint_uses_manifest_format_for_payload_path(
    monkeypatch,
):
    import parascale.runtime.backends.fsdp as fsdp_module
    from parascale.runtime.backends.fsdp import FSDPTrainingBackend

    root = Path(tempfile.gettempdir()) / "parascale-test-runs" / "fsdp_load_path"
    manager = CheckpointManager(str(root))
    payload_path = manager.payload_path(3, "rank-00001/fsdp_state.pt")
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(b"payload")
    loaded_paths = []

    class TorchStub:
        @staticmethod
        def load(path, **_kwargs):
            loaded_paths.append(path)
            return {"backend_state": {}, "client_state": {}}

    backend = FSDPTrainingBackend(
        config=ParaScaleConfig(
            training_backend="fsdp",
            fsdp_state_dict_type="full",
        ),
        local_rank=1,
    )
    backend._rank = lambda: 1
    monkeypatch.setattr(fsdp_module, "_require_torch", lambda: TorchStub())

    backend.load_checkpoint(manager, step=3, state_dict_type="sharded")

    assert loaded_paths == [payload_path]

def test_fsdp_load_state_dict_uses_saved_format_context(monkeypatch):
    from contextlib import contextmanager

    from parascale.runtime.backends.fsdp import FSDPTrainingBackend

    events = []

    class Model:
        def load_state_dict(self, state):
            events.append(("load", state))

    @contextmanager
    def state_dict_context(state_type, *, rank0_only):
        events.append(("enter", state_type, rank0_only))
        yield
        events.append(("exit", state_type, rank0_only))

    backend = FSDPTrainingBackend(
        model=Model(),
        config=ParaScaleConfig(training_backend="fsdp"),
    )
    monkeypatch.setattr(
        backend,
        "_fsdp_state_dict_context",
        state_dict_context,
        raising=False,
    )

    backend.load_state_dict(
        {
            "model_state_dict": {"weight": "shard"},
            "state_dict_type": "sharded",
        }
    )

    assert events == [
        ("enter", "sharded", False),
        ("load", {"weight": "shard"}),
        ("exit", "sharded", False),
    ]

def test_checkpoint_resume_replays_to_consumed_data_position():
    root = Path(tempfile.gettempdir()) / "parascale-test-runs" / "data_replay_resume"
    manager = CheckpointManager(str(root))
    batches = [
        {"sample_id": "a", "num_images": 2},
        {"sample_id": "b", "num_images": 2},
        {"sample_id": "c", "num_images": 2},
    ]
    backend = _BackendSpecificCheckpoint("fsdp")
    train = TrainEngine(
        config=ParaScaleConfig(training_backend="fsdp"),
        training_backend=backend,
    )
    train.state.initialized = True
    seen = []
    train.fit(
        batches,
        max_steps=2,
        step_fn=lambda batch: seen.append(batch["sample_id"]),
        checkpoint_manager=manager,
        checkpoint_interval=2,
    )

    manifest = manager.read_manifest(2)

    assert seen == ["a", "b"]
    assert manifest.consumed_samples == 4
    assert manifest.data_state["consumed_micro_batches"] == 2
    assert manifest.data_state["resume_mode"] == "replay_skip"

    restored = TrainEngine(
        config=ParaScaleConfig(training_backend="fsdp"),
        training_backend=_BackendSpecificCheckpoint("fsdp"),
    )
    restored.state.initialized = True
    restored.load_checkpoint(manager, 2)
    resumed_seen = []
    with pytest.warns(RuntimeWarning, match="replaying and skipping 2"):
        restored.fit(
            batches,
            max_steps=1,
            step_fn=lambda batch: resumed_seen.append(batch["sample_id"]),
        )

    assert resumed_seen == ["c"]

def test_checkpoint_replay_resume_rejects_insufficient_finite_data_window():
    restored = TrainEngine(
        config=ParaScaleConfig(training_backend="fsdp"),
        training_backend=_BackendSpecificCheckpoint("fsdp"),
    )
    restored.state.initialized = True
    restored.state.data_state = {
        "resume_mode": "replay_skip",
        "consumed_micro_batches": 2,
    }

    with pytest.raises(ValueError, match="requires 4 micro-batches"):
        restored.fit(
            [{"sample_id": "a"}, {"sample_id": "b"}, {"sample_id": "c"}],
            max_steps=2,
            step_fn=lambda _batch: None,
        )

def test_checkpoint_resume_restores_stateful_dataloader_without_replay():
    class StatefulLoader:
        def __init__(self, batches):
            self.batches = list(batches)
            self.cursor = 0
            self.loaded_state = None

        def __iter__(self):
            while self.cursor < len(self.batches):
                batch = self.batches[self.cursor]
                self.cursor += 1
                yield batch

        def __len__(self):
            return len(self.batches) - self.cursor

        def state_dict(self):
            return {"cursor": self.cursor}

        def load_state_dict(self, state):
            self.loaded_state = dict(state)
            self.cursor = int(state["cursor"])

    root = Path(tempfile.gettempdir()) / "parascale-test-runs" / "data_state_resume"
    manager = CheckpointManager(str(root))
    batches = [
        {"sample_id": "a", "num_images": 1},
        {"sample_id": "b", "num_images": 1},
        {"sample_id": "c", "num_images": 1},
    ]
    loader = StatefulLoader(batches)
    train = TrainEngine(
        config=ParaScaleConfig(training_backend="fsdp"),
        training_backend=_BackendSpecificCheckpoint("fsdp"),
    )
    train.state.initialized = True
    train.fit(
        loader,
        max_steps=2,
        step_fn=lambda _batch: None,
        checkpoint_manager=manager,
        checkpoint_interval=2,
    )

    manifest = manager.read_manifest(2)

    assert manifest.data_state == {
        "consumed_micro_batches": 2,
        "resume_mode": "state_dict",
        "state": {"cursor": 2},
        "target": "dataloader",
    }

    restored_loader = StatefulLoader(batches)
    restored = TrainEngine(
        config=ParaScaleConfig(training_backend="fsdp"),
        training_backend=_BackendSpecificCheckpoint("fsdp"),
    )
    restored.state.initialized = True
    restored.load_checkpoint(manager, 2)
    resumed_seen = []
    restored.fit(
        restored_loader,
        max_steps=1,
        step_fn=lambda batch: resumed_seen.append(batch["sample_id"]),
    )

    assert restored_loader.loaded_state == {"cursor": 2}
    assert resumed_seen == ["c"]

def test_checkpoint_manifest_round_trip():
    root = Path(tempfile.gettempdir()) / "parascale-test-runs" / "checkpoint_manifest"
    manager = CheckpointManager(str(root))
    manifest = CheckpointManifest(
        step=7,
        shards=["rank0.pt"],
        backend="fsdp",
        consumed_samples=128,
        consumed_tokens=4096,
        parallel_plan={"dp": 2},
        files=[{"path": "rank0.pt", "role": "model", "dtype": "bf16"}],
        metadata={"note": "roundtrip"},
    )

    path = manager.write_manifest(manifest)
    restored = manager.read_manifest(7)

    assert path.name == "manifest.json"
    assert restored.step == 7
    assert restored.global_step == 7
    assert restored.shards == ["rank0.pt"]
    assert restored.backend == "fsdp"
    assert restored.consumed_samples == 128
    assert restored.consumed_tokens == 4096
    assert restored.parallel_plan == {"dp": 2}
    assert restored.files[0]["role"] == "model"

def test_checkpoint_manager_adds_checksums_and_validator_detects_corruption():
    root = Path(tempfile.gettempdir()) / "parascale-test-runs" / "checkpoint_validator"
    manager = CheckpointManager(str(root))
    payload = manager.payload_path(2, "rank0.pt")
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(b"checkpoint-payload")
    (payload.parent / "deepspeed").mkdir(exist_ok=True)
    manifest = CheckpointManifest(
        step=2,
        files=[
            {"path": "rank0.pt", "role": "backend_state", "format": "torch"},
            {
                "path": "deepspeed",
                "role": "deepspeed_checkpoint",
                "format": "deepspeed",
            },
        ],
    )

    manager.write_manifest(manifest)
    restored = manager.read_manifest(2)
    file_entry = restored.files[0]

    assert file_entry["size_bytes"] == len(b"checkpoint-payload")
    assert len(file_entry["sha256"]) == 64
    assert restored.files[1]["entry_type"] == "directory"
    report = manager.validate_manifest(restored)
    assert report.ok is True
    assert report.checked_files == 1
    assert report.checked_directories == 1

    payload.write_bytes(b"corrupted")
    report = manager.validate(2)

    assert report.ok is False
    assert report.checksum_mismatches[0]["path"] == "rank0.pt"
    assert report.size_mismatches[0]["path"] == "rank0.pt"

def test_checkpoint_validator_fails_backend_checkpoint_errors():
    root = Path(tempfile.gettempdir()) / "parascale-test-runs" / "checkpoint_error"
    manager = CheckpointManager(str(root))
    manifest = CheckpointManifest(
        step=5,
        files=[
            {
                "path": "fsdp_state.pt",
                "role": "fsdp_state",
                "error": "rank0 save failed",
            }
        ],
        metadata={
            "backend_state_written": False,
            "backend_checkpoint_error": "rank0 save failed",
        },
    )

    report = manager.validate_manifest(manifest)

    assert report.ok is False
    assert any("backend_checkpoint_error" in item for item in report.errors)
    assert any("backend_state_written" in item for item in report.errors)

def test_checkpoint_manifest_validation_rejects_negative_counters():
    try:
        CheckpointManifest(step=1, consumed_tokens=-1)
    except ValueError as exc:
        assert "consumed_tokens" in str(exc)
    else:
        raise AssertionError("negative consumed_tokens must be rejected")
