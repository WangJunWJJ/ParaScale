# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import tempfile
from pathlib import Path

import pytest

from parascale import CheckpointManager, CheckpointManifest, ParaScaleConfig


def test_checkpoint_controller_rejects_corruption_before_backend_setup():
    from parascale.runtime.training.checkpointing import CheckpointController

    root = Path(tempfile.gettempdir()) / "parascale-test-runs" / "resume_corrupt"
    manager = CheckpointManager(str(root))
    payload = manager.payload_path(3, "backend_state.pt")
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(b"valid-state")
    manager.write_manifest(
        CheckpointManifest(
            step=3,
            backend="native",
            files=[{"path": payload.name, "role": "backend_state"}],
            metadata={"world_size": 1},
        )
    )
    payload.write_bytes(b"corrupted")

    class Engine:
        config = type("Config", (), {"training_backend": "native"})()
        training_backend = None
        state = type("State", (), {"global_step": 0, "last_metrics": {}})()

        def _distributed_world_size(self):
            return 1

    with pytest.raises(RuntimeError, match="validation failed before resume"):
        CheckpointController(Engine()).load(manager, 3)

def test_checkpoint_controller_rejects_world_size_change_by_default():
    from parascale.runtime.training.checkpointing import CheckpointController

    root = Path(tempfile.gettempdir()) / "parascale-test-runs" / "resume_world_size"
    manager = CheckpointManager(str(root))
    manager.write_manifest(
        CheckpointManifest(
            step=4,
            backend="native_ddp",
            metadata={"world_size": 2, "rank_count": 2},
        )
    )

    class Engine:
        config = type("Config", (), {"training_backend": "native_ddp"})()
        training_backend = None
        state = type("State", (), {"global_step": 0, "last_metrics": {}})()

        def _distributed_world_size(self):
            return 1

    with pytest.raises(ValueError, match="world_size mismatch.*checkpoint=2.*runtime=1"):
        CheckpointController(Engine()).load(manager, 4)

def test_checkpoint_controller_skips_manifest_write_on_nonzero_rank_without_torch():
    from parascale.runtime.training.checkpointing import CheckpointController

    class State:
        global_step = 3
        last_metrics = {"loss": 1.0}

    class Backend:
        name = "native"

        def state_dict(self):
            return {"backend": "native"}

    class Engine:
        state = State()
        training_backend = Backend()
        config = ParaScaleConfig(training_backend="native")

        def _distributed_rank(self):
            return 1

        def _distributed_world_size(self):
            return 2

        def _distributed_barrier(self):
            self.barrier_called = True

        def plan(self):
            class Plan:
                def to_dict(self):
                    return {"backend": "native"}

            return Plan()

    class Manager:
        def __init__(self):
            self.written = False

        def payload_path(self, *_args):
            return Path(tempfile.gettempdir()) / "unused.pt"

        def write_manifest(self, _manifest):
            self.written = True
            raise AssertionError("nonzero rank must not write manifest")

    engine = Engine()
    manager = Manager()

    result = CheckpointController(engine).save(manager)

    assert result["skipped"] is True
    assert result["rank"] == 1
    assert result["world_size"] == 2
    assert manager.written is False
    assert engine.barrier_called is True

def test_checkpoint_controller_deepspeed_all_ranks_save_rank0_manifest_only():
    from parascale.runtime.training.checkpointing import CheckpointController

    class State:
        global_step = 4
        last_metrics = {"loss": 0.5}

    class Backend:
        name = "deepspeed"

        def __init__(self):
            self.saved = False

        def save_checkpoint(self, _manager, step=None, client_state=None):
            self.saved = True
            return {
                "files": [
                    {
                        "path": "deepspeed",
                        "role": "deepspeed_checkpoint",
                        "format": "deepspeed",
                        "tag": f"global_step{step}",
                    }
                ],
                "metadata": {"backend_checkpoint": "deepspeed"},
            }

    class Engine:
        def __init__(self):
            self.state = State()
            self.training_backend = Backend()
            self.config = ParaScaleConfig(training_backend="deepspeed")
            self.barrier_called = False

        def _distributed_rank(self):
            return 1

        def _distributed_world_size(self):
            return 2

        def _distributed_barrier(self):
            self.barrier_called = True

        def plan(self):
            class Plan:
                def to_dict(self):
                    return {"backend": "deepspeed"}

            return Plan()

    class Manager:
        def payload_path(self, *_args):
            return Path(tempfile.gettempdir()) / "unused.pt"

        def write_manifest(self, _manifest):
            raise AssertionError("nonzero DeepSpeed rank must not write manifest")

    engine = Engine()

    result = CheckpointController(engine).save(Manager())

    assert engine.training_backend.saved is True
    assert result["skipped"] is True
    assert result["rank"] == 1
    assert result["files"][0]["role"] == "deepspeed_checkpoint"
    assert engine.barrier_called is True

def test_checkpoint_controller_allows_nonzero_rank_fsdp_shard_without_manifest():
    from parascale.runtime.training.checkpointing import CheckpointController

    class State:
        global_step = 6
        last_metrics = {"loss": 0.25}

    class Backend:
        name = "fsdp"

        def __init__(self):
            self.saved = False

        def save_checkpoint(self, _manager, step=None, client_state=None):
            self.saved = True
            return {
                "files": [
                    {
                        "path": "rank-00001/fsdp_state.pt",
                        "role": "fsdp_state",
                        "rank": 1,
                    }
                ],
                "metadata": {"fsdp_state_dict_type": "sharded"},
            }

    class Engine:
        def __init__(self):
            self.state = State()
            self.training_backend = Backend()
            self.config = ParaScaleConfig(
                training_backend="fsdp",
                fsdp_state_dict_type="sharded",
            )
            self.barrier_called = False

        def _distributed_rank(self):
            return 1

        def _distributed_world_size(self):
            return 2

        def _distributed_barrier(self):
            self.barrier_called = True

        def plan(self):
            class Plan:
                def to_dict(self):
                    return {"backend": "fsdp"}

            return Plan()

    class Manager:
        def payload_path(self, *_args):
            return Path(tempfile.gettempdir()) / "unused.pt"

        def write_manifest(self, _manifest):
            raise AssertionError("nonzero shard rank must not write manifest")

    engine = Engine()

    result = CheckpointController(engine).save(Manager())

    assert engine.training_backend.saved is True
    assert result["skipped"] is True
    assert result["files"][0]["path"] == "rank-00001/fsdp_state.pt"
    assert engine.barrier_called is True

def test_checkpoint_controller_full_fsdp_save_runs_on_nonzero_rank():
    from parascale.runtime.training.checkpointing import CheckpointController

    class State:
        global_step = 7
        last_metrics = {}

    class Backend:
        name = "fsdp"

        def __init__(self):
            self.saved = False

        def save_checkpoint(self, _manager, step=None, client_state=None):
            self.saved = True
            return {
                "files": [],
                "metadata": {"fsdp_state_dict_type": "full"},
            }

    class Engine:
        def __init__(self):
            self.state = State()
            self.training_backend = Backend()
            self.config = ParaScaleConfig(
                training_backend="fsdp",
                fsdp_state_dict_type="full",
            )
            self.barrier_called = False

        def _distributed_rank(self):
            return 1

        def _distributed_world_size(self):
            return 2

        def _distributed_barrier(self):
            self.barrier_called = True

        def plan(self):
            raise AssertionError("nonzero rank must not write a manifest")

    class Manager:
        def payload_path(self, *_args):
            return Path(tempfile.gettempdir()) / "unused.pt"

        def write_manifest(self, _manifest):
            raise AssertionError("nonzero rank must not write a manifest")

    engine = Engine()

    result = CheckpointController(engine).save(Manager())

    assert engine.training_backend.saved is True
    assert result["skipped"] is True
    assert result["reason"] == (
        "backend checkpoint written; manifest is written by rank 0"
    )
    assert engine.barrier_called is True

def test_fsdp_full_checkpoint_nonzero_rank_participates_without_writing(
    monkeypatch,
):
    import parascale.runtime.backends.fsdp as fsdp_module
    from parascale.runtime.backends.fsdp import FSDPTrainingBackend

    saved_paths = []

    class TorchStub:
        @staticmethod
        def save(_payload, path):
            saved_paths.append(path)

    backend = FSDPTrainingBackend(
        config=ParaScaleConfig(
            training_backend="fsdp",
            fsdp_state_dict_type="full",
        ),
        local_rank=1,
    )
    backend.state_dict = lambda: {"backend": "fsdp"}
    backend._rank = lambda: 1
    monkeypatch.setattr(fsdp_module, "_require_torch", lambda: TorchStub())

    result = backend.save_checkpoint(
        CheckpointManager(str(Path(tempfile.gettempdir()) / "fsdp-rank1-no-write")),
        step=3,
        client_state={"global_step": 3},
    )

    assert saved_paths == []
    assert result["files"] == []
    assert result["metadata"]["rank"] == 1

def test_checkpoint_controller_rank0_manifest_lists_expected_fsdp_shards():
    from parascale.runtime.training.checkpointing import CheckpointController

    class State:
        global_step = 8
        last_metrics = {}

    class Backend:
        name = "fsdp"

        def save_checkpoint(self, _manager, step=None, client_state=None):
            return {
                "files": [
                    {
                        "path": "rank-00000/fsdp_state.pt",
                        "role": "fsdp_state",
                        "rank": 0,
                    }
                ],
                "metadata": {"fsdp_state_dict_type": "sharded"},
            }

    class Engine:
        state = State()
        training_backend = Backend()
        config = ParaScaleConfig(
            training_backend="fsdp", fsdp_state_dict_type="sharded"
        )

        def _distributed_rank(self):
            return 0

        def _distributed_world_size(self):
            return 2

        def _distributed_barrier(self):
            return None

        def plan(self):
            class Plan:
                def to_dict(self):
                    return {"backend": "fsdp"}

            return Plan()

    class Manager:
        def __init__(self):
            self.manifest = None

        def payload_path(self, *_args):
            return Path(tempfile.gettempdir()) / "unused.pt"

        def write_manifest(self, manifest):
            self.manifest = manifest
            return Path("manifest.json")

    manager = Manager()

    CheckpointController(Engine()).save(manager)

    paths = [entry["path"] for entry in manager.manifest.files]
    assert paths == ["rank-00000/fsdp_state.pt", "rank-00001/fsdp_state.pt"]
    assert manager.manifest.metadata["shard_count"] == 2
    assert manager.manifest.metadata["checkpoint_write_policy"] == "rank_sharded"
