# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import tempfile
from pathlib import Path

import pytest

from parascale import (
    CheckpointConverter,
    CheckpointManager,
    CheckpointManifest,
    ClusterTopology,
    CpuDeviceBackend,
    InferenceEngine,
    MockCollectiveBackend,
    ParaScaleConfig,
    ServeRequest,
    ServingEngine,
    TorchDistributedCollectiveBackend,
    TrainEngine,
    build_heterogeneous_parallel_plan,
    create_runtime_training_backend,
    default_training_backend_registry,
)
from parascale.workloads.serving import default_serving_model_registry

GB = 1024**3


def test_runtime_backend_modules_do_not_depend_on_legacy_facade():
    backend_dir = Path("parascale/runtime/backends")
    offenders = []
    for path in backend_dir.glob("*.py"):
        if path.name == "__init__.py":
            continue
        source = path.read_text(encoding="utf-8")
        if "parascale.runtime.backend" in source:
            offenders.append(path.name)

    assert offenders == []


def test_device_and_collective_backends_are_torch_free():
    device = CpuDeviceBackend()
    collective = MockCollectiveBackend()

    collective.init_process_group(world_size=2, rank=0)
    group = collective.new_group([0, 1], name="dp")
    value = collective.all_reduce({"loss": 1.0})
    gathered = collective.all_gather(None, "rank0", group=group)
    scattered = collective.reduce_scatter(None, ["rank0"], group=group)
    collective.barrier()

    assert device.is_available()
    assert device.device() == "cpu"
    assert device.device_name() == "cpu"
    assert device.memory_allocated() == 0
    assert device.supports_bf16() is False
    assert collective.initialized is True
    assert value == {"loss": 1.0}
    assert gathered == ["rank0"]
    assert scattered == ["rank0"]
    assert [event["op"] for event in collective.history] == [
        "init_process_group",
        "new_group",
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "barrier",
    ]


def test_torch_distributed_backend_exposes_clear_uninitialized_error():
    backend = TorchDistributedCollectiveBackend(backend="gloo")
    try:
        backend.barrier()
    except (ImportError, RuntimeError) as exc:
        assert "torch.distributed" in str(exc)
    else:
        raise AssertionError(
            "torch distributed backend must require an initialized process group"
        )


def test_cluster_topology_builds_heterogeneous_islands():
    topology = ClusterTopology.from_dicts(
        [
            {
                "hostname": "gpu-0",
                "devices": {"kind": "cuda", "count": 8, "memory_bytes": 80 * GB},
            },
            {
                "hostname": "npu-0",
                "devices": {"kind": "npu", "count": 8, "memory_bytes": 64 * GB},
            },
        ]
    )

    plan = topology.build_parallel_plan()

    assert topology.world_size == 16
    assert topology.is_heterogeneous is True
    assert plan.world_size == 16
    assert plan.placement_policy == "heterogeneous_islands"
    assert plan.cross_group_parallelism == "weighted_data_parallel"


def test_heterogeneous_plan_keeps_homogeneous_fast_path():
    plan = build_heterogeneous_parallel_plan(
        [
            {"device_type": "cuda", "device_count": 4, "memory_bytes": 40 * GB},
            {"device_type": "cuda", "device_count": 4, "memory_bytes": 40 * GB},
        ]
    )

    assert plan.world_size == 8
    assert plan.placement_policy == "homogeneous_fast_path"
    assert plan.groups[0].device_type == "cuda"


def test_train_and_serve_runtime_entrypoints_share_collective_contract():
    config = ParaScaleConfig(training_backend="auto")
    train = TrainEngine(
        config=config,
        model_profile={
            "total_params": 20_000_000,
            "total_memory": 80_000_000,
            "num_layers": 4,
            "model_type": "mlp",
        },
        hardware_profile={
            "num_gpus": 1,
            "gpu_memory": 24 * GB,
            "available_memory": 20 * GB,
            "gpus_per_node": 1,
        },
    )
    serve = InferenceEngine()

    train.initialize()
    serve.initialize(world_size=1)
    train.train_step({"input_ids": [1, 2]})
    train.evaluate([])
    serve.load_model(model="mock")
    generated = serve.generate(["hello"])
    embedded = serve.embed(["hello"])
    prefetched = serve.prefill({"input_ids": [1]})
    decoded = serve.decode({"input_ids": [1]})

    assert train.state.initialized is True
    assert train.plan().backend == "native"
    assert train.state.global_step == 1
    assert serve.state.initialized is True
    assert serve.state.requests == 2
    assert generated["outputs"] == ["generated"]
    assert embedded["embeddings"] == [[]]
    assert prefetched["state"] == "prefilled"
    assert decoded["state"] == "decoded"

    train.shutdown()
    serve.shutdown()
    assert train.state.initialized is False
    assert serve.state.initialized is False


class _ToyServingModel:
    def generate(self, requests):
        return [f"generated:{item}" for item in requests]

    def embed(self, requests):
        return [[len(item)] for item in requests]


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


def test_serve_engine_requires_model_unless_explicit_mock():
    serve = InferenceEngine()

    try:
        serve.generate(["hello"])
    except RuntimeError as exc:
        assert "requires load_model" in str(exc)
    else:
        raise AssertionError("generate() must fail fast without model or mock mode")

    serve.load_model(model="mock")
    assert serve.generate(["hello"])["mode"] == "mock"


def test_serve_engine_runs_loaded_model_generate_and_embed():
    serve = InferenceEngine().initialize(world_size=1).load_model(
        model=_ToyServingModel()
    )

    generated = serve.generate(["hello"])
    embedded = serve.embed(["abc"])

    assert generated["mode"] == "model"
    assert generated["outputs"] == ["generated:hello"]
    assert embedded["embeddings"] == [[3]]
    assert serve.state.requests == 2


def test_serving_engine_step_returns_runtime_outputs():
    runtime = InferenceEngine().load_model(model=_ToyServingModel())
    serving = ServingEngine(runtime=runtime)

    serving.submit(ServeRequest(request_id="r1", payload="hello"))
    responses = serving.step()

    assert responses[0].request_id == "r1"
    assert responses[0].output == "generated:hello"
    assert responses[0].metadata["mode"] == "model"


def test_serving_engine_batches_requests_and_reports_metrics():
    runtime = InferenceEngine().load_model(model=_ToyServingModel())
    serving = ServingEngine(runtime=runtime)

    serving.submit(ServeRequest(request_id="r1", payload="hello"))
    serving.submit(ServeRequest(request_id="r2", payload="world"))
    responses = serving.step()
    metrics = serving.metrics()

    assert [response.output for response in responses] == [
        "generated:hello",
        "generated:world",
    ]
    assert responses[0].metadata["batch_size"] == 2
    assert metrics["requests_completed"] == 2
    assert metrics["batches_completed"] == 1
    assert metrics["kv_cache"]["blocks"] == 0


def test_mock_serve_engine_returns_one_output_per_request():
    runtime = InferenceEngine().load_model(model="mock", mock=True)

    generated = runtime.generate(["a", "b", "c"])
    embedded = runtime.embed(["x", "y"])

    assert generated["mode"] == "mock"
    assert generated["outputs"] == ["generated", "generated", "generated"]
    assert embedded["embeddings"] == [[], []]


def test_mock_serving_engine_rejects_length_mismatch():
    serving = ServingEngine(
        runtime=InferenceEngine().load_model(model="mock", mock=True)
    )
    serving.submit(ServeRequest(request_id="r1", payload="hello"))
    serving.submit(ServeRequest(request_id="r2", payload="world"))

    responses = serving.step()

    assert len(responses) == 2
    assert [response.output for response in responses] == ["generated", "generated"]


def test_serving_engine_returns_request_errors_without_sticking_cache():
    class BrokenModel:
        def generate(self, requests):
            raise RuntimeError("boom")

    serving = ServingEngine(runtime=InferenceEngine().load_model(model=BrokenModel()))
    serving.submit(ServeRequest(request_id="bad", payload="hello"))

    responses = serving.step()

    assert responses[0].ok is False
    assert "boom" in responses[0].error
    assert serving.metrics()["requests_failed"] == 1
    assert serving.metrics()["kv_cache"]["blocks"] == 0


def test_train_engine_fit_requires_real_step_contract():
    config = ParaScaleConfig(training_backend="native")
    train = TrainEngine(
        config=config,
        model_profile={
            "total_params": 20_000_000,
            "total_memory": 80_000_000,
            "num_layers": 4,
            "model_type": "mlp",
        },
        hardware_profile={
            "num_gpus": 1,
            "gpu_memory": 24 * GB,
            "available_memory": 20 * GB,
            "gpus_per_node": 1,
        },
    )

    try:
        train.fit([{"x": 1}])
    except RuntimeError as exc:
        assert "requires either step_fn" in str(exc)
    else:
        raise AssertionError("fit() must not silently fake a training loop")


def test_train_engine_fit_runs_step_fn_and_respects_max_steps():
    config = ParaScaleConfig(training_backend="native")
    train = TrainEngine(
        config=config,
        model_profile={
            "total_params": 20_000_000,
            "total_memory": 80_000_000,
            "num_layers": 4,
            "model_type": "mlp",
        },
        hardware_profile={
            "num_gpus": 1,
            "gpu_memory": 24 * GB,
            "available_memory": 20 * GB,
            "gpus_per_node": 1,
        },
    )
    seen = []

    def step_fn(batch, engine):
        seen.append((batch["x"], engine.state.global_step))
        return {"loss": batch["x"] * 0.5}

    state = train.fit([{"x": 2}, {"x": 4}, {"x": 6}], max_steps=2, step_fn=step_fn)

    assert seen == [(2, 0), (4, 1)]
    assert state.global_step == 2
    assert state.last_metrics["loss"] == 2.0
    assert state.last_metrics["dataloader_wait_ms"] >= 0


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


def test_checkpoint_manifest_validation_rejects_negative_counters():
    try:
        CheckpointManifest(step=1, consumed_tokens=-1)
    except ValueError as exc:
        assert "consumed_tokens" in str(exc)
    else:
        raise AssertionError("negative consumed_tokens must be rejected")


def test_training_backend_registry_and_serving_components_are_available():
    registry = default_training_backend_registry()
    backend = registry.create("native")
    runtime_backend = create_runtime_training_backend(
        config=ParaScaleConfig(training_backend="native")
    )
    converter = CheckpointConverter()
    plan = converter.build_plan("fsdp")
    serving = ServingEngine(runtime=InferenceEngine().load_model(model="mock"))

    serving.submit(ServeRequest(request_id="r1", payload="hello"))
    responses = serving.step()

    assert backend.state_dict()["backend"] == "native"
    assert runtime_backend.name == "native"
    assert plan.target_format == "parascale"
    assert "inspect_source" in plan.steps
    assert converter.convert(plan)["converted"] is False
    assert responses[0].request_id == "r1"
    assert responses[0].metadata["mode"] == "mock"


def test_default_serving_model_registry_exposes_tiny_loader():
    registry = default_serving_model_registry()

    assert "torch_tiny_mlp" in registry.loaders
    try:
        registry.create("missing")
    except ValueError as exc:
        assert "unsupported serving workload" in str(exc)
    else:
        raise AssertionError("unknown serving workload must fail clearly")


def test_checkpoint_converter_validates_format_matrix():
    converter = CheckpointConverter()
    plan = converter.build_plan(
        "hf", target_format="serve_manifest", source_path="missing"
    )

    assert plan.source_format == "hf"
    assert plan.target_format == "serve_manifest"
    assert plan.metadata["requires_weight_rewrite"] is True
    assert plan.metadata["source_exists"] is False
    try:
        converter.build_plan("unknown")
    except ValueError as exc:
        assert "unsupported checkpoint source format" in str(exc)
    else:
        raise AssertionError("unknown checkpoint source format must be rejected")


def test_checkpoint_converter_emits_serve_manifest_for_parascale_manifest():
    root = Path(tempfile.gettempdir()) / "parascale-test-runs" / "checkpoint_converter"
    manager = CheckpointManager(str(root))
    manifest = CheckpointManifest(
        step=3,
        backend="native",
        files=[
            {"path": "backend_state.pt", "role": "backend_state", "format": "torch"}
        ],
        metadata={"note": "source"},
    )
    source = manager.write_manifest(manifest)
    target = root / "serve" / "manifest.json"

    converter = CheckpointConverter()
    plan = converter.build_plan(
        "parascale",
        target_format="serve_manifest",
        source_path=str(source),
        target_path=str(target),
    )
    result = converter.convert(plan)

    assert result["converted"] is True
    assert result["target_manifest"] == str(target)
    converted = CheckpointManifest.from_dict(
        __import__("json").loads(target.read_text(encoding="utf-8"))
    )
    assert converted.format == "parascale_serve_manifest_v1"
    assert converted.metadata["serve_ready"] is True
    assert converted.metadata["conversion_target"] == "serve_manifest"


def test_checkpoint_converter_inspects_hf_checkpoint_directory():
    root = Path(tempfile.gettempdir()) / "parascale-test-runs" / "hf_converter"
    source = root / "hf"
    target = root / "converted" / "manifest.json"
    source.mkdir(parents=True, exist_ok=True)
    (source / "config.json").write_text(
        '{"model_type": "tiny", "hidden_size": 8}\n', encoding="utf-8"
    )
    (source / "model.safetensors").write_bytes(b"tiny-weights")

    converter = CheckpointConverter()
    plan = converter.build_plan(
        "hf",
        target_format="serve_manifest",
        source_path=str(source),
        target_path=str(target),
    )
    result = converter.convert(plan)

    assert result["converted"] is True
    assert result["weight_files"] == 1
    assert result["weight_rewrite_performed"] is False
    converted = CheckpointManifest.from_dict(
        __import__("json").loads(target.read_text(encoding="utf-8"))
    )
    assert converted.format == "parascale_serve_manifest_v1"
    assert converted.backend == "hf"
    assert converted.files[0]["format"] == "safetensors"
    assert converted.metadata["hf_config"]["model_type"] == "tiny"
    assert converted.metadata["serve_layout"]["loader"] == "hf_reference"
