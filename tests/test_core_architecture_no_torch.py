# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

from pathlib import Path

from parascale import (
    CheckpointConverter,
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
