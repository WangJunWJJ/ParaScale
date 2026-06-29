# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:22
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

from importlib import import_module
from pathlib import Path

import pytest


def test_runtime_plan_contract_is_lightweight_and_serializable():
    module = import_module("parascale.contracts.plan")

    device = module.DevicePlan(kind="cuda", communication_backend="nccl")
    backend = module.BackendPlan(name="native", distributed=True)
    communication = module.CommunicationPlan(
        backend="nccl",
        ddp_hook="bf16_compress",
        bucket_cap_mb=64,
        use_no_sync=True,
        adapter_only_sync=True,
        overlap_h2d=True,
        reasons=("gradient accumulation",),
        evidence={"trainable_ratio": 0.01},
    )
    data = module.DataPlan(kind="multimodal", cache_enabled=True)
    checkpoint = module.CheckpointPlan(enabled=True, interval_steps=10)
    inference = module.InferencePlan(enabled=False)
    plan = module.RuntimePlan(
        mode="train",
        device=device,
        backend=backend,
        communication=communication,
        data=data,
        checkpoint=checkpoint,
        inference=inference,
    )

    payload = plan.to_dict()

    assert payload["device"]["kind"] == "cuda"
    assert payload["backend"]["name"] == "native"
    assert payload["communication"]["ddp_hook"] == "bf16_compress"
    assert payload["communication"]["use_no_sync"] is True
    assert payload["communication"]["adapter_only_sync"] is True
    assert payload["communication"]["reasons"] == ["gradient accumulation"]
    assert payload["data"]["cache_enabled"] is True
    assert payload["checkpoint"]["interval_steps"] == 10
    assert payload["inference"]["enabled"] is False
    assert "__dict__" not in payload


def test_communication_builder_returns_the_contract_type():
    contracts = import_module("parascale.contracts")
    communication = import_module("parascale.communication")

    plan = communication.build_communication_plan(
        backend="native_ddp",
        precision="bf16",
        task_type="vlm_lora",
        gradient_accumulation_steps=4,
        trainable_ratio=0.01,
        dataloader_wait_ms=10.0,
    )

    assert isinstance(plan, contracts.CommunicationPlan)
    assert plan.use_no_sync is True
    assert plan.adapter_only_sync is True
    assert plan.overlap_h2d is True
    assert plan.evidence["gradient_accumulation_steps"] == 4


def test_device_backends_are_split_by_hardware_layer():
    cuda = import_module("parascale.core.device.cuda")
    ascend = import_module("parascale.core.device.ascend")
    cpu = import_module("parascale.core.device.cpu")
    registry = import_module("parascale.core.device.registry")

    assert cuda.CudaDeviceBackend().accelerator == "cuda"
    assert ascend.AscendDeviceBackend().accelerator == "npu"
    assert cpu.CpuDeviceBackend().accelerator == "cpu"
    assert registry.create_device_backend("cpu").device_name() == "cpu"


def test_new_runtime_namespaces_expose_training_inference_and_reporting():
    training = import_module("parascale.runtime.training")
    inference = import_module("parascale.runtime.inference")
    reporting = import_module("parascale.reporting")

    assert hasattr(training, "TrainEngine")
    assert hasattr(training, "FitLoopRunner")
    assert hasattr(inference, "InferenceEngine")
    assert hasattr(inference, "InferenceBatcher")
    assert hasattr(reporting, "BenchmarkResult")
    assert hasattr(reporting, "build_backend_matrix_report")


def test_ascend_native_backend_is_registered_but_fails_fast_without_npu():
    backends = import_module("parascale.runtime.backends")
    registry = backends.default_training_backend_registry()

    assert "ascend_native" in registry.factories

    backend = registry.create("ascend_native", config=None, local_rank=0)
    try:
        backend.setup()
    except RuntimeError as exc:
        assert "torch_npu" in str(exc) or "Ascend" in str(exc)
    else:
        raise AssertionError("ascend_native must fail fast without an NPU runtime")


def test_reset_runtime_has_no_legacy_core_implementation_modules():
    legacy_paths = [
        Path("parascale/core/distributed.py"),
        Path("parascale/runtime/launcher.py"),
        Path("parascale/runtime/train.py"),
        Path("parascale/runtime/infer.py"),
        Path("parascale/runtime/benchmark.py"),
        Path("parascale/runtime/matrix.py"),
        Path("parascale/runtime/core"),
        Path("parascale/data/multimodal.py"),
    ]

    assert [str(path) for path in legacy_paths if path.exists()] == []


def test_reset_plan_target_packages_are_materialized():
    required_files = [
        "parascale/core/distributed/__init__.py",
        "parascale/core/distributed/collective.py",
        "parascale/core/distributed/process_group.py",
        "parascale/core/distributed/registry.py",
        "parascale/runtime/launcher/__init__.py",
        "parascale/runtime/launcher/local.py",
        "parascale/runtime/launcher/torchrun.py",
        "parascale/runtime/launcher/deepspeed.py",
        "parascale/data/multimodal/batch.py",
        "parascale/data/multimodal/cache.py",
        "parascale/data/multimodal/processor.py",
        "parascale/data/multimodal/prompt.py",
        "parascale/data/multimodal/profiler.py",
        "parascale/checkpoint/adapter.py",
    ]

    missing = [path for path in required_files if not Path(path).is_file()]

    assert missing == []


def test_multimodal_package_init_is_only_public_exports():
    init_path = Path("parascale/data/multimodal/__init__.py")
    source = init_path.read_text(encoding="utf-8")

    assert "class MultiModalDataPipeline" not in source
    assert "def normalize_multimodal_sample" not in source
    assert "class TokenCostEstimate" not in source


def test_production_code_imports_reset_runtime_namespaces():
    forbidden = [
        "from parascale.runtime.train import",
        "import parascale.runtime.train",
        "from parascale.runtime.infer import",
        "import parascale.runtime.infer",
        "from parascale.runtime.benchmark import",
        "import parascale.runtime.benchmark",
        "from parascale.runtime.matrix import",
        "import parascale.runtime.matrix",
        "from parascale.runtime.core import",
        "import parascale.runtime.core",
    ]
    offenders = []
    for path in Path("parascale").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                offenders.append((str(path), token))

    assert offenders == []


def test_workload_adapter_registry_resolves_once_and_builds():
    from parascale.contracts import WorkloadAdapter
    from parascale.workloads.adapters import (
        WorkloadAdapter as PublicWorkloadAdapter,
    )
    from parascale.workloads.adapters import WorkloadAdapterRegistry

    class DemoAdapter:
        name = "demo"

        def build(self, config_data):
            return {"value": config_data["value"]}

    registry = WorkloadAdapterRegistry()
    registry.register(DemoAdapter())

    assert PublicWorkloadAdapter is WorkloadAdapter
    assert registry.resolve(" DEMO ").name == "demo"
    assert registry.create("demo", {"value": 7}) == {"value": 7}
    assert registry.names() == ["demo"]


def test_workload_adapter_registry_rejects_duplicates_and_unknown_names():
    from parascale.workloads.adapters import WorkloadAdapterRegistry

    class DemoAdapter:
        name = "demo"

        def build(self, config_data):
            return config_data

    registry = WorkloadAdapterRegistry()
    registry.register(DemoAdapter())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(DemoAdapter())
    with pytest.raises(ValueError, match="unknown workload adapter"):
        registry.resolve("missing")


def test_default_inference_task_adapters_have_real_behavior():
    from parascale.runtime.inference.tasks import default_inference_task_registry

    class Model:
        def detect(self, batch):
            return {"detections": batch["num_images"]}

        def generate(self, batch):
            return {"generated": batch.get("tokens", 0)}

    registry = default_inference_task_registry()
    vision = registry.resolve("vision")
    text = registry.resolve("text")
    multimodal = registry.resolve("multimodal")

    vision_batch = {"num_images": 2}
    assert vision.predict(Model(), vision.prepare_batch(vision_batch)) == {
        "detections": 2
    }
    assert vision.metric_counts(vision_batch) == {"images": 2}
    assert text.predict(Model(), {"tokens": 8}) == {"generated": 8}
    assert text.metric_counts({"tokens": 8}) == {"tokens": 8}
    assert multimodal.metric_counts(
        {"num_images": 2, "num_pairs": 2, "tokens": 8}
    ) == {"images": 2, "image_text_pairs": 2, "tokens": 8}
    assert vision.execution_hints()["task"] == "vision"


def test_inference_task_registry_rejects_duplicate_registration():
    from parascale.runtime.inference.tasks import (
        InferenceTaskRegistry,
        VisionInferenceTaskAdapter,
    )

    registry = InferenceTaskRegistry()
    registry.register(VisionInferenceTaskAdapter())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(VisionInferenceTaskAdapter())


def test_default_inference_task_registry_resolves_workload_task_aliases():
    from parascale.runtime.inference.tasks import default_inference_task_registry

    registry = default_inference_task_registry()

    assert registry.resolve("vision_detection").name == "vision"
    assert registry.resolve("multimodal_embedding").name == "multimodal"


def test_inference_runtime_has_one_public_engine_name():
    import parascale
    import parascale.runtime as runtime
    from parascale.runtime.inference import InferenceEngine

    assert InferenceEngine.__name__ == "InferenceEngine"
    assert not hasattr(runtime, "ServeEngine")
    assert not hasattr(parascale, "ServeEngine")


def test_inference_engine_accepts_a_resolved_task_adapter():
    from parascale.runtime.inference import InferenceEngine
    from parascale.runtime.inference.tasks import VisionInferenceTaskAdapter

    class Model:
        def detect(self, batch):
            return [batch["num_images"]]

    engine = InferenceEngine(task_adapter=VisionInferenceTaskAdapter()).load_model(
        model=Model()
    )

    result = engine.infer({"num_images": 3})

    assert result == {"outputs": [3], "mode": "task", "task": "vision"}


def test_deprecated_facades_and_empty_placeholders_are_physically_removed():
    deprecated_paths = [
        Path("parascale/runtime/backend.py"),
        Path("parascale/runtime/factory.py"),
        Path("parascale/checkpoint/manifest.py"),
        Path("parascale/checkpoint/validator.py"),
        Path("parascale/runtime/inference/memory.py"),
        Path("parascale/reporting/profile.py"),
        Path("parascale/reporting/tuner.py"),
        Path("parascale/data/text"),
    ]

    assert [str(path) for path in deprecated_paths if path.exists()] == []


def test_production_code_does_not_import_deprecated_facades():
    forbidden = (
        "from parascale.runtime.backend import",
        "from parascale.runtime.factory import",
        "from parascale.checkpoint.manifest import",
        "from parascale.checkpoint.validator import",
    )
    violations = []
    for path in Path("parascale").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if any(module in source for module in forbidden):
            violations.append(str(path))

    assert violations == []
