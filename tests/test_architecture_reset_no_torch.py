# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:22
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

from importlib import import_module
from pathlib import Path


def test_runtime_plan_contract_is_lightweight_and_serializable():
    module = import_module("parascale.contracts.plan")

    device = module.DevicePlan(kind="cuda", communication_backend="nccl")
    backend = module.BackendPlan(name="native", distributed=True)
    communication = module.CommunicationPlan(
        backend="nccl",
        ddp_hook="bf16_compress",
        no_sync=True,
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
    assert payload["data"]["cache_enabled"] is True
    assert payload["checkpoint"]["interval_steps"] == 10
    assert payload["inference"]["enabled"] is False
    assert "__dict__" not in payload


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
        "parascale/strategy/device_plan.py",
        "parascale/strategy/data_plan.py",
        "parascale/strategy/backend_plan.py",
        "parascale/strategy/communication_plan.py",
        "parascale/strategy/inference_plan.py",
        "parascale/checkpoint/adapter.py",
        "parascale/checkpoint/validator.py",
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
