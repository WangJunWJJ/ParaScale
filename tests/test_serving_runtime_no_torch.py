# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

from parascale import InferenceEngine, ServeRequest, ServingEngine
from parascale.checkpoint import CheckpointManager, CheckpointManifest
from parascale.runtime.serve_runner import run_serve_from_config
from parascale.workloads.serving import default_serving_model_registry


class _ToyServingModel:
    def generate(self, requests):
        return [f"generated:{item}" for item in requests]

    def embed(self, requests):
        return [[len(item)] for item in requests]

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


def test_serving_engine_strict_errors_raise_after_releasing_cache():
    class BrokenModel:
        def generate(self, requests):
            raise RuntimeError("boom")

    serving = ServingEngine(
        runtime=InferenceEngine().load_model(model=BrokenModel()),
        strict_errors=True,
    )
    serving.submit(ServeRequest(request_id="bad", payload="hello"))

    try:
        serving.step()
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("strict serving mode must raise runtime errors")

    assert serving.metrics()["requests_failed"] == 1
    assert serving.metrics()["kv_cache"]["blocks"] == 0


def test_serve_runner_uses_serving_engine_for_non_strict_request_errors(
    tmp_path, monkeypatch
):
    class BrokenModel:
        def generate(self, requests):
            raise RuntimeError("boom")

    manager = CheckpointManager(str(tmp_path))
    manifest_path = manager.write_manifest(
        CheckpointManifest(step=1, backend="native", files=[])
    )
    monkeypatch.setattr(
        "parascale.runtime.serve_runner.build_serving_model_from_checkpoint",
        lambda *_args, **_kwargs: BrokenModel(),
    )

    payload = run_serve_from_config(
        {"serving": {"requests": ["a", "b"]}},
        checkpoint=str(manifest_path),
    )

    assert payload["strict_errors"] is False
    assert payload["result"]["mode"] == "error"
    assert payload["result"]["ok"] is False
    assert payload["result"]["outputs"] == [None, None]
    assert payload["serving_metrics"]["requests_failed"] == 2
    assert payload["serving_metrics"]["kv_cache"]["blocks"] == 0
    assert payload["evidence"]["runtime_status"] == "real_local"
    assert payload["evidence"]["strict_errors"] is False


def test_serve_runner_honors_strict_errors_config(tmp_path, monkeypatch):
    class BrokenModel:
        def generate(self, requests):
            raise RuntimeError("boom")

    manager = CheckpointManager(str(tmp_path))
    manifest_path = manager.write_manifest(
        CheckpointManifest(step=1, backend="native", files=[])
    )
    monkeypatch.setattr(
        "parascale.runtime.serve_runner.build_serving_model_from_checkpoint",
        lambda *_args, **_kwargs: BrokenModel(),
    )

    try:
        run_serve_from_config(
            {"serving": {"requests": ["a"], "strict_errors": True}},
            checkpoint=str(manifest_path),
        )
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("strict serving config must raise model errors")


def test_default_serving_model_registry_exposes_tiny_loader():
    registry = default_serving_model_registry()

    assert "torch_tiny_mlp" in registry.loaders
    try:
        registry.create("missing")
    except ValueError as exc:
        assert "unsupported serving workload" in str(exc)
    else:
        raise AssertionError("unknown serving workload must fail clearly")
