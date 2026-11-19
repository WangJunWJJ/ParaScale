# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午12:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Execution safety and request isolation for inference entrypoints."""

import subprocess
import sys

import pytest

from parascale import InferenceEngine, ServeRequest, ServingEngine
from parascale.runtime.inference.runner import InferenceRunner


class IdentityTask:
    name = "identity"

    def prepare_batch(self, batch):
        return batch

    def predict(self, model, batch):
        return model(batch)

    def postprocess(self, output):
        return output * 2

    def metric_counts(self, batch):
        return {}


@pytest.mark.parametrize(
    "operation",
    ["generate", "embed", "infer", "prefill", "decode", "callable", "runner"],
)
def test_torch_execution_is_eval_and_grad_free(operation):
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(3)
            self.states = []

        def forward(self, batch):
            self.states.append((self.training, torch.is_grad_enabled()))
            return self.bn(batch)

        generate = embed = prefill = decode = forward

    model = Model()
    original = {key: value.clone() for key, value in model.bn.state_dict().items()}
    batch = torch.randn(4, 3, requires_grad=True)
    with torch.enable_grad():
        if operation == "runner":
            result = InferenceRunner(
                model, "identity", task_adapter=IdentityTask()
            ).run([batch, batch], warmup_steps=1)["outputs"][0]
        else:
            engine = InferenceEngine().load_model(model)
            if operation == "infer":
                engine.task_adapter = IdentityTask()
            if operation == "callable":
                model.generate = None
                operation = "generate"
            payload = getattr(engine, operation)(batch)
            key = {"embed": "embeddings", "prefill": "state", "decode": "state"}.get(
                operation, "outputs"
            )
            result = payload[key]
        assert torch.is_grad_enabled()
    assert model.states == [(False, False)] * (2 if operation == "runner" else 1)
    assert not result.requires_grad
    assert result.grad_fn is None
    for key, value in model.bn.state_dict().items():
        torch.testing.assert_close(value, original[key])


@pytest.mark.parametrize("outputs", [{"scores": [1, 2]}, "shared", None, (1, 2)])
def test_serving_rejects_ambiguous_multi_request_output(outputs):
    with pytest.raises(ValueError, match="one output per request"):
        ServingEngine._normalize_outputs(outputs, 2)


def test_serving_rejects_tensor_batch_without_leaking_outputs():
    torch = pytest.importorskip("torch")
    outputs = torch.tensor([[1.0], [2.0]])
    serving = ServingEngine(runtime=InferenceEngine().load_model(lambda _: outputs))
    for index in range(2):
        serving.submit(ServeRequest(request_id=str(index), payload=index))
    responses = serving.step()
    assert len(responses) == 2
    assert all(
        response.output is None and "one output per request" in response.error
        for response in responses
    )
    assert serving.metrics()["requests_failed"] == 2
    assert serving.metrics()["kv_cache"]["blocks"] == 0


def test_serving_keeps_explicit_request_outputs_isolated():
    outputs = [{"value": 1}, {"value": 2}]
    serving = ServingEngine(runtime=InferenceEngine().load_model(lambda _: outputs))
    for index in range(2):
        serving.submit(ServeRequest(request_id=str(index), payload=index))
    responses = serving.step()
    assert [response.request_id for response in responses] == ["0", "1"]
    assert [response.output for response in responses] == outputs
    responses[0].output["value"] = 9
    assert responses[1].output == {"value": 2}


@pytest.mark.parametrize("output", [42, None, {"value": 1}])
def test_serving_preserves_singleton_compatibility(output):
    assert ServingEngine._normalize_outputs(output, 1) == [output]


def test_serving_rejects_wrong_list_cardinality():
    with pytest.raises(ValueError, match="1 outputs for batch size 2"):
        ServingEngine._normalize_outputs([1], 2)


@pytest.mark.parametrize("operation", ["generate", "embed", "infer", "runner"])
def test_model_errors_propagate_and_restore_grad_mode(operation):
    torch = pytest.importorskip("torch")

    class Broken(torch.nn.Module):
        def forward(self, batch):
            raise ValueError("model failure")

        generate = embed = forward

    model = Broken()
    with torch.enable_grad():
        with pytest.raises(ValueError, match="model failure"):
            if operation == "runner":
                InferenceRunner(model, "identity", task_adapter=IdentityTask()).run([1])
            else:
                engine = InferenceEngine(model=model, task_adapter=IdentityTask())
                getattr(engine, operation)(1)
        assert torch.is_grad_enabled()


def test_duck_models_work_without_importing_torch():
    script = """
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("torch must not be imported")
sys.meta_path.insert(0, BlockTorch())
from parascale import InferenceEngine
from parascale.runtime.inference.runner import InferenceRunner
class Duck:
    def eval(self):
        self.ready = True
    def generate(self, batch):
        assert self.ready
        return batch
    embed = predict = generate
model = Duck()
engine = InferenceEngine().load_model(model)
assert engine.generate([1])["outputs"] == [1]
assert engine.embed([2])["embeddings"] == [2]
runner = InferenceRunner(model, "vision_detection", memory_getter=lambda: None)
assert runner.run([{"num_images": 1}], warmup_steps=0)["outputs"] == [{"num_images": 1}]
assert "torch" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
