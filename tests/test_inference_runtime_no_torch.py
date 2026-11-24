# -*- coding: utf-8 -*-
# @Time : 2026/6/25 下午4:09
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Inference runtime tests that avoid importing torch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _identity_runner():
    from parascale.runtime.inference.runner import InferenceRunner

    return InferenceRunner(
        SimpleNamespace(predict=lambda batch: batch),
        "vision_detection",
        memory_getter=lambda: None,
    )


@pytest.mark.parametrize("count", [0, 1, 5])
def test_replayed_warmup_does_not_consume_requests(count):
    seen, consumed, received = [], [], []
    runner = _identity_runner()
    runner.model.predict = lambda value: seen.append(value) or value

    def batches():
        for value in range(count):
            assert len(consumed) <= len(received)
            consumed.append(value)
            yield value

    result = runner.run(
        batches(),
        warmup_steps=3,
        warmup_policy="replay_first",
        on_output=received.append,
        max_outputs=0,
    )
    assert seen == ([0] * 3 if count else []) + list(range(count))
    assert received == list(range(count))
    assert result["metrics"]["requests"] == count
    assert result["measurement_window"]["measured_batches"] == count
    assert result["timing"]["warmup_policy"] == "replay_first"
    assert result["timing"]["warmup_seconds"] >= 0


def test_invalid_warmup_policy_fails_before_model_preparation():
    runner = _identity_runner()
    runner.model.eval = lambda: pytest.fail("prepared invalid request")
    with pytest.raises(ValueError, match="warmup_policy"):
        runner.run([1], warmup_policy="discard_all")


def test_generation_result_counts_are_distinct_from_input_and_batches():
    from parascale.runtime.inference.runner import InferenceRunner

    outputs = [
        {
            "records": [{"id": "one"}, {"id": "two"}],
            "counts": {"requests": 2, "prompt_tokens": 8, "generated_tokens": 4},
        }
    ]
    runner = InferenceRunner(
        SimpleNamespace(generate=lambda _: outputs.pop(0)),
        "causal_lm_generation",
        memory_getter=lambda: None,
    )
    result = runner.run(
        [{"tokens": 1000}],
        warmup_steps=0,
        max_outputs=0,
        on_output=lambda output: output.clear(),
    )
    metrics = result["metrics"]
    assert metrics["requests"] == 2
    assert metrics["prompt_tokens"] == 8
    assert metrics["generated_tokens"] == metrics["tokens"] == 4
    assert metrics["generated_tokens_per_second"] == metrics["tokens_per_second"]
    assert result["measurement_window"]["measured_batches"] == 1
    assert result["output_retention"]["dropped"] == 1


def test_generation_batch_latency_is_not_amortized_request_cost(monkeypatch):
    from parascale.runtime.inference import runner as runtime

    now = [0.0]
    monkeypatch.setattr(runtime.time, "perf_counter", lambda: now[0])

    def generate(batch):
        count, seconds = batch["requests"], batch["seconds"]
        now[0] += seconds
        return {
            "records": [{} for _ in range(count)],
            "counts": {
                "requests": count,
                "prompt_tokens": count,
                "generated_tokens": count,
            },
        }

    result = runtime.InferenceRunner(
        SimpleNamespace(generate=generate),
        "causal_lm_generation",
        memory_getter=lambda: None,
    ).run(
        [{"requests": 4, "seconds": 0.1}, {"requests": 1, "seconds": 0.2}],
        warmup_steps=0,
    )
    metrics = result["metrics"]
    assert metrics["requests"] == 5
    assert metrics["requests_per_second"] == pytest.approx(5 / 0.3)
    assert metrics["latency_ms_avg"] == pytest.approx(120)
    assert metrics["latency_ms_total"] == pytest.approx(600)
    assert metrics["batch_latency_ms_avg"] == pytest.approx(150)
    assert metrics["batch_latency_ms_total"] == pytest.approx(300)
    assert metrics["amortized_request_time_ms"] == pytest.approx(60)
    assert "batch completion" in metrics["latency_scope"]


@pytest.mark.parametrize(
    "counts",
    [
        None,
        {},
        {"requests": True},
        {
            "requests": 1,
            "prompt_tokens": -1,
            "generated_tokens": 1,
        },
    ],
)
def test_invalid_generation_counts_never_report_success(counts):
    from parascale.runtime.inference.runner import InferenceRunner

    runner = InferenceRunner(
        SimpleNamespace(generate=lambda _: {"records": [{}], "counts": counts}),
        "causal_lm_generation",
        memory_getter=lambda: None,
    )
    with pytest.raises(ValueError, match="counts"):
        runner.run([{}], warmup_steps=0)


def test_inference_outputs_are_bounded_by_default_without_losing_counts():
    result = _identity_runner().run(iter(range(100)), warmup_steps=0)
    assert result["outputs"] == list(range(32))
    assert result["metrics"]["requests"] == 100
    assert result["output_retention"] == {
        "max_outputs": 32,
        "retained": 32,
        "dropped": 68,
        "callback_enabled": False,
    }


@pytest.mark.parametrize("limit", [0, 2, None])
def test_inference_callback_consumes_all_measured_outputs_in_order(limit):
    received = []
    result = _identity_runner().run(
        iter(range(8)), warmup_steps=2, max_outputs=limit, on_output=received.append
    )
    assert received == list(range(2, 8))
    assert result["outputs"] == received[:limit]
    assert result["metrics"]["requests"] == 6
    assert result["output_retention"]["dropped"] == 6 - len(result["outputs"])
    assert result["output_retention"]["callback_enabled"] is True


def test_inference_callback_failure_stops_consuming_input():
    consumed = []

    def batches():
        for value in range(10):
            consumed.append(value)
            yield value

    def sink(value):
        if value == 2:
            raise OSError("output sink unavailable")

    with pytest.raises(OSError, match="output sink unavailable"):
        _identity_runner().run(batches(), warmup_steps=0, max_outputs=0, on_output=sink)
    assert consumed == [0, 1, 2]


def test_inference_callback_mutation_does_not_change_metric_counts():
    result = _identity_runner().run(
        [{"num_images": 7}],
        warmup_steps=0,
        max_outputs=0,
        on_output=lambda output: output.clear(),
    )
    assert result["metrics"]["requests"] == 1
    assert result["metrics"]["images"] == 7


@pytest.mark.parametrize("limit", [-1, True, 1.5, "2"])
def test_inference_rejects_invalid_retention_before_model_preparation(limit):
    runner = _identity_runner()
    prepared = []
    runner.model.eval = lambda: prepared.append(True)
    with pytest.raises(ValueError, match="max_outputs"):
        runner.run([1], max_outputs=limit)
    assert not prepared


def test_inference_rejects_noncallable_output_callback():
    with pytest.raises(TypeError, match="on_output must be callable"):
        _identity_runner().run([1], on_output=42)


def test_inference_rejects_async_output_callbacks():
    async def sink(value):
        pass

    with pytest.raises(TypeError, match="synchronous"):
        _identity_runner().run([1], warmup_steps=0, on_output=sink)
    with pytest.raises(TypeError, match="synchronous"):
        _identity_runner().run([1], warmup_steps=0, on_output=lambda value: sink(value))


def test_streaming_inference_releases_consumed_objects():
    import weakref

    runner = _identity_runner()
    live = weakref.WeakSet()

    class Output:
        pass

    def predict(batch):
        assert len(live) <= 1
        output = Output()
        live.add(output)
        return output

    runner.model.predict = predict
    result = runner.run(range(1000), warmup_steps=0, max_outputs=0)
    assert result["outputs"] == []
    assert result["metrics"]["requests"] == 1000
    assert not live


def test_streaming_inference_does_not_retain_per_request_latency_history():
    import tracemalloc

    runner = _identity_runner()
    runner.run([None], warmup_steps=0, max_outputs=0)
    tracemalloc.start()
    try:
        result = runner.run((None for _ in range(20000)), warmup_steps=0, max_outputs=0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result["metrics"]["requests"] == 20000
    assert peak < 256 * 1024


@pytest.mark.parametrize("limit", [0, 2, None])
def test_inference_configuration_exposes_output_retention(limit, monkeypatch):
    from parascale.commands.run import run_inference_from_config

    monkeypatch.setattr(
        "parascale.commands.run.build_inference_components",
        lambda config: (_identity_runner().model, iter(range(8)), "vision_detection"),
    )
    result = run_inference_from_config(
        {
            "runtime": {"accelerator": "cpu"},
            "inference": {"warmup_steps": 0, "max_outputs": limit},
        }
    )
    assert result["outputs"] == list(range(8))[:limit]
    assert result["output_retention"]["max_outputs"] == limit


def test_inference_runner_moves_nested_batch_and_reports_metrics_without_torch():
    from parascale.runtime.inference.runner import InferenceRunner

    class FakeTensor:
        def __init__(self):
            self.moved_to = None

        def to(self, device, non_blocking=False):
            self.moved_to = str(device)
            self.non_blocking = non_blocking
            return self

    class FakeModel:
        def __init__(self):
            self.eval_called = False
            self.seen = None

        def eval(self):
            self.eval_called = True
            return self

        def predict(self, batch):
            self.seen = batch
            return {"scores": [1.0, 0.5], "num_images": 2}

    image = FakeTensor()
    runner = InferenceRunner(
        model=FakeModel(),
        task="vision_detection",
        device="npu:0",
        memory_getter=lambda: None,
    )

    payload = runner.run([{"images": image, "num_images": 2}], warmup_steps=0)

    assert image.moved_to == "npu:0"
    assert image.non_blocking is True
    assert payload["task"] == "vision_detection"
    assert payload["device"] == "npu:0"
    assert payload["outputs"][0]["scores"] == [1.0, 0.5]
    assert payload["metrics"]["requests"] == 1
    assert payload["metrics"]["images"] == 2
    assert payload["metrics"]["images_per_second"] > 0.0
    assert payload["metrics"]["latency_ms_avg"] >= 0.0


def test_inference_runner_excludes_warmup_batches_from_measured_outputs():
    from parascale.runtime.inference.runner import InferenceRunner

    class FakeModel:
        def __init__(self):
            self.seen = []

        def eval(self):
            return self

        def predict(self, batch):
            self.seen.append(batch["id"])
            return {"id": batch["id"], "num_images": batch["num_images"]}

    def batches():
        for index in range(3):
            yield {"id": index, "num_images": 1}

    model = FakeModel()
    runner = InferenceRunner(
        model=model,
        task="vision_detection",
        device="cpu",
        memory_getter=lambda: None,
    )

    payload = runner.run(batches(), warmup_steps=1)

    assert model.seen == [0, 1, 2]
    assert [item["id"] for item in payload["outputs"]] == [1, 2]
    assert payload["metrics"]["requests"] == 2
    assert payload["metrics"]["images"] == 2
    assert payload["measurement_window"] == {
        "warmup_steps_requested": 1,
        "warmup_steps_effective": 1,
        "measured_batches": 2,
        "warmup_excluded_from_metrics": True,
    }


def test_inference_runner_keeps_one_measured_batch_when_warmup_covers_finite_input():
    from parascale.runtime.inference.runner import InferenceRunner

    class FakeModel:
        def eval(self):
            return self

        def predict(self, batch):
            return {"id": batch["id"], "num_images": batch["num_images"]}

    runner = InferenceRunner(
        model=FakeModel(),
        task="vision_detection",
        device="cpu",
        memory_getter=lambda: None,
    )

    payload = runner.run([{"id": 0, "num_images": 2}], warmup_steps=1)

    assert payload["outputs"] == [{"id": 0, "num_images": 2}]
    assert payload["metrics"]["requests"] == 1
    assert payload["metrics"]["images"] == 2
    assert payload["measurement_window"] == {
        "warmup_steps_requested": 1,
        "warmup_steps_effective": 0,
        "measured_batches": 1,
        "warmup_excluded_from_metrics": False,
    }


def test_synthetic_clip_and_yolo_inference_adapters_are_generic_without_torch():
    from parascale.workloads.inference import build_inference_components

    clip_model, clip_batches, clip_task = build_inference_components(
        {
            "inference": {
                "workload": "clip_synthetic",
                "batch_size": 2,
                "num_batches": 1,
            }
        }
    )
    yolo_model, yolo_batches, yolo_task = build_inference_components(
        {
            "inference": {
                "workload": "yolo_world_synthetic",
                "batch_size": 2,
                "num_batches": 1,
            }
        }
    )

    assert clip_task == "multimodal_embedding"
    assert yolo_task == "vision_detection"
    assert len(clip_batches) == 1
    assert len(yolo_batches) == 1
    assert clip_model.to("cpu") is clip_model
    assert yolo_model.to("cpu") is yolo_model
    assert "image_embeddings" in clip_model.embed(clip_batches[0])
    assert "boxes" in yolo_model.detect(yolo_batches[0])


def test_real_inference_workloads_require_explicit_model_path_without_torch():
    import pytest

    from parascale.workloads.inference import build_inference_components

    with pytest.raises(ValueError, match="model.path"):
        build_inference_components({"inference": {"workload": "clip_real"}})

    with pytest.raises(ValueError, match="model.path"):
        build_inference_components({"inference": {"workload": "yolo_world_real"}})


def test_real_inference_workloads_build_generic_pil_batches_without_loading_model():
    from parascale.workloads.inference import (
        HFClipInferenceModel,
        UltralyticsYoloWorldInferenceModel,
        build_inference_components,
    )

    clip_model, clip_batches, clip_task = build_inference_components(
        {
            "model": {"path": "/tmp/clip"},
            "inference": {
                "workload": "clip_real",
                "batch_size": 2,
                "num_batches": 1,
            },
        }
    )
    yolo_model, yolo_batches, yolo_task = build_inference_components(
        {
            "model": {"path": "/tmp/yolo-world.pt"},
            "inference": {
                "workload": "yolo_world_real",
                "batch_size": 2,
                "num_batches": 1,
                "postprocess_mode": "async_cpu",
                "confidence_threshold": 0.2,
                "iou_threshold": 0.4,
            },
        }
    )

    assert isinstance(clip_model, HFClipInferenceModel)
    assert isinstance(yolo_model, UltralyticsYoloWorldInferenceModel)
    assert clip_task == "multimodal_embedding"
    assert yolo_task == "vision_detection"
    assert yolo_model.postprocess_config.mode == "async_cpu"
    assert yolo_model.postprocess_config.confidence_threshold == 0.2
    assert yolo_model.postprocess_config.iou_threshold == 0.4
    assert clip_batches[0]["num_pairs"] == 2
    assert yolo_batches[0]["num_images"] == 2
    assert hasattr(clip_batches[0]["images"][0], "size")
    assert hasattr(yolo_batches[0]["images"][0], "size")


def test_detection_postprocessor_runs_python_nms_without_torch():
    from parascale.runtime.inference.postprocess import (
        DetectionPostprocessConfig,
        DetectionPostprocessor,
    )

    boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
        [30.0, 30.0, 40.0, 40.0],
    ]
    scores = [0.9, 0.8, 0.7]
    class_ids = [0, 0, 0]

    result = DetectionPostprocessor(
        DetectionPostprocessConfig(confidence_threshold=0.1, iou_threshold=0.5)
    ).from_boxes_scores(
        boxes,
        scores,
        class_ids,
        image_index=0,
    )

    assert result["postprocess"]["backend"] == "python"
    assert result["postprocess"]["nms_backend"] == "python"
    assert result["num_boxes"] == 2
    assert [box["score"] for box in result["boxes"]] == [0.9, 0.7]


def test_detection_postprocessor_supports_async_cpu_mode_without_torch():
    from parascale.runtime.inference.postprocess import (
        DetectionPostprocessConfig,
        DetectionPostprocessor,
    )

    result = DetectionPostprocessor(
        DetectionPostprocessConfig(mode="async_cpu", confidence_threshold=0.1)
    ).from_boxes_scores(
        [[0.0, 0.0, 10.0, 10.0]],
        [0.9],
        [1],
        image_index=2,
    )

    assert result["postprocess"]["backend"] == "async_cpu"
    assert result["postprocess"]["nms_backend"] == "python"
    assert result["boxes"][0]["image_index"] == 2


def test_yolo_adapter_sets_writable_ultralytics_config_dir(monkeypatch):
    from parascale.workloads.inference import _ensure_ultralytics_config_dir

    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)

    path = _ensure_ultralytics_config_dir()

    assert path.endswith("parascale_ultralytics")
