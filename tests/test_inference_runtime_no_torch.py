# -*- coding: utf-8 -*-
# @Time : 2026/6/25 下午4:09
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Inference runtime tests that avoid importing torch."""

from __future__ import annotations


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
