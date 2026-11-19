# -*- coding: utf-8 -*-
# @Time : 2026/11/16 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Detection workload routing and execution-boundary contracts."""

import pytest

from parascale.config import ParaScaleConfig
from parascale.runtime.distillation import validate_distillation_request
from parascale.workloads.capability import describe_workload
from parascale.workloads.specs.yolo import YoloDetectionSpec


def test_detection_spec_requires_official_loss():
    config = {"model": {"path": "student.pt"}, "data": {"data_dir": "dataset"}}
    assert YoloDetectionSpec.from_config(config).loss_type == "official"
    config["training"] = {"loss_type": "proxy"}
    with pytest.raises(ValueError, match="official"):
        YoloDetectionSpec.from_config(config)


def test_detection_workload_capability_is_not_world():
    capability = describe_workload({"training": {"workload": "yolo_detection"}})
    assert capability.capability_level == "local_native_yolo_detection"
    assert capability.flags["yolo_detection"]
    assert not capability.flags["yolo_world"]


@pytest.mark.parametrize("workload", ["torch_tiny", "yolo_world", "ground_dino"])
def test_detection_distillation_rejects_other_workloads(workload):
    with pytest.raises(ValueError, match="yolo_detection"):
        validate_distillation_request(
            {
                "training": {"workload": workload},
                "distillation": {"enabled": True, "task": "detection"},
            },
            ParaScaleConfig(),
        )


def test_detection_distillation_requires_detection_teacher(tmp_path):
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.touch()
    config = {
        "training": {"workload": "yolo_detection"},
        "distillation": {
            "enabled": True,
            "task": "detection",
            "feature_weight": 0.25,
            "teacher": {
                "checkpoint": str(checkpoint),
                "config": {"training": {"workload": "torch_tiny"}},
            },
        },
    }
    with pytest.raises(ValueError, match="teacher.*yolo_detection"):
        validate_distillation_request(config, ParaScaleConfig())
    config["distillation"]["teacher"]["config"]["training"]["workload"] = (
        "yolo_detection"
    )
    parsed, _ = validate_distillation_request(config, ParaScaleConfig())
    assert parsed.feature_weight == 0.25


@pytest.mark.parametrize(
    "settings,match",
    [
        ({"training_backend": "native_ddp"}, "single-process"),
        ({"precision": "fp16"}, "fp32"),
        ({"gradient_accumulation_steps": 2}, "accumulation"),
    ],
)
def test_detection_execution_guards(settings, match):
    with pytest.raises(ValueError, match=match):
        validate_distillation_request(
            {
                "training": {"workload": "yolo_detection"},
                "distillation": {"enabled": True, "task": "detection"},
            },
            ParaScaleConfig(**settings),
        )


@pytest.mark.parametrize(
    "destination",
    [
        "teacher.pt",
        "step-00000002/backend_state.pt",
        "step-00000002/manifest.json",
        "step-00000002/runtime_state_rank0.pt",
    ],
)
def test_student_export_cannot_overwrite_source(tmp_path, monkeypatch, destination):
    from parascale.runtime import train_runner

    checkpoint = tmp_path / "teacher.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        train_runner, "_write_runtime_config_artifacts", lambda *a, **k: {}
    )
    with pytest.raises(ValueError, match="must not"):
        train_runner.run_train_from_config(
            {
                "training": {
                    "workload": "yolo_detection",
                    "student_output": str(tmp_path / destination),
                    "checkpoint_dir": str(tmp_path),
                },
                "distillation": {
                    "enabled": True,
                    "task": "detection",
                    "teacher": {
                        "checkpoint": str(checkpoint),
                        "config": {"training": {"workload": "yolo_detection"}},
                    },
                },
            }
        )
