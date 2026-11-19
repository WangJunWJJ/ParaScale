# -*- coding: utf-8 -*-
# @Time : 2026/11/16 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Actual Ultralytics detection training, export and resume integration."""

import copy
import json
from pathlib import Path

import pytest

from parascale.cli import main
from parascale.runtime.train_runner import run_train_from_config
from parascale.workloads import build_training_components

torch = pytest.importorskip("torch")
ultralytics = pytest.importorskip("ultralytics")
Image = pytest.importorskip("PIL.Image")


@pytest.fixture(params=["yolov8", "yolo11"])
def detection_config(tmp_path, monkeypatch, request):
    import yaml

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    family = request.param
    folder = "v8" if family == "yolov8" else "11"
    source = Path(ultralytics.__file__).parent / f"cfg/models/{folder}/{family}.yaml"
    architecture = yaml.safe_load(source.read_text(encoding="utf-8"))
    # Real Detect/C2f/SPPF layers with small widths keep regression tests cheap.
    if family == "yolov8":
        architecture["scales"] = {"n": [0.33, 0.0625, 256], "s": [0.33, 0.125, 256]}
    architecture["nc"] = 2
    for scale in ("n", "s"):
        (model_dir / f"{family}{scale}.yaml").write_text(
            yaml.safe_dump(architecture), encoding="utf-8"
        )
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "labels").mkdir()
    for index in range(4):
        Image.new("RGB", (80, 60), (index * 30 + 20, 50, 80)).save(
            data / "images" / f"{index}.jpg"
        )
        (data / "labels" / f"{index}.txt").write_text(
            "0 0.5 0.5 0.4 0.3\n" if index % 2 else "", encoding="utf-8"
        )
    student = {
        "backend": {"training_backend": "native", "precision": "fp32"},
        "model": {"path": str(model_dir / f"{family}n.yaml")},
        "data": {
            "data_dir": str(data),
            "image_size": 64,
            "batch_size": 2,
            "num_samples": 4,
            "num_workers": 0,
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
        "training": {
            "workload": "yolo_detection",
            "max_steps": 2,
            "checkpoint_interval": 2,
            "seed": 42,
            "checkpoint_dir": str(tmp_path / "student"),
        },
    }
    teacher = copy.deepcopy(student)
    teacher["model"]["path"] = str(model_dir / f"{family}s.yaml")
    teacher["training"]["seed"] = 7
    teacher_model, _, _, _ = build_training_components(teacher)
    checkpoint = tmp_path / "teacher.pt"
    torch.save(
        (
            teacher_model.wrapped_model if family == "yolo11" else teacher_model
        ).state_dict(),
        checkpoint,
    )
    student["distillation"] = {
        "enabled": True,
        "task": "detection",
        "feature_weight": 0.1,
        "teacher": {"config": teacher, "checkpoint": str(checkpoint)},
    }
    yield student
    torch.use_deterministic_algorithms(previous)


def _state(result):
    return torch.load(
        Path(result["checkpoint"]).parent / "backend_state.pt",
        map_location="cpu",
        weights_only=True,
    )["backend_state"]["model_state_dict"]


def test_detection_cli_resume_export_and_objective_contract(detection_config, tmp_path):
    config = detection_config
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "result.json"
    assert main(["train", "--config", str(config_path), "--output", str(output)]) == 0
    first = json.loads(output.read_text(encoding="utf-8"))
    state = _state(first)
    assert any(key.startswith("projectors.") for key in state)
    assert not any("teacher" in key for key in state)
    student_state = torch.load(
        first["distillation"]["student_checkpoint"], weights_only=True
    )
    assert not any("projector" in key or "student" in key for key in student_state)
    fresh, _, data, _ = build_training_components(config)
    fresh.wrapped_model.load_state_dict(student_state, strict=True)
    fresh.eval()
    with torch.no_grad():
        prediction = fresh(pixel_values=next(data)["img"])
    assert torch.isfinite(prediction[0]).all()

    resumed = run_train_from_config(config, resume_step=2)
    assert resumed["global_step"] == 4
    continuous_config = copy.deepcopy(config)
    continuous_config["training"].update(
        max_steps=4, checkpoint_dir=str(tmp_path / "continuous")
    )
    continuous = run_train_from_config(continuous_config)
    continuous_state = _state(continuous)
    for key, value in _state(resumed).items():
        torch.testing.assert_close(value, continuous_state[key], rtol=0, atol=1e-6)

    changed = copy.deepcopy(config)
    changed["distillation"]["feature_weight"] = 0.2
    with pytest.raises(ValueError, match="objective"):
        run_train_from_config(changed, resume_step=2)
    changed = copy.deepcopy(config)
    teacher_path = Path(config["distillation"]["teacher"]["checkpoint"])
    replacement = torch.load(teacher_path, weights_only=True)
    next(value for value in replacement.values() if value.is_floating_point()).add_(1)
    other_path = tmp_path / "changed_teacher.pt"
    torch.save(replacement, other_path)
    changed["distillation"]["teacher"]["checkpoint"] = str(other_path)
    with pytest.raises(ValueError, match="objective"):
        run_train_from_config(changed, resume_step=2)


def test_zero_weight_matches_standard_detection(detection_config, tmp_path):
    config = detection_config
    config["distillation"]["feature_weight"] = 0.0
    distilled = run_train_from_config(config)
    config = copy.deepcopy(config)
    config.pop("distillation")
    config["training"]["checkpoint_dir"] = str(tmp_path / "baseline")
    baseline = _state(run_train_from_config(config))
    student = torch.load(
        distilled["distillation"]["student_checkpoint"], weights_only=True
    )
    for key, value in student.items():
        torch.testing.assert_close(
            value, baseline[f"wrapped_model.{key}"], rtol=0, atol=0
        )
