# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午4:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Runtime validation and optional real native distillation tests."""

import copy
import json
from pathlib import Path

import pytest

from parascale.cli import main
from parascale.runtime.train_runner import run_train_from_config


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"training_backend": "fsdp"}, "native"),
        ({"precision": "bf16"}, "fp32"),
        ({"adapter_only_checkpoint": True}, "full checkpoints"),
        ({"quantization": {"enabled": True}}, "quantization"),
        ({"gradient_accumulation_steps": 2}, "accumulation"),
        ({"tensor_parallel_size": 2}, "single-process"),
    ],
)
def test_distillation_rejects_unvalidated_combinations(settings, match):
    with pytest.raises(ValueError, match=match):
        run_train_from_config(
            {"parascale": settings, "distillation": {"enabled": True}}
        )


def test_distillation_requires_explicit_teacher_checkpoint():
    with pytest.raises(ValueError, match="teacher"):
        run_train_from_config({"distillation": {"enabled": True}})


def test_distillation_rejects_distributed_launch(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single-process"):
        run_train_from_config({"distillation": {"enabled": True}})


def _vision_config(path, hidden_dim=8, seed=42):
    return {
        "parascale": {"training_backend": "native"},
        "model": {"hidden_dim": hidden_dim, "num_classes": 3, "patch_size": 4},
        "data": {"image_size": 8, "batch_size": 2, "num_samples": 16},
        "optimizer": {"type": "adamw", "lr": 0.001},
        "training": {
            "workload": "vision_synthetic",
            "max_steps": 2,
            "seed": seed,
            "checkpoint_dir": str(path),
        },
    }


def test_distillation_train_resume_and_teacher_contract(tmp_path):
    torch = pytest.importorskip("torch")
    teacher_config = _vision_config(tmp_path / "teacher", hidden_dim=16, seed=7)
    teacher_run = run_train_from_config(teacher_config)
    teacher_checkpoint = Path(teacher_run["checkpoint"]).parent / "backend_state.pt"
    student_config = _vision_config(tmp_path / "student")
    student_config["distillation"] = {
        "enabled": True,
        "task": "classification",
        "temperature": 2.0,
        "alpha": 0.7,
        "teacher": {"config": teacher_config, "checkpoint": str(teacher_checkpoint)},
    }
    config_path = tmp_path / "distillation.json"
    config_path.write_text(json.dumps(student_config), encoding="utf-8")
    output_path = tmp_path / "result.json"
    assert main(["train", "--config", str(config_path), "--output", str(output_path)]) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["global_step"] == 2
    assert result["distillation"]["last_components"]["valid_count"] == 2
    payload = torch.load(
        Path(result["checkpoint"]).parent / "backend_state.pt", weights_only=True
    )
    assert not any(
        "teacher" in name for name in payload["backend_state"]["model_state_dict"]
    )
    resumed = run_train_from_config(student_config, resume_step=2)
    assert resumed["global_step"] == 4
    changed = copy.deepcopy(student_config)
    changed["distillation"]["temperature"] = 3.0
    with pytest.raises(ValueError, match="objective"):
        run_train_from_config(changed, resume_step=2)
    disabled = copy.deepcopy(student_config)
    disabled.pop("distillation")
    with pytest.raises(ValueError, match="objective"):
        run_train_from_config(disabled, resume_step=2)
    replacement = tmp_path / "different_teacher.pt"
    teacher_state = torch.load(
        teacher_checkpoint, map_location="cpu", weights_only=True
    )
    teacher_state["backend_state"]["model_state_dict"]["head.bias"].add_(1)
    torch.save(teacher_state, replacement)
    changed = copy.deepcopy(student_config)
    changed["distillation"]["teacher"]["checkpoint"] = str(replacement)
    with pytest.raises(ValueError, match="objective"):
        run_train_from_config(changed, resume_step=2)
