# -*- coding: utf-8 -*-
# @Time : 2026/7/11 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import json
from types import SimpleNamespace

import pytest

from parascale.workloads.registry import build_training_components

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")


class _TinyGroundDino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.score = torch.nn.Parameter(torch.tensor(0.2))
        self.box = torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.25, 0.25]))
        self.seen_labels = None

    def forward(self, pixel_values=None, labels=None, **kwargs):
        self.seen_labels = labels
        if labels is not None:
            return {"loss": self.score.square() + self.box.square().mean()}
        batch = int(pixel_values.shape[0])
        logits = self.score.reshape(1, 1, 1).expand(batch, 2, 1)
        boxes = self.box.reshape(1, 1, 4).expand(batch, 2, 4)
        return SimpleNamespace(logits=logits, pred_boxes=boxes)


def _write_detection_sample(root, name="000001"):
    image_dir = root / "images"
    label_dir = root / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    image = Image.new("RGB", (20, 16), color=(30, 80, 120))
    image.save(image_dir / f"{name}.jpg")
    (label_dir / f"{name}.txt").write_text(
        "0 0.5 0.5 0.25 0.25\n",
        encoding="utf-8",
    )


def _write_phrase_grounding_sample(root, name="000001"):
    image_dir = root / "images"
    annotation_dir = root / "annotations"
    image_dir.mkdir()
    annotation_dir.mkdir()
    image = Image.new("RGB", (20, 16), color=(120, 30, 40))
    image.save(image_dir / f"{name}.jpg")
    payload = {
        "prompt": "red cube.",
        "objects": [
            {
                "phrase": "red cube",
                "class_label": 0,
                "bbox": [0.5, 0.5, 0.25, 0.25],
            }
        ],
    }
    (annotation_dir / f"{name}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_ground_dino_workload_builds_batch_and_loss(tmp_path, monkeypatch):
    _write_detection_sample(tmp_path)
    from parascale.workloads import ground_dino

    monkeypatch.setattr(
        ground_dino,
        "_load_ground_dino_model",
        lambda model_path: _TinyGroundDino(),
    )

    config = {
        "parascale": {"training_backend": "native"},
        "training": {
            "workload": "ground_dino",
            "max_steps": 1,
            "batch_size": 1,
        },
        "model": {"path": "test/tiny-ground-dino"},
        "data": {
            "data_dir": str(tmp_path),
            "batch_size": 1,
            "image_size": 32,
            "num_samples": 1,
        },
        "optimizer": {"lr": 0.001},
    }
    model, optimizer, dataloader, loss_fn = build_training_components(config)
    batch = next(iter(dataloader))
    output = model(**batch)
    loss = loss_fn(output, batch)
    loss.backward()
    optimizer.step()

    assert batch["pixel_values"].shape == (1, 3, 32, 32)
    assert batch["target_boxes"].shape == (1, 4)
    assert float(loss.detach()) >= 0.0
    assert any(param.grad is not None for param in model.parameters())


def test_ground_dino_official_loss_builds_hf_labels(tmp_path, monkeypatch):
    _write_detection_sample(tmp_path)
    from parascale.workloads import ground_dino

    tiny_model = _TinyGroundDino()
    monkeypatch.setattr(
        ground_dino,
        "_load_ground_dino_model",
        lambda model_path: tiny_model,
    )
    monkeypatch.setattr(
        ground_dino,
        "_load_ground_dino_tokenizer",
        lambda model_path: None,
    )

    config = {
        "parascale": {"training_backend": "native"},
        "training": {
            "workload": "ground_dino",
            "loss_type": "official",
            "max_steps": 1,
            "batch_size": 1,
        },
        "model": {"path": "test/tiny-ground-dino"},
        "data": {
            "data_dir": str(tmp_path),
            "batch_size": 1,
            "image_size": 32,
            "num_samples": 1,
        },
        "optimizer": {"lr": 0.001},
    }
    model, optimizer, dataloader, loss_fn = build_training_components(config)
    batch = next(iter(dataloader))
    output = model(**batch)
    loss = loss_fn(output, batch)
    loss.backward()
    optimizer.step()

    assert "labels" in batch
    assert batch["labels"][0]["class_labels"].tolist() == [0]
    assert batch["labels"][0]["boxes"].shape == (1, 4)
    assert batch["text"] == ["object."]
    assert tiny_model.seen_labels is not None
    assert float(loss.detach()) > 0.0


def test_ground_dino_official_loss_uses_phrase_grounding_labels(
    tmp_path, monkeypatch
):
    _write_phrase_grounding_sample(tmp_path)
    from parascale.workloads import ground_dino

    tiny_model = _TinyGroundDino()
    monkeypatch.setattr(
        ground_dino,
        "_load_ground_dino_model",
        lambda model_path: tiny_model,
    )
    monkeypatch.setattr(
        ground_dino,
        "_load_ground_dino_tokenizer",
        lambda model_path: None,
    )

    config = {
        "parascale": {"training_backend": "native"},
        "training": {
            "workload": "ground_dino",
            "loss_type": "official",
            "max_steps": 1,
            "batch_size": 1,
        },
        "model": {"path": "test/tiny-ground-dino"},
        "data": {
            "type": "phrase_grounding",
            "data_dir": str(tmp_path),
            "batch_size": 1,
            "image_size": 32,
            "num_samples": 1,
        },
        "optimizer": {"lr": 0.001},
    }
    model, optimizer, dataloader, loss_fn = build_training_components(config)
    batch = next(iter(dataloader))
    output = model(**batch)
    loss = loss_fn(output, batch)
    loss.backward()
    optimizer.step()

    assert batch["text"] == ["red cube."]
    assert batch["labels"][0]["class_labels"].tolist() == [0]
    assert batch["labels"][0]["boxes"].tolist() == [[0.5, 0.5, 0.25, 0.25]]
    assert tiny_model.seen_labels is not None
