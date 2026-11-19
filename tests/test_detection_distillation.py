# -*- coding: utf-8 -*-
# @Time : 2026/11/16 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Focused detection KD contract tests, without requiring Ultralytics."""

import copy
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from parascale.distillation import detection as kd  # noqa: E402
from parascale.distillation.detection_config import (  # noqa: E402
    DetectionDistillationConfig,
)


class Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Detect(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.nc, self.nl, self.end2end = 2, 3, False
        self.register_buffer("stride", torch.tensor([8.0, 16.0, 32.0]))
        self.cv2 = nn.ModuleList([nn.Sequential(Conv(c)) for c in channels])

    def forward(self, features):
        for i, conv in enumerate(self.cv2):
            features[i] = conv(features[i])
        return sum(x.square().mean() for x in features)


class DetectionModel(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.neck = nn.ModuleList([nn.Conv2d(3, width, 1) for _ in range(3)])
        self.model = nn.ModuleList([Detect([width] * 3)])
        self.names = {0: "a", 1: "b"}
        self.yaml = {
            "scale": "n",
            "backbone": [[-1, 1, "Conv", [width, 3, 2]]],
            "head": [[[0, 1, 2], 1, "Detect", ["nc"]]],
        }
        self.calls = 0
        self.fail = False

    def predict(self, img):
        self.calls += 1
        out = self.model[-1](
            [conv(img[:, :, :: 2**i, :: 2**i]) for i, conv in enumerate(self.neck)]
        )
        if self.fail:
            raise RuntimeError("injected failure")
        return out


class Proxy(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.wrapped_model = DetectionModel(width)

    def forward(self, **batch):
        assert batch["official_loss"]
        return (self.wrapped_model.predict(batch["img"]), "official terms")


@pytest.fixture
def pair(monkeypatch):
    monkeypatch.setattr(kd, "_ultralytics_types", lambda: (DetectionModel, Detect))
    return Proxy(), Proxy(6)


def make(pair, weight=1.0):
    return kd.create_detection_distillation(
        *pair,
        DetectionDistillationConfig(feature_weight=weight),
        lambda out, batch: out[0].sum(),
    )


def batch(n=2):
    return {"img": torch.randn(n, 3, 8, 8), "official_loss": True}


@pytest.mark.parametrize("weight", [0.0, 1.0])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_factory_preserves_student_rng(pair, weight, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pair[0].to(device)
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    cpu_before = torch.get_rng_state().clone()
    device_before = torch.cuda.get_rng_state().clone() if devices else None
    with torch.random.fork_rng(devices=devices):
        expected_order = torch.randperm(32)
        expected_dropout = nn.functional.dropout(
            torch.ones(64, device=device), p=0.5, training=True
        )
    make(pair, weight)
    assert torch.equal(torch.get_rng_state(), cpu_before)
    if devices:
        assert torch.equal(torch.cuda.get_rng_state(), device_before)
    assert torch.equal(torch.randperm(32), expected_order)
    assert torch.equal(
        nn.functional.dropout(torch.ones(64, device=device), p=0.5, training=True),
        expected_dropout,
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"enabled": 1},
        {"task": "pose"},
        {"task": []},
        {"feature_weight": True},
        {"feature_weight": -1},
        {"feature_weight": float("nan")},
        {"feature_weight": float("inf")},
        {"unknown": 1},
    ],
)
def test_config_rejects(settings):
    with pytest.raises((TypeError, ValueError)):
        DetectionDistillationConfig.from_dict(settings)


def test_config_roundtrip_and_torch_free():
    config = DetectionDistillationConfig()
    assert config.to_dict() == {
        "enabled": False,
        "task": "detection",
        "feature_weight": 1.0,
    }
    assert DetectionDistillationConfig.from_dict(config.to_dict()) == config
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from parascale.distillation.detection_config import DetectionDistillationConfig; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_ownership_gradients_checkpoint_and_capture(pair):
    model, objective = make(pair)
    model.train().to("cpu")
    assert not objective.teacher.training
    assert not isinstance(objective, nn.Module)
    assert set(model._modules) == {"student", "projectors"}
    assert any(k.startswith("projectors.") for k in model.state_dict())
    assert set(model.student.state_dict()) == set(pair[0].state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = batch()
    output = model(**data)
    assert [x.shape[1] for x in output.features] == [6, 6, 6]
    total = objective(output, data)
    total.backward()
    assert all(p.grad is not None for p in model.projectors.parameters())
    assert any(p.grad is not None for p in model.student.parameters())
    assert all(p.grad is None and not p.requires_grad for p in pair[1].parameters())
    assert not {id(p) for p in pair[1].parameters()} & {
        id(p) for g in optimizer.param_groups for p in g["params"]
    }
    assert objective.last_metrics["total"] == pytest.approx(
        objective.last_metrics["supervised"] + objective.last_metrics["distillation"]
    )
    assert not pair[0].wrapped_model.model[-1]._forward_pre_hooks
    assert not pair[1].wrapped_model.model[-1]._forward_pre_hooks


def test_zero_weight_exact_baseline(pair):
    model, objective = make(pair, 0)
    data = batch()
    output = model(**data)
    baseline = output.original[0]
    objective.detection_loss_fn = lambda original, batch: baseline
    assert objective(output, data) is baseline
    assert pair[1].wrapped_model.calls == 0
    assert objective.last_metrics["distillation"] == 0


@pytest.mark.parametrize("weight", [0.0, 1.0])
@pytest.mark.parametrize("flag", [None, False, 1, "true"])
def test_forward_requires_official_loss(pair, weight, flag):
    model, _ = make(pair, weight)
    data = batch()
    if flag is None:
        data.pop("official_loss")
    else:
        data["official_loss"] = flag
    with pytest.raises(ValueError, match="official_loss=True"):
        model(**data)
    assert pair[0].wrapped_model.calls == 0
    assert not pair[0].wrapped_model.model[-1]._forward_pre_hooks


@pytest.mark.parametrize("which", [0, 1])
def test_failure_removes_hooks(pair, which):
    model, objective = make(pair)
    pair[which].wrapped_model.fail = True
    data = batch()
    with pytest.raises(RuntimeError, match="injected"):
        objective(model(**data), data)
    assert all(not p.wrapped_model.model[-1]._forward_pre_hooks for p in pair)


@pytest.mark.parametrize(
    "kind",
    ["names", "classes", "stride", "family", "end2end", "subclass", "scales", "alias"],
)
def test_incompatible_rejected(pair, kind):
    student, teacher = pair
    base, head = teacher.wrapped_model, teacher.wrapped_model.model[-1]
    if kind == "names":
        base.names[0] = "different"
    if kind == "classes":
        head.nc = 3
    if kind == "stride":
        head.stride[0] = 4
    if kind == "family":
        base.yaml["backbone"][0][2] = "C3k2"
    if kind == "end2end":
        head.end2end = True
    if kind == "subclass":
        head.__class__ = type("Segment", (Detect,), {})
    if kind == "scales":
        head.nl = 4
    if kind == "alias":
        base.neck[0].weight = nn.Parameter(
            student.wrapped_model.neck[0].weight.detach()
        )
    with pytest.raises(ValueError):
        make(pair)


def test_batch_sum_and_normalization(pair):
    model, objective = make(pair, 2)
    data = batch(1)
    output = model(**data)
    objective(output, data)
    first = objective.last_metrics.copy()
    data2 = {**data, "img": data["img"].repeat(2, 1, 1, 1)}
    objective(model(**data2), data2)
    assert objective.last_metrics["distillation"] == pytest.approx(
        2 * first["distillation"], rel=1e-5
    )


def test_exact_normalized_feature_loss_and_unchanged_supervised(pair):
    model, objective = make(pair, 2.5)
    data = batch()
    output = model(**data)
    with torch.no_grad():
        _, targets = kd._capture(
            pair[1].wrapped_model.model[-1],
            lambda: pair[1].wrapped_model.predict(data["img"]),
        )

    def normalize(x):
        return x / (x.square().mean((1, 2, 3), keepdim=True) + 1e-6).sqrt()

    expected = (
        sum(
            (normalize(s) - normalize(t)).square().mean()
            for s, t in zip(output.features, targets)
        )
        / 3
        * 2
        * 2.5
    )
    seen = []
    supervised = output.original[0]

    def loss_fn(original, supplied_batch):
        seen.append((original is output.original, supplied_batch is data))
        return supervised

    objective.detection_loss_fn = loss_fn
    total = objective(output, data)
    assert seen == [(True, True)]
    torch.testing.assert_close(total, supervised + expected)


def test_projector_checkpoint_roundtrip_and_update(pair):
    model, objective = make(pair)
    before = [p.detach().clone() for p in model.projectors.parameters()]
    student_before = [p.detach().clone() for p in model.student.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = batch()
    objective(model(**data), data).backward()
    optimizer.step()
    assert any(
        not torch.equal(p, old) for p, old in zip(model.projectors.parameters(), before)
    )
    assert any(
        not torch.equal(p, old)
        for p, old in zip(model.student.parameters(), student_before)
    )
    restored, _ = make((Proxy(), Proxy(6)))
    restored.load_state_dict(model.state_dict())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])


def test_teacher_batchnorm_buffers_unchanged(pair):
    teacher = pair[1]
    teacher.wrapped_model.neck = nn.ModuleList(
        [
            nn.Sequential(conv, nn.BatchNorm2d(conv.out_channels))
            for conv in teacher.wrapped_model.neck
        ]
    )
    model, objective = make(pair)
    before = {name: buffer.clone() for name, buffer in teacher.named_buffers()}
    bn = [module for module in teacher.modules() if isinstance(module, nn.BatchNorm2d)]
    assert len(bn) == 3
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(2):
        model.train().to("cpu")
        assert all(not module.training for module in bn)
        optimizer.zero_grad()
        data = batch()
        objective(model(**data), data).backward()
        optimizer.step()
    assert teacher.wrapped_model.calls == 2
    for name, buffer in teacher.named_buffers():
        assert torch.equal(buffer, before[name]), name


def test_teacher_lazy_fp32_and_config_snapshot(pair):
    pair[1].half()
    config = DetectionDistillationConfig(feature_weight=1.0)
    model, objective = kd.create_detection_distillation(
        *pair, config, lambda o, b: o[0]
    )
    config.feature_weight = 0
    assert next(pair[1].parameters()).dtype == torch.float16
    data = batch()
    objective(model(**data), data)
    assert next(pair[1].parameters()).dtype == torch.float32
    assert pair[1].wrapped_model.calls == 1


def test_missing_capture_and_shape_mismatch_cleanup(pair):
    model, objective = make(pair)
    data = batch()
    output = model(**data)
    malformed = output._replace(
        features=tuple(x[:, :, :1, :1] for x in output.features)
    )
    with pytest.raises(ValueError, match="shapes"):
        objective(malformed, data)
    pair[0].wrapped_model.predict = lambda img: img.mean()
    with pytest.raises(ValueError, match="invocation"):
        model(**data)
    assert all(not p.wrapped_model.model[-1]._forward_pre_hooks for p in pair)


def test_family_allows_scale_width_not_semantic_changes(pair):
    teacher = pair[1].wrapped_model
    teacher.yaml.update(
        scale="x", width_multiple=1.25, depth_multiple=1.33, yaml_file="renamed.yaml"
    )
    make(pair)
    teacher.yaml["backbone"][0][3][1] = 5
    with pytest.raises(ValueError, match="family"):
        make(pair)


@pytest.mark.parametrize("kind", ["identity", "buffer"])
def test_shared_state_rejected_before_freezing(pair, kind):
    if kind == "identity":
        pair = (pair[0], pair[0])
    else:
        pair[1].wrapped_model.model[-1].stride = (
            pair[0].wrapped_model.model[-1].stride.view(3)
        )
    with pytest.raises(ValueError, match="share"):
        make(pair)
    assert all(p.requires_grad for p in pair[0].parameters())


def test_actual_ultralytics_optional():
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel as RealModel
    from ultralytics.utils import DEFAULT_CFG

    student, teacher = Proxy(), Proxy()
    student.wrapped_model = RealModel("yolov8n.yaml", nc=2, verbose=False)
    teacher.wrapped_model = RealModel("yolov8s.yaml", nc=2, verbose=False)
    student.wrapped_model.args = copy.deepcopy(DEFAULT_CFG)
    model, objective = make((student, teacher))
    data = {"img": torch.randn(2, 3, 64, 64), "official_loss": True}
    # Real official loss is invoked by this proxy replacement, without downloads.
    student.forward = lambda **b: student.wrapped_model(
        {
            **b,
            "cls": torch.empty(0, 1),
            "bboxes": torch.empty(0, 4),
            "batch_idx": torch.empty(0),
        }
    )
    loss = objective(model(**data), data)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.projectors.parameters())


@pytest.mark.parametrize("requested", ["cpu", "cuda:1"])
def test_benchmark_rejects_wrong_measurement_device(monkeypatch, requested):
    from tests.benchmarks.scripts.validate_detection_distillation import (
        validate_native_device,
    )

    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="Native backend selects"):
        validate_native_device(torch, torch.device(requested))
    validate_native_device(torch, torch.device("cuda:0"))
