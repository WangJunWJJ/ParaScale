# -*- coding: utf-8 -*-
# @Time : 2026/11/16 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com
"""Three-scale standard YOLO neck KD, preserving the supplied detection loss."""

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch
from torch import nn

from .detection_config import DetectionDistillationConfig
from .online import _storage


def _ultralytics_types():
    from ultralytics.nn.modules import Detect
    from ultralytics.nn.tasks import DetectionModel

    return DetectionModel, Detect


def _family(base):
    """Compare topology and semantic args, not filenames or size multipliers.

    Only known channel-producing modules lose their output-width argument;
    kernel sizes, routing, base repeats and block options remain significant.
    Unknown/custom modules retain all arguments (conservative rejection).
    """
    yaml = getattr(base, "yaml", None)
    if (
        not isinstance(yaml, Mapping)
        or not yaml.get("backbone")
        or not yaml.get("head")
    ):
        raise ValueError("detection models require architecture YAML")
    channel_modules = {
        "Conv",
        "DWConv",
        "GhostConv",
        "Bottleneck",
        "GhostBottleneck",
        "SPP",
        "SPPF",
        "C1",
        "C2",
        "C2f",
        "C3",
        "C3k2",
        "C3TR",
        "C3Ghost",
        "C3x",
        "RepC3",
        "C2fAttn",
        "C2fCIB",
        "CIB",
        "AConv",
        "ADown",
        "SCDown",
        "PSA",
        "C2PSA",
        "C2fPSA",
        "A2C2f",
        "Focus",
        "RepNCSPELAN4",
    }

    def freeze(value):
        if isinstance(value, Mapping):
            return tuple(sorted((str(k), freeze(v)) for k, v in value.items()))
        if isinstance(value, (tuple, list)):
            return tuple(freeze(v) for v in value)
        if isinstance(value, type):
            return value.__name__
        return value

    topology = []
    for section in ("backbone", "head"):
        for layer in yaml[section]:
            if not isinstance(layer, (tuple, list)) or len(layer) != 4:
                raise ValueError("invalid detection architecture YAML layer")
            source, repeats, module, args = layer
            name = module if isinstance(module, str) else module.__name__
            args = list(args)
            if name in channel_modules and args:
                args[0] = "<channels>"
            topology.append(freeze((source, repeats, name, args)))
    ignored = {
        "backbone",
        "head",
        "nc",
        "scale",
        "scales",
        "depth_multiple",
        "width_multiple",
        "max_channels",
        "yaml_file",
        "ch",
        "channels",
    }
    return tuple(topology), freeze({k: v for k, v in yaml.items() if k not in ignored})


def _inspect(proxy):
    model_type, head_type = _ultralytics_types()
    base = getattr(proxy, "wrapped_model", None)
    if type(base) is not model_type:
        raise ValueError(
            "only standard Ultralytics DetectionModel proxies are supported"
        )
    head = base.model[-1]
    if type(head) is not head_type or getattr(head, "end2end", False):
        raise ValueError("only exact standard non-end2end Detect heads are supported")
    if head.nl != 3 or len(head.cv2) != 3:
        raise ValueError("detection KD requires three scales")
    channels = tuple(block[0].conv.in_channels for block in head.cv2)
    if any(type(c) is not int or c <= 0 for c in channels):
        raise ValueError("invalid Detect neck channels")
    names = getattr(base, "names", None)
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not isinstance(names, Mapping) or set(names) != set(range(head.nc)):
        raise ValueError("detection class names must cover all class indices")
    strides = head.stride.detach().float().cpu()
    if (
        strides.shape != (3,)
        or not torch.isfinite(strides).all()
        or not (strides > 0).all()
    ):
        raise ValueError("invalid three-scale detection strides")
    return base, head, channels, dict(names), strides


def _capture(head, forward):
    captured = []

    def prehook(module, args):
        # Detect replaces list entries in-place; snapshot before it does so.
        captured.append(tuple(args[0]))

    handle = head.register_forward_pre_hook(prehook)
    try:
        output = forward()
    finally:
        handle.remove()
    if len(captured) != 1 or len(captured[0]) != 3:
        raise ValueError("expected exactly one three-scale Detect invocation")
    if any(not isinstance(x, torch.Tensor) or x.ndim != 4 for x in captured[0]):
        raise ValueError("neck features must be BCHW tensors")
    return output, captured[0]


class DetectionDistillationOutput(NamedTuple):
    original: Any
    features: tuple


class DetectionDistillationModel(nn.Module):
    """Student and pre-created trainable 1x1 projections; never owns teacher.

    Export only ``wrapper.student.state_dict()`` for deployment. Training
    checkpoints use the full wrapper state, including projection parameters.
    """

    def __init__(self, student, student_channels, teacher_channels, active):
        super().__init__()
        self.student = student
        parameter = next(student.parameters())
        device = parameter.device
        # Projector initialization must not advance sampler/dropout RNG streams.
        devices = [] if device.type == "cpu" else [device.index]
        with torch.random.fork_rng(
            devices=devices,
            device_type="cuda" if device.type == "cpu" else device.type,
        ):
            self.projectors = nn.ModuleList(
                [
                    nn.Conv2d(s, t, 1, bias=False, device="cpu").to(
                        device=device, dtype=parameter.dtype
                    )
                    for s, t in zip(student_channels, teacher_channels)
                ]
            )
        self._active = active
        if not active:
            self.projectors.requires_grad_(False)

    def forward(self, *args, **kwargs):
        if kwargs.get("official_loss") is not True:
            raise ValueError("detection distillation requires official_loss=True")
        if not self._active:
            return DetectionDistillationOutput(self.student(*args, **kwargs), ())
        output, features = _capture(
            self.student.wrapped_model.model[-1], lambda: self.student(*args, **kwargs)
        )
        return DetectionDistillationOutput(
            output, tuple(p(f) for p, f in zip(self.projectors, features))
        )


class DetectionDistillationObjective:
    """Plain callable owning the frozen, lazily placed FP32 teacher.

    Each projected student/teacher stage is divided by its per-sample RMS
    over C,H,W (epsilon 1e-6). MSE averages all elements and then the three
    levels. Multiplication by B matches Ultralytics batch-summed loss units.
    Metrics report the weighted additive distillation term.
    """

    def __init__(self, teacher, config, detection_loss_fn):
        self.teacher = teacher
        self.config = DetectionDistillationConfig.from_dict(config.to_dict())
        self.detection_loss_fn = detection_loss_fn
        self.last_metrics = {}
        self._device = None
        teacher.requires_grad_(False)
        for parameter in teacher.parameters():
            parameter.grad = None
        teacher.eval()

    def __call__(self, output, batch):
        if not isinstance(output, DetectionDistillationOutput):
            raise TypeError("objective requires detection distillation wrapper output")
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping")
        supervised = self.detection_loss_fn(output.original, batch)
        if not isinstance(supervised, torch.Tensor) or supervised.ndim != 0:
            raise ValueError("detection_loss_fn must return a scalar tensor")
        distillation = supervised.new_zeros(())
        total = supervised
        if self.config.feature_weight:
            images = batch.get("img", batch.get("pixel_values"))
            if (
                not isinstance(images, torch.Tensor)
                or images.ndim != 4
                or images.shape[0] == 0
            ):
                raise ValueError("detection KD requires nonempty BCHW img")
            if len(output.features) != 3:
                raise ValueError("expected three projected neck features")
            device = output.features[0].device
            if self._device != device:
                self.teacher.to(device=device, dtype=torch.float32)
                self._device = device
            self.teacher.eval()
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type, enabled=False),
            ):
                _, targets = _capture(
                    self.teacher.wrapped_model.model[-1],
                    lambda: self.teacher.wrapped_model.predict(
                        images.to(device=device, dtype=torch.float32)
                    ),
                )
            terms = []
            with torch.autocast(device_type=device.type, enabled=False):
                for student, target in zip(output.features, targets):
                    if (
                        student.shape != target.shape
                        or student.shape[0] != images.shape[0]
                    ):
                        raise ValueError(
                            "student/teacher neck feature shapes must match after projection"
                        )

                    def normalize(x):
                        x = x.float()
                        return (
                            x
                            / x.square()
                            .mean(dim=(1, 2, 3), keepdim=True)
                            .add(1e-6)
                            .sqrt()
                        )

                    terms.append(
                        (normalize(student) - normalize(target)).square().mean()
                    )
                distillation = (
                    torch.stack(terms).mean()
                    * images.shape[0]
                    * self.config.feature_weight
                )
                total = supervised + distillation
        if not torch.isfinite(total).all():
            raise ValueError("detection distillation loss must be finite")
        self.last_metrics = {
            "total": total.detach().item(),
            "supervised": supervised.detach().item(),
            "distillation": distillation.detach().item(),
        }
        return total


def create_detection_distillation(student, teacher, config, detection_loss_fn):
    """Return (wrapped_model, objective); call before building the optimizer.

    Only independent, same-family, same-class, three-scale standard detectors
    are supported. ``feature_weight=0`` bypasses capture/projection/teacher work
    and returns the exact supplied supervised loss tensor.
    """
    if not isinstance(student, nn.Module) or not isinstance(teacher, nn.Module):
        raise TypeError("student and teacher must be torch modules")
    if not isinstance(config, DetectionDistillationConfig):
        raise TypeError("config must be DetectionDistillationConfig")
    config.validate()
    if not callable(detection_loss_fn):
        raise TypeError("detection_loss_fn must be callable")
    student_state = [*student.parameters(), *student.buffers()]
    teacher_state = [*teacher.parameters(), *teacher.buffers()]
    identities = {id(p) for p in student_state}
    storages = {_storage(p) for p in student_state} - {None}
    if student is teacher or any(
        id(p) in identities or _storage(p) in storages for p in teacher_state
    ):
        raise ValueError(
            "student and teacher must not share parameter/buffer identity or storage"
        )
    sb, sh, sc, sn, ss = _inspect(student)
    tb, th, tc, tn, ts = _inspect(teacher)
    if sh.nc != th.nc or sn != tn:
        raise ValueError("student and teacher must have identical classes and names")
    if not torch.equal(ss, ts):
        raise ValueError("student and teacher strides must match")
    if _family(sb) != _family(tb):
        raise ValueError("student and teacher must have the same architecture family")
    model = DetectionDistillationModel(student, sc, tc, config.feature_weight > 0)
    return model, DetectionDistillationObjective(teacher, config, detection_loss_fn)
