# -*- coding: utf-8 -*-
# @Time : 2026/11/13 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Numerical training contracts shared by native and accumulating backends."""

import pytest

torch = pytest.importorskip("torch")

from parascale.config import ParaScaleConfig  # noqa: E402
from parascale.runtime.backends.base import TrainingBackend  # noqa: E402
from parascale.runtime.training import TrainEngine  # noqa: E402


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, **kwargs):
        return self.weight * x


def mse(output, batch):
    return (output - batch["labels"]).square().mean()


def test_global_normalization_keeps_denominator_on_device(monkeypatch):
    from types import SimpleNamespace

    from parascale.runtime.training.loss_weight import global_normalization

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    work = SimpleNamespace(wait=lambda: None)

    def reduce(value, op, async_op):
        assert async_op is True
        value.mul_(2)
        return work

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    model = ScalarModel()
    denominator, replicas, pending = global_normalization(
        3.0, SimpleNamespace(name="native_ddp"), model
    )
    assert isinstance(denominator, torch.Tensor)
    assert denominator.device == model.weight.device
    assert not denominator.requires_grad
    assert replicas == 2
    assert pending is work
    torch.testing.assert_close(denominator, torch.tensor(6.0))


@pytest.mark.parametrize("accumulation", [1, 2])
def test_normalization_wait_follows_forward(monkeypatch, accumulation):
    from types import SimpleNamespace

    from parascale.runtime.training import accumulation as accumulation_module
    from parascale.runtime.training import step as step_module

    events = []
    model = ScalarModel()
    model.register_forward_hook(lambda *args: events.append("forward"))
    work = SimpleNamespace(wait=lambda: events.append("wait"))
    module = accumulation_module if accumulation > 1 else step_module
    monkeypatch.setattr(
        module,
        "global_normalization",
        lambda weight, backend, model: (torch.tensor(weight), 1, work),
    )

    def loss(output, batch):
        events.append("loss")
        return mse(output, batch)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    TrainEngine(ParaScaleConfig(gradient_accumulation_steps=accumulation)).fit(
        [{"x": torch.ones(1), "labels": torch.ones(1)}] * accumulation,
        max_steps=1,
        model=model,
        optimizer=optimizer,
        loss_fn=loss,
    )
    assert events.index("forward") < events.index("wait")
    assert events.count("wait") == 1
    assert model.weight.item() == pytest.approx(0.2)


def test_unequal_microbatches_match_full_batch_update():
    batches = [
        {"x": torch.ones(1), "labels": torch.ones(1)},
        {"x": torch.full((3,), 3.0), "labels": torch.ones(3)},
    ]
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = TrainEngine(ParaScaleConfig(gradient_accumulation_steps=2))
    engine.fit(batches, max_steps=1, model=model, optimizer=optimizer, loss_fn=mse)
    assert model.weight.item() == pytest.approx(0.5)


def test_explicit_valid_token_weight_controls_accumulation():
    batches = [
        {"x": torch.ones(1, 2), "labels": torch.ones(1, 2), "valid": 2},
        {"x": torch.full((1, 6), 3.0), "labels": torch.ones(1, 6), "valid": 6},
    ]

    def loss(output, batch):
        return mse(output, batch)

    loss.normalization_weight = lambda batch: batch["valid"]
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    TrainEngine(ParaScaleConfig(gradient_accumulation_steps=2)).fit(
        batches, max_steps=1, model=model, optimizer=optimizer, loss_fn=loss
    )
    assert model.weight.item() == pytest.approx(0.5)


def test_native_applies_configured_gradient_clipping():
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    TrainEngine(ParaScaleConfig(grad_clip_norm=0.1)).fit(
        [{"x": torch.ones(1), "labels": torch.full((1,), 100.0)}],
        max_steps=1,
        model=model,
        optimizer=optimizer,
        loss_fn=mse,
    )
    assert model.weight.item() == pytest.approx(0.01)


def test_accumulation_reuses_equal_weight_divisors_and_detaches_metrics(monkeypatch):
    from parascale.runtime.training import accumulation as module

    divisions = []

    class Denominator:
        def __truediv__(self, value):
            divisions.append(value)
            return 2.0 / value

    monkeypatch.setattr(
        module, "global_normalization", lambda *args: (Denominator(), 1, None)
    )

    def metric(value):
        assert not value.requires_grad
        return value.item()

    monkeypatch.setattr(module, "metric_value", metric)
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = TrainEngine(ParaScaleConfig(gradient_accumulation_steps=2)).fit(
        [{"x": torch.ones(1), "labels": torch.ones(1)}] * 2,
        max_steps=1,
        model=model,
        optimizer=optimizer,
        loss_fn=mse,
    )
    assert divisions == [1.0]
    assert state.last_metrics["loss"] == pytest.approx(1.0)
    assert model.weight.item() == pytest.approx(0.2)


class AccumulatingBackend(TrainingBackend):
    name = "deepspeed"
    manages_gradient_accumulation = True

    def __init__(self, model, optimizer, config):
        super().__init__(model, optimizer, config)
        self.micro_steps = 0
        self.updates = 0

    def backward(self, loss):
        (loss / self.config.gradient_accumulation_steps).backward()

    def step(self, optimizer=None):
        self.micro_steps += 1
        if self.micro_steps % self.config.gradient_accumulation_steps:
            return False
        super().step(optimizer)
        self.updates += 1
        return True

    def no_sync(self):
        raise AssertionError("ZeRO accumulation must not use DDP no_sync")


def test_backend_managed_accumulation_counts_optimizer_updates():
    config = ParaScaleConfig(
        training_backend="deepspeed", gradient_accumulation_steps=2
    )
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    backend = AccumulatingBackend(model, optimizer, config)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
    engine = TrainEngine(config, training_backend=backend)
    batches = [{"x": torch.ones(1), "labels": torch.ones(1)} for _ in range(8)]
    engine.fit(
        batches,
        max_steps=4,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=mse,
    )
    assert backend.micro_steps == 8
    assert backend.updates == engine.state.global_step == scheduler.last_epoch == 4
    assert engine.state.consumed_micro_batches == 8
    assert model.weight.item() == pytest.approx(1 - 0.8**4)


@pytest.mark.parametrize("mode", ["ptq", "qat"])
def test_engine_rejects_unprepared_quantization(mode):
    model = torch.nn.Linear(2, 1)
    config = ParaScaleConfig(quantization={"enabled": True, "mode": mode, "bits": 4})
    with pytest.raises(ValueError, match="PTQ|prepared|QAT"):
        TrainEngine(config).fit(
            [torch.ones(1, 2)],
            max_steps=1,
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_fn=lambda out, batch: out.square().mean(),
        )


def test_amp_unscales_before_clipping_and_preserves_loaded_state():
    from parascale.runtime.training.precision import PrecisionController

    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = ParaScaleConfig(precision="fp16", grad_clip_norm=0.1)
    backend = TrainingBackend(model, optimizer, config)
    backend.name = "native"
    engine = TrainEngine(config, training_backend=backend)
    calls = []

    class Scaler:
        scale = 8

        def get_scale(self):
            return self.scale

        def unscale_(self, opt):
            calls.append("unscale")
            model.weight.grad.div_(self.scale)

        def step(self, opt):
            calls.append("step")
            assert model.weight.grad.abs().item() == pytest.approx(0.1)
            opt.step()

        def update(self):
            calls.append("update")

        def state_dict(self):
            return {"scale": self.scale}

        def load_state_dict(self, state):
            self.scale = state["scale"]

    engine.amp_scaler = Scaler()
    engine.precision = PrecisionController(engine)
    engine.precision.load_state_dict({"amp_scaler": {"scale": 16}})
    engine.precision.setup_scaler()
    assert engine.precision.state_dict() == {"amp_scaler": {"scale": 16}}
    model.weight.grad = torch.tensor(-200.0 * 16)
    assert engine.step(optimizer)
    assert calls == ["unscale", "step", "update"]
    assert model.weight.item() == pytest.approx(0.01)


def test_fsdp_clipping_delegates_global_norm():
    from parascale.runtime.backends.fsdp import FSDPTrainingBackend

    class Model:
        calls = []

        def clip_grad_norm_(self, norm):
            self.calls.append(norm)

    model = Model()
    FSDPTrainingBackend(model=model).clip_gradients(0.3)
    assert model.calls == [0.3]


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf")])
def test_invalid_loss_weight_fails_before_update(weight):
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def loss(output, batch):
        return mse(output, batch)

    loss.normalization_weight = lambda batch: weight
    with pytest.raises(ValueError, match="normalization weight"):
        TrainEngine(ParaScaleConfig()).fit(
            [{"x": torch.ones(1), "labels": torch.ones(1)}],
            max_steps=1,
            model=model,
            optimizer=optimizer,
            loss_fn=loss,
        )
    assert model.weight.item() == 0
