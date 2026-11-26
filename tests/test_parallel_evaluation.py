# -*- coding: utf-8 -*-
# @Time : 2026/11/26 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Forward-only stage evaluation must not change the training trajectory."""

import copy

import pytest

from parascale.parallel.distributed import DistributedParallelConfig
from parascale.parallel.execution import DistributedTrainer
from parascale.parallel.loss import CausalLMLoss

torch = pytest.importorskip("torch")


def make_trainer(precision="fp32"):
    trainer = object.__new__(DistributedTrainer)
    trainer._config = DistributedParallelConfig(microbatches=2)
    trainer._precision = precision
    trainer.device = torch.device("cpu")
    trainer.input_shape = (2, 3, 4)
    trainer.input_dtype = torch.float32
    trainer.module = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.Dropout(0.3))
    trainer.execution_module = trainer.module
    trainer.summed_loss = CausalLMLoss(vocab_size=7)
    trainer.loss_fn = None
    trainer.schedule = None
    trainer.pp_rank = trainer.dp_rank = trainer.tp_rank = 0
    trainer._normalization_count = 913
    trainer._loss_scale = 0.125
    return trainer


def batches():
    inputs = torch.randn(4, 3, 4)
    labels = torch.tensor([[0, -100, -100], [0, 2, -100], [0, 3, 4], [0, 5, 6]])
    return inputs, labels


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("ambient", [False, True])
@pytest.mark.parametrize("outputs", [False, True])
def test_evaluate_weighted_loss_preserves_modes_gradients_and_precision(
    precision, ambient, outputs
):
    trainer = make_trainer(precision)
    trainer.module[0].eval()
    modes = [module.training for module in trainer.module.modules()]
    gradients = []
    for parameter in trainer.parameters():
        parameter.grad = torch.randn_like(parameter)
        gradients.append((parameter.grad, parameter.grad.clone()))
    inputs, labels = batches()
    reference = copy.deepcopy(trainer.module).eval()
    with (
        torch.no_grad(),
        torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"),
    ):
        predictions = torch.cat([reference(batch) for batch in inputs.chunk(2)])
        expected = trainer.summed_loss(predictions, labels) / 5
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=ambient):
        result = trainer.evaluate(inputs, labels, return_outputs=outputs)
        assert torch.is_autocast_enabled("cpu") == ambient
    assert result.loss == pytest.approx(expected.item(), rel=1e-6)
    assert result.normalization_count == 5
    assert (result.output is not None) == outputs
    if outputs:
        torch.testing.assert_close(result.output, predictions, rtol=0, atol=0)
        assert not result.output.requires_grad
    assert [module.training for module in trainer.module.modules()] == modes
    for parameter, (original, value) in zip(trainer.parameters(), gradients):
        assert parameter.grad is original
        torch.testing.assert_close(parameter.grad, value, rtol=0, atol=0)
    assert trainer._normalization_count == 913
    assert trainer._loss_scale == 0.125


@pytest.mark.parametrize("failure", [False, True])
def test_evaluate_preserves_rng_and_training_state_on_forward_failure(failure):
    trainer = make_trainer()
    inputs, labels = batches()
    modes = [module.training for module in trainer.module.modules()]

    def random_forward(module, args):
        assert not torch.is_grad_enabled()
        assert not module.training
        torch.rand(13)
        if failure:
            raise RuntimeError("injected evaluation failure")

    handle = trainer.module.register_forward_pre_hook(random_forward)
    rng = torch.get_rng_state().clone()
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            trainer.evaluate(inputs, labels)
    else:
        trainer.evaluate(inputs, labels)
    handle.remove()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert [module.training for module in trainer.module.modules()] == modes
    assert trainer._normalization_count == 913
    assert trainer._loss_scale == 0.125
    assert all(parameter.grad is None for parameter in trainer.parameters())


def test_evaluation_does_not_change_next_training_update():
    trainer = make_trainer()
    reference = copy.deepcopy(trainer)
    inputs, labels = batches()
    rng = torch.get_rng_state().clone()
    trainer.evaluate(inputs, labels)
    actual = trainer.forward_backward(inputs, labels)
    torch.set_rng_state(rng)
    expected = reference.forward_backward(inputs, labels)
    assert actual.loss == expected.loss
    for left, right in zip(trainer.parameters(), reference.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    for item in (trainer, reference):
        torch.optim.AdamW(item.parameters(), lr=0.01).step()
    for left, right in zip(trainer.parameters(), reference.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["empty", "shape", "nonfinite"])
def test_invalid_evaluation_restores_modes_and_loss_state(bad):
    trainer = make_trainer()
    inputs, labels = batches()
    if bad == "empty":
        labels.fill_(-100)
    elif bad == "shape":
        inputs = inputs[:1]
    else:
        inputs.fill_(float("nan"))
    with pytest.raises((ValueError, FloatingPointError)):
        trainer.evaluate(inputs, labels)
    assert trainer.module.training
    assert trainer._normalization_count == 913
    assert trainer._loss_scale == 0.125


def test_mean_loss_evaluation_uses_equal_microbatch_contract():
    trainer = make_trainer()
    trainer.summed_loss = None
    trainer.loss_fn = torch.nn.MSELoss()
    trainer._normalization_count = None
    inputs, _ = batches()
    targets = torch.randn(4, 3, 7)
    reference = copy.deepcopy(trainer.module).eval()
    with torch.no_grad():
        expected = trainer.loss_fn(reference(inputs), targets)
    result = trainer.evaluate(inputs, targets)
    assert result.loss == pytest.approx(expected.item(), rel=1e-6)
    assert result.normalization_count is None


@pytest.mark.parametrize("failure", [False, True])
def test_evaluation_reshards_fsdp_before_return_or_failure(monkeypatch, failure):
    from unittest.mock import Mock

    import torch.distributed.fsdp as fsdp

    trainer = make_trainer()
    monkeypatch.setattr(fsdp, "FSDPModule", type(trainer.module))
    trainer.module.reshard = Mock()
    inputs, labels = batches()
    if failure:
        inputs.fill_(float("nan"))
        with pytest.raises(FloatingPointError):
            trainer.evaluate(inputs, labels)
    else:
        trainer.evaluate(inputs, labels)
    trainer.module.reshard.assert_called_once_with()
