# -*- coding: utf-8 -*-
# @Time : 2026/11/21 下午6:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Local contract tests; real multi-rank coverage lives in the GPU validator."""

import weakref
from contextlib import contextmanager

import pytest

from parascale.parallel import DistributedParallelConfig
from parascale.parallel.execution import DistributedTrainer


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_requires_caller_initialized_process_group(torch):
    with pytest.raises(RuntimeError, match="process group"):
        DistributedTrainer(
            torch.nn.Linear(2, 2),
            DistributedParallelConfig(),
            torch.zeros(1, 2),
            torch.nn.MSELoss(),
        )


def test_untested_torch_version_rejected_before_mutation(torch, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch, "__version__", "2.5.0a0+nv24.08")
    with pytest.raises(ValueError, match="torch==2.9.1"):
        DistributedTrainer(
            torch.nn.Linear(2, 2),
            DistributedParallelConfig(),
            torch.zeros(1, 2),
            torch.nn.MSELoss(),
        )


def test_microbatch_mean_loss_normalization(torch):
    trainer = DistributedTrainer.__new__(DistributedTrainer)
    trainer.config = DistributedParallelConfig(microbatches=4)
    trainer.loss_fn = torch.nn.MSELoss()
    trainer.summed_loss = None
    assert trainer._microbatch_loss(torch.ones(2, 3), torch.zeros(2, 3)).item() == 0.25


def test_non_scalar_loss_rejected(torch):
    trainer = DistributedTrainer.__new__(DistributedTrainer)
    trainer.config = DistributedParallelConfig()
    trainer.loss_fn = torch.nn.MSELoss(reduction="none")
    trainer.summed_loss = None
    with pytest.raises(ValueError, match="scalar"):
        trainer._microbatch_loss(torch.ones(2, 3), torch.zeros(2, 3))


@pytest.mark.parametrize("batch", [0, 1, 3, 5])
def test_uneven_batches_fail_before_collectives(torch, batch):
    trainer = DistributedTrainer.__new__(DistributedTrainer)
    trainer.config = DistributedParallelConfig(microbatches=2)
    trainer.input_shape = (2, 3)
    trainer.device = torch.device("cpu")
    with pytest.raises(ValueError, match="batch size"):
        trainer.forward_backward(torch.zeros(batch, 3), torch.zeros(batch, 3))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_numerical_validator_does_not_hide_nonfinite_errors(torch, value):
    import math

    from tests.benchmarks.scripts.validate_distributed_parallel import maximum_error

    assert math.isinf(maximum_error(torch.tensor([value]), torch.zeros(1)))


@pytest.mark.parametrize("dp,tp", [(1, 1), (2, 1), (2, 2)])
@pytest.mark.parametrize("return_outputs", [False, True])
def test_ddp_accumulation_syncs_only_final_forward_and_backward(
    torch, monkeypatch, dp, tp, return_outputs
):
    forward_sync, backward_sync = [], []
    predictions = []

    class Wrapped(torch.nn.Linear):
        syncing = True

        @contextmanager
        def no_sync(self):
            self.syncing = False
            try:
                yield
            finally:
                self.syncing = True

        def forward(self, inputs):
            if predictions and not return_outputs:
                assert predictions[-1]() is None
            forward_sync.append(self.syncing)
            prediction = super().forward(inputs)
            predictions.append(weakref.ref(prediction))
            return prediction

    model = Wrapped(2, 2)
    model.weight.register_hook(lambda grad: backward_sync.append(model.syncing))
    trainer = DistributedTrainer.__new__(DistributedTrainer)
    trainer.config = DistributedParallelConfig(dp_size=dp, tp_size=tp, microbatches=3)
    trainer.input_shape = (2, 2)
    trainer.device = torch.device("cpu")
    trainer.pp_rank = 0
    trainer.schedule = None
    trainer.execution_module = model
    trainer.module = model
    trainer.loss_fn = torch.nn.MSELoss()
    trainer.summed_loss = None
    trainer.input_dtype = torch.float32
    trainer._normalization_count = None

    class Mesh:
        def __getitem__(self, key):
            return self

        def get_group(self):
            return None

    trainer.mesh = Mesh()
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    inputs, targets = torch.randn(6, 2), torch.randn(6, 2)
    result = trainer.forward_backward(inputs, targets, return_outputs=return_outputs)
    assert (result.output is not None) == return_outputs
    if return_outputs:
        assert result.output.shape == targets.shape
        assert not result.output.requires_grad
    expected = [False, False, True] if dp == 2 and tp == 1 else [True] * 3
    assert forward_sync == expected
    assert backward_sync == expected
    gradient = model.weight.grad.clone()
    model.zero_grad()
    torch.nn.functional.mse_loss(model(inputs), targets).backward()
    torch.testing.assert_close(gradient, model.weight.grad)
