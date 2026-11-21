# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Token-summed objective contracts before distributed stage execution."""

import pytest

torch = pytest.importorskip("torch")


def causal_loss(**kwargs):
    from parascale.parallel.loss import CausalLMLoss

    return CausalLMLoss(vocab_size=7, **kwargs)


def test_causal_sum_matches_shifted_cross_entropy_and_gradients():
    torch.manual_seed(3)
    logits = torch.randn(2, 4, 7, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    labels = torch.tensor([[6, 2, -100, 3], [5, -100, 1, 0]])
    objective = causal_loss()
    actual = objective(logits, labels)
    expected = torch.nn.functional.cross_entropy(
        reference[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1), reduction="sum"
    )
    assert objective.normalization_count(labels).item() == 4
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, reference.grad)


def test_empty_labels_keep_a_connected_zero_gradient():
    logits = torch.randn(2, 4, 7, requires_grad=True)
    labels = torch.full((2, 4), -100, dtype=torch.int64)
    objective = causal_loss()
    assert objective.normalization_count(labels).item() == 0
    loss = objective(logits, labels)
    assert loss.item() == 0
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("value", [-1, 7, 100])
def test_invalid_scored_labels_rejected_before_execution(value):
    labels = torch.tensor([[0, value, 1]])
    with pytest.raises(ValueError, match="vocabulary"):
        causal_loss().normalization_count(labels)


@pytest.mark.parametrize(
    "labels",
    [
        torch.ones(2, 3),
        torch.ones(3, dtype=torch.int64),
        torch.ones(2, 1, dtype=torch.int64),
    ],
)
def test_label_shape_and_dtype_checked(labels):
    with pytest.raises(ValueError, match="labels"):
        causal_loss().normalization_count(labels)


@pytest.mark.parametrize("vocab_size", [0, -1, True, 2.5])
def test_invalid_vocabulary_size_rejected(vocab_size):
    from parascale.parallel.loss import CausalLMLoss

    with pytest.raises(ValueError, match="vocab_size"):
        CausalLMLoss(vocab_size)


def test_loss_checks_output_contract():
    objective = causal_loss()
    labels = torch.zeros(2, 3, dtype=torch.int64)
    for output in (
        torch.zeros(2, 3, 8),
        torch.zeros(2, 2, 7),
        torch.zeros(2, 3, 7, dtype=torch.int64),
    ):
        with pytest.raises(ValueError, match="logits"):
            objective(output, labels)


@pytest.mark.parametrize(
    "stage,dtype,accepted",
    [
        (0, torch.int64, True),
        (1, torch.int64, False),
        (0, torch.float32, True),
        (1, torch.float32, True),
        (0, torch.float16, False),
    ],
)
def test_stage_input_dtype_is_independent_of_parameter_dtype(stage, dtype, accepted):
    from parascale.parallel.execution import _validate_example_input

    example = torch.zeros(2, 3, dtype=dtype)
    if accepted:
        _validate_example_input(example, stage)
    else:
        with pytest.raises(ValueError, match="input"):
            _validate_example_input(example, stage)


def test_summed_microbatch_gradients_use_global_token_denominator():
    from parascale.parallel.distributed import DistributedParallelConfig
    from parascale.parallel.execution import DistributedTrainer

    torch.manual_seed(5)
    values = torch.randn(4, 4, 7)
    labels = torch.tensor(
        [[0, -100, -100, -100], [0, 1, 2, -100], [0, 3, -100, -100], [0, 1, 4, 5]]
    )
    full = values.clone().requires_grad_()
    objective = causal_loss()
    count = objective.normalization_count(labels)
    (objective(full, labels) / count).backward()
    trainer = object.__new__(DistributedTrainer)
    trainer._precision = "fp32"
    trainer.device = torch.device("cpu")
    trainer._config = DistributedParallelConfig(dp_size=2, microbatches=2)
    trainer.loss_fn = None
    trainer.summed_loss = objective
    trainer._loss_scale = 2.0 / count
    actual = values.clone().requires_grad_()
    for batch, target in zip(actual.split(1), labels.split(1)):
        trainer._microbatch_loss(batch, target).backward()
    # DDP averages gradients over two replicas, not over token/microbatch counts.
    torch.testing.assert_close(actual.grad / 2, full.grad)


def test_mean_microbatch_contract_remains_explicit():
    from parascale.parallel.distributed import DistributedParallelConfig
    from parascale.parallel.execution import DistributedTrainer

    trainer = object.__new__(DistributedTrainer)
    trainer._precision = "fp32"
    trainer.device = torch.device("cpu")
    trainer._config = DistributedParallelConfig(microbatches=2)
    trainer.loss_fn = torch.nn.MSELoss()
    trainer.summed_loss = None
    assert trainer._microbatch_loss(torch.ones(2), torch.zeros(2)).item() == 0.5
