# -*- coding: utf-8 -*-
# @Time : 2026/11/26 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Independent evaluation accounting, finite metrics and replay isolation."""

import copy
import math
import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from parascale.data.text import TokenBatchCursor  # noqa: E402
from tests.test_text_cursor import corpus  # noqa: E402


def setup(tmp_path, *, error=None):
    cursor = TokenBatchCursor(
        corpus(tmp_path),
        batch_size=2,
        sequence_length=4,
        world_size=2,
        shuffle=False,
        repeat=False,
    )
    calls = []

    def evaluate(inputs, labels, *, return_outputs):
        assert return_outputs is False
        assert torch.equal(inputs, labels)
        random.random()
        torch.rand(1)
        if error:
            raise error
        calls.append(inputs.clone())
        return SimpleNamespace(
            loss=3.0 if len(calls) % 2 else 1.0,
            normalization_count=5 if len(calls) % 2 else 15,
        )

    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        config=SimpleNamespace(dp_size=2, microbatches=2),
        evaluate=evaluate,
    )
    return trainer, cursor, calls


def test_evaluation_weights_tokens_and_restores_cursor_and_rng(tmp_path):
    from parascale.runtime.training.stage_evaluation import evaluate_stage

    trainer, cursor, calls = setup(tmp_path)
    cursor.position = 1
    before = copy.deepcopy(cursor.state_dict())
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    result = evaluate_stage(trainer, cursor, max_batches=2, global_step=17)
    assert cursor.state_dict() == before
    assert random.getstate() == python_rng
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
    assert len(calls) == 2
    assert result["nll"] == 1.5
    assert result["perplexity"] == pytest.approx(math.exp(1.5))
    assert result["scored_tokens"] == 20
    assert result["input_tokens"] == 64
    assert result["block_count"] == 16
    assert result["covered_block_range"] == [0, 16]
    assert result["excluded_tokens"] == len(cursor.corpus.tokens) - 64
    assert result["global_step"] == 17
    assert result["sequence_length"] == 4
    assert result["corpus_identity"] == cursor.corpus.identity
    assert result["seconds"] > 0
    repeated = evaluate_stage(trainer, cursor, max_batches=2, global_step=18)
    assert repeated["nll"] == result["nll"]
    torch.testing.assert_close(calls[0], calls[2], rtol=0, atol=0)


def test_evaluation_exception_restores_cursor_and_rng(tmp_path):
    from parascale.runtime.training.stage_evaluation import evaluate_stage

    trainer, cursor, _ = setup(tmp_path, error=RuntimeError("evaluation failed"))
    state = copy.deepcopy(cursor.state_dict())
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="evaluation failed"):
        evaluate_stage(trainer, cursor, max_batches=2, global_step=0)
    assert cursor.state_dict() == state
    assert random.getstate() == python_rng
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1000.0])
def test_evaluation_rejects_nonfinite_loss_or_perplexity(tmp_path, value):
    from parascale.runtime.training.stage_evaluation import evaluate_stage

    trainer, cursor, _ = setup(tmp_path)
    trainer.evaluate = lambda *args, **kwargs: SimpleNamespace(
        loss=value, normalization_count=4
    )
    with pytest.raises(FloatingPointError):
        evaluate_stage(trainer, cursor, max_batches=1, global_step=0)
    assert cursor.position == 0


@pytest.mark.parametrize("count", [None, 0, -1, True, 1.5])
def test_evaluation_rejects_invalid_scored_count(tmp_path, count):
    from parascale.runtime.training.stage_evaluation import evaluate_stage

    trainer, cursor, _ = setup(tmp_path)
    trainer.evaluate = lambda *args, **kwargs: SimpleNamespace(
        loss=1.0, normalization_count=count
    )
    with pytest.raises(ValueError, match="scored"):
        evaluate_stage(trainer, cursor, max_batches=1, global_step=0)


@pytest.mark.parametrize("change", ["shuffle", "repeat", "exhaust", "budget", "step"])
def test_evaluation_rejects_invalid_plan_before_forward(tmp_path, change):
    from parascale.runtime.training.stage_evaluation import evaluate_stage

    trainer, cursor, calls = setup(tmp_path)
    budget, step = 1, 0
    if change in ("shuffle", "repeat"):
        setattr(cursor, change, True)
    elif change == "exhaust":
        budget = cursor.batches_per_epoch
    elif change == "budget":
        budget = True
    else:
        step = -1
    with pytest.raises(ValueError):
        evaluate_stage(trainer, cursor, max_batches=budget, global_step=step)
    assert not calls
