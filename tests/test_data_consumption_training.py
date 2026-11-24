# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Real optimizer/scheduler/RNG recovery with consumed data cursors."""

import copy
import random

import pytest

torch = pytest.importorskip("torch")

from parascale.checkpoint import CheckpointManager  # noqa: E402
from parascale.config import ParaScaleConfig  # noqa: E402
from parascale.runtime.training import TrainEngine  # noqa: E402


class RandomBatches:
    """Independent data RNG, checkpointed separately from model dropout RNG."""

    def __init__(self):
        self.position = 0
        self.rng = random.Random(713)

    def state_dict(self):
        return {"position": self.position, "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        self.position = state["position"]
        version, words, gaussian = state["rng"]
        self.rng.setstate((version, tuple(words), gaussian))

    def __iter__(self):
        while True:
            index = self.position
            value = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32
            value = value + self.rng.random()
            if torch.cuda.is_available():
                value = value.pin_memory()
            self.position += 1
            yield {"x": value, "sample_id": index, "batch_size": 4}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Dropout(0.25), torch.nn.Linear(8, 1))
        self.seen = []

    def forward(self, x, sample_id, **metadata):
        self.seen.append(sample_id)
        return self.layers(x)


class TorchBatches(torch.utils.data.DataLoader):
    def __init__(self):
        class Dataset(RandomBatches, torch.utils.data.IterableDataset):
            pass

        super().__init__(Dataset(), batch_size=None, num_workers=0)

    def state_dict(self):
        return self.dataset.state_dict()

    def load_state_dict(self, state):
        self.dataset.load_state_dict(state)


def assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_state_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("accumulation", [1, 3])
@pytest.mark.parametrize("restart", ["fit", "checkpoint"])
@pytest.mark.parametrize("loader_type", [RandomBatches, TorchBatches])
def test_training_consumption_matches_uninterrupted(
    tmp_path, prefetch, accumulation, restart, loader_type
):
    if prefetch and not torch.cuda.is_available():
        pytest.skip("CUDA stream prefetch requires an accelerator")

    def create():
        torch.manual_seed(321)
        model = Model()
        if torch.cuda.is_available():
            model = model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        engine = TrainEngine(
            ParaScaleConfig(
                training_backend="native",
                gradient_accumulation_steps=accumulation,
                device_prefetch=prefetch,
                precision="fp32",
            )
        )
        return engine, model, optimizer, scheduler, loader_type()

    def train(objects, steps):
        engine, model, optimizer, scheduler, data = objects
        engine.fit(
            data,
            max_steps=steps,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=lambda output, batch: output.square().mean(),
        )

    reference = create()
    train(reference, 4)
    expected_model = copy.deepcopy(reference[1].state_dict())
    expected_optimizer = copy.deepcopy(reference[2].state_dict())
    expected_scheduler = reference[3].state_dict()
    expected_data = reference[0].data_resume.capture()

    split = create()
    train(split, 2)
    prefix = list(split[1].seen)
    if restart == "checkpoint":
        manager = CheckpointManager(str(tmp_path / "checkpoint"))
        split[0].save_checkpoint(manager, scheduler=split[3])
        split = create()
        train(split, 0)
        assert split[0].data_resume.capture()["state"]["position"] == 0
        split[0].load_checkpoint(manager, 2, scheduler=split[3])
        assert split[0].data_resume.capture()["state"]["position"] == 2 * accumulation
    else:
        split[1].seen.clear()
    train(split, 2)
    assert prefix + split[1].seen == list(range(4 * accumulation))
    assert_state_equal(split[1].state_dict(), expected_model)
    assert_state_equal(split[2].state_dict(), expected_optimizer)
    assert split[3].state_dict() == expected_scheduler
    assert split[0].data_resume.capture() == expected_data
    assert split[0].state.global_step == 4
    assert split[0].state.consumed_micro_batches == 4 * accumulation
