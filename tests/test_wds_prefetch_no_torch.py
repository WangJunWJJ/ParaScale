# -*- coding: utf-8 -*-
# @Time : 2026/11/23 上午12:50
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""CPU lookahead must never advance the consumed WDS cursor."""

import io
import tarfile
import threading
from types import SimpleNamespace

import pytest

from parascale import ParaScaleConfig
from parascale.data.wds_cursor import WdsCursorBatches
from parascale.runtime.training.engine import TrainEngine


def make_loader(tmp_path, transform, **kwargs):
    path = tmp_path / "samples.tar"
    if not path.exists():
        with tarfile.open(path, "w") as archive:
            for i in range(6):
                for suffix, blob in (("jpg", b"image"), ("txt", str(i).encode())):
                    info = tarfile.TarInfo(f"{i}.{suffix}")
                    info.size = len(blob)
                    archive.addfile(info, io.BytesIO(blob))
    spec = SimpleNamespace(
        data_dir=str(tmp_path),
        batch_size=2,
        num_batches=3,
        num_samples=5,
        pretrained_model_name_or_path="clip",
        text_length=77,
    )
    return WdsCursorBatches(spec, transform, **kwargs)


def test_prefetch_is_one_batch_ahead_with_consumed_only_state(tmp_path):
    ready = threading.Event()
    calls = []

    def transform(batch):
        result = [row["text"] for row in batch]
        calls.append(result)
        if result[0] == "2":
            ready.set()
        return result

    loader = make_loader(tmp_path, transform, prefetch=True)
    iterator = iter(loader)
    try:
        assert next(iterator) == ["0", "1"]
        assert ready.wait(5)
        state = loader.state_dict()
        assert state["position"] == 2
        assert calls == [["0", "1"], ["2", "3"]]
    finally:
        iterator.close()
    assert not any(t.name.startswith("parascale-wds") for t in threading.enumerate())
    restored = make_loader(tmp_path, transform, prefetch=True)
    restored.load_state_dict(state)
    assert list(restored)[0] == ["2", "3"]
    plain = make_loader(tmp_path, transform)
    prefetched = make_loader(tmp_path, transform, prefetch=True)
    assert list(plain) == list(prefetched) == [["0", "1"], ["2", "3"], ["4"]]
    assert plain.state_dict() == prefetched.state_dict()


def test_prefetch_error_does_not_advance_cursor_or_leave_worker(tmp_path):
    def transform(batch):
        if batch[0]["text"] == "2":
            raise ValueError("bad input")
        return batch

    loader = make_loader(tmp_path, transform, prefetch=True)
    iterator = iter(loader)
    next(iterator)
    with pytest.raises(ValueError, match="bad input"):
        next(iterator)
    assert loader.position == 2
    assert not any(t.name.startswith("parascale-wds") for t in threading.enumerate())


def test_active_iterator_rejects_concurrent_use_and_restore(tmp_path):
    loader = make_loader(tmp_path, lambda batch: batch, prefetch=True)
    iterator = iter(loader)
    next(iterator)
    state = loader.state_dict()
    with pytest.raises(RuntimeError, match="one WDS iterator"):
        next(iter(loader))
    with pytest.raises(RuntimeError, match="Close the active"):
        loader.load_state_dict(state)
    iterator.close()
    loader.load_state_dict(state)


def test_lookahead_submission_failure_does_not_commit_undelivered_batch(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    class FailingExecutor(ThreadPoolExecutor):
        calls = 0

        def submit(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("submit failed")
            return super().submit(*args, **kwargs)

    monkeypatch.setattr("parascale.data.wds_cursor.ThreadPoolExecutor", FailingExecutor)
    loader = make_loader(tmp_path, lambda batch: batch, prefetch=True)
    with pytest.raises(RuntimeError, match="submit failed"):
        next(iter(loader))
    assert loader.position == 0
    assert not any(t.name.startswith("parascale-wds") for t in threading.enumerate())


def test_repeated_fit_closes_unconsumed_lookahead(tmp_path):
    loader = make_loader(tmp_path, lambda batch: batch, prefetch=True)
    engine = TrainEngine(ParaScaleConfig())
    consumed = []

    def step(batch):
        consumed.append([item["text"] for item in batch])
        return {"loss": 1.0, "batch_size": len(batch)}

    engine.fit(loader, max_steps=1, step_fn=step)
    engine.fit(loader, max_steps=1, step_fn=step)
    assert consumed == [["0", "1"], ["2", "3"]]
    assert engine.data_resume.capture()["state"]["position"] == 4
    assert not any(t.name.startswith("parascale-wds") for t in threading.enumerate())


def test_cpu_prefetch_config_and_unsupported_sources():
    from parascale.workloads.clip import build_clip_contrastive_components
    from parascale.workloads.specs.clip import ClipContrastiveSpec

    spec = ClipContrastiveSpec.from_config({"data": {"cpu_prefetch": True}})
    assert spec.cpu_prefetch
    with pytest.raises(ValueError, match="CPU lookahead requires"):
        build_clip_contrastive_components(spec)
    spec = ClipContrastiveSpec.from_config(
        {
            "parascale": {"device_prefetch": True},
            "model": {"type": "hf_clip"},
            "data": {"cpu_prefetch": True, "streaming": True},
        }
    )
    with pytest.raises(ValueError, match="CPU lookahead requires"):
        build_clip_contrastive_components(spec)


def test_fit_does_not_close_borrowed_one_shot_generator():
    batches = (value for value in range(3))
    engine = TrainEngine(ParaScaleConfig())
    consumed = []

    def step(batch):
        consumed.append(batch)
        return {"loss": 1.0, "batch_size": 1}

    engine.fit(batches, max_steps=1, step_fn=step)
    engine.fit(batches, max_steps=1, step_fn=step)
    assert consumed == [0, 1]


def test_unstarted_superseded_tracker_does_not_close_borrowed_source():
    class Source:
        closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return 1

        def state_dict(self):
            return {"position": 0}

        def close(self):
            self.closed = True

    engine = TrainEngine(ParaScaleConfig())
    borrowed = Source()
    old = engine.data_resume.prepare_iterator(borrowed)
    engine.data_resume.prepare_iterator(Source())
    with pytest.raises(RuntimeError, match="superseded"):
        next(old)
    assert not borrowed.closed


@pytest.mark.parametrize("fail", [False, True])
def test_fit_closes_stateful_source_on_limit_and_exception(tmp_path, fail):
    loader = make_loader(tmp_path, lambda batch: batch, prefetch=True)
    engine = TrainEngine(ParaScaleConfig())

    def step(batch):
        if fail:
            raise ValueError("failed step")
        return {"loss": 1.0, "batch_size": len(batch)}

    if fail:
        with pytest.raises(ValueError, match="failed step"):
            engine.fit(loader, max_steps=1, step_fn=step)
    else:
        engine.fit(loader, max_steps=1, step_fn=step)
    assert not any(t.name.startswith("parascale-wds") for t in threading.enumerate())
    assert engine.data_resume.capture()["state"]["position"] == (0 if fail else 2)


@pytest.mark.parametrize("accumulation", [1, 2])
def test_prefetch_real_checkpoint_resume_matches_next_update(tmp_path, accumulation):
    torch = pytest.importorskip("torch")
    from parascale.checkpoint import CheckpointManager

    def setup(prefetch):
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(1, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        engine = TrainEngine(
            ParaScaleConfig(
                training_backend="native",
                precision="fp32",
                gradient_accumulation_steps=accumulation,
            )
        )
        loader = make_loader(
            tmp_path,
            lambda batch: torch.tensor([[float(row["text"]) / 6] for row in batch]),
            prefetch=prefetch,
        )
        loader.spec.num_samples = 24
        loader.spec.num_batches = 12
        return engine, model, optimizer, loader

    def train(parts, steps):
        engine, model, optimizer, loader = parts
        engine.fit(
            loader,
            max_steps=steps,
            model=model,
            optimizer=optimizer,
            loss_fn=lambda output, batch: output.square().mean(),
        )

    continuous = setup(False)
    train(continuous, 3)
    expected_rng = torch.get_rng_state().clone()
    split = setup(True)
    train(split, 1)
    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    split[0].save_checkpoint(manager)
    assert manager.validate(1).ok
    restored = setup(True)
    restored[0].initialize()
    restored[0].load_checkpoint(
        manager, 1, model=restored[1], optimizer=restored[2], dataloader=restored[3]
    )
    train(restored, 2)
    torch.testing.assert_close(
        restored[1].state_dict(), continuous[1].state_dict(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        restored[2].state_dict(), continuous[2].state_dict(), atol=0, rtol=0
    )
    assert restored[0].data_resume.capture() == continuous[0].data_resume.capture()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert not any(t.name.startswith("parascale-wds") for t in threading.enumerate())
