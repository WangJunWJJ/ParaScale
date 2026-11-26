# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Learning-rate schedules bind to optimizer updates, not invocation length."""

import json
import os
import time
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from parascale.checkpoint import CheckpointManager  # noqa: E402
from parascale.config import ParaScaleConfig  # noqa: E402
from parascale.runtime.train_runner import run_train_from_config  # noqa: E402
from parascale.runtime.training import TrainEngine  # noqa: E402


def configuration(path, steps=2):
    return {
        "parascale": {
            "training_backend": "native",
            "gradient_accumulation_steps": 2,
            "precision": "fp32",
        },
        "training": {
            "workload": "torch_tiny",
            "max_steps": steps,
            "seed": 42,
            "checkpoint_dir": str(path),
            "checkpoint_interval": 1,
            "scheduler": {
                "type": "cosine",
                "total_steps": 4,
                "warmup_steps": 1,
                "min_lr_ratio": 0.1,
            },
        },
        "optimizer": {"type": "adamw", "lr": 0.01},
    }


def load_payload(path, step):
    return torch.load(
        path / f"step-{step:08d}" / "backend_state.pt",
        map_location="cpu",
        weights_only=True,
    )


def test_tensor_learning_rates_are_json_scalars_in_report(tmp_path, monkeypatch):
    from parascale.optimizers.schedule import SchedulerSpec

    original = SchedulerSpec.build

    def tensor_schedule(spec, optimizer):
        for group in optimizer.param_groups:
            group["lr"] = torch.tensor(group["lr"])
        return original(spec, optimizer)

    monkeypatch.setattr(SchedulerSpec, "build", tensor_schedule)
    result = run_train_from_config(configuration(tmp_path))
    json.dumps(result, allow_nan=False)
    assert all(type(rate) is float for rate in result["scheduler"]["next_learning_rates"])
    payload = load_payload(tmp_path, 2)
    assert isinstance(payload["scheduler_state_dict"]["base_lrs"][0], torch.Tensor)


def test_product_cli_checkpoints_schedule_and_exact_resume(tmp_path):
    run_train_from_config(configuration(tmp_path / "full", 4))
    config = configuration(tmp_path / "split")
    first = run_train_from_config(config)
    resumed = run_train_from_config(config, resume_step=2)
    expected = load_payload(tmp_path / "full", 4)
    actual = load_payload(tmp_path / "split", 4)
    assert expected["scheduler_state_dict"] is not None
    assert actual["scheduler_state_dict"] == expected["scheduler_state_dict"]
    assert actual["scheduler_state_dict"]["last_epoch"] == 4
    torch.testing.assert_close(
        actual["backend_state"]["optimizer_state_dict"],
        expected["backend_state"]["optimizer_state_dict"],
        rtol=0,
        atol=0,
    )
    for name, value in expected["backend_state"]["model_state_dict"].items():
        torch.testing.assert_close(
            actual["backend_state"]["model_state_dict"][name], value, rtol=0, atol=0
        )
    assert actual["backend_state"]["optimizer_state_dict"]["param_groups"][0][
        "lr"
    ] == pytest.approx(0.001)
    assert first["scheduler"]["last_epoch"] == 2
    assert resumed["scheduler"]["last_epoch"] == 4
    assert resumed["last_metrics"]["next_learning_rates"] == pytest.approx([0.001])


@pytest.mark.parametrize(
    "change", [None, {"total_steps": 8}, {"type": "linear"}, {"warmup_steps": 0}]
)
def test_product_resume_rejects_schedule_changes_before_backend_load(
    tmp_path, monkeypatch, change
):
    from parascale.runtime.training.checkpointing import CheckpointController

    config = configuration(tmp_path)
    run_train_from_config(config)
    if change is None:
        config["training"].pop("scheduler")
    else:
        config["training"]["scheduler"].update(change)

    def forbidden(*args, **kwargs):
        raise AssertionError("backend state must not load before schedule preflight")

    monkeypatch.setattr(CheckpointController, "_load_portable_backend_state", forbidden)
    with pytest.raises(ValueError, match="scheduler"):
        run_train_from_config(config, resume_step=2)


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_schedule_curve_and_horizon_are_explicit(kind):
    from parascale.optimizers.schedule import SchedulerSpec

    spec = SchedulerSpec(type=kind, total_steps=6, warmup_steps=2, min_lr_ratio=0.2)
    expected = (
        [0, 0.5, 1, 0.8, 0.6, 0.4, 0.2, 0.2]
        if kind == "linear"
        else [0, 0.5, 1, 0.8828427125, 0.6, 0.3171572875, 0.2, 0.2]
    )
    assert [spec.factor(step) for step in range(8)] == pytest.approx(expected)
    assert SchedulerSpec.from_config({}) is None


@pytest.mark.parametrize(
    "invalid",
    [
        {"total_steps": True},
        {"total_steps": 0},
        {"warmup_steps": -1},
        {"warmup_steps": 4},
        {"min_lr_ratio": float("nan")},
        {"min_lr_ratio": float("inf")},
        {"min_lr_ratio": 1.1},
        {"type": "plateau"},
        {"typo": 5},
    ],
)
def test_invalid_schedule_rejected(invalid):
    from parascale.optimizers.schedule import SchedulerSpec

    with pytest.raises(ValueError, match="scheduler"):
        SchedulerSpec.from_config(
            {"training": {"scheduler": {"type": "cosine", "total_steps": 4, **invalid}}}
        )


def test_scheduler_builder_binds_final_optimizer_and_is_retained(tmp_path):
    model = torch.nn.Linear(2, 1)
    original = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = TrainEngine(ParaScaleConfig())
    built = []

    def build(optimizer):
        built.append(optimizer)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    def fit(**options):
        return engine.fit(
            [torch.ones(1, 2)],
            max_steps=1,
            model=model,
            optimizer=original,
            loss_fn=lambda output, batch: output.square().mean(),
            **options,
        )

    fit(
        optimizer_builder=lambda wrapped: torch.optim.AdamW(
            wrapped.parameters(), lr=0.01
        ),
        scheduler_builder=build,
    )
    assert built == [engine.training_backend.optimizer]
    assert built[0] is not original
    fit()
    assert engine.scheduler.last_epoch == 2
    assert built[0].param_groups[0]["lr"] == pytest.approx(0.0025)
    engine.save_checkpoint(CheckpointManager(str(tmp_path)))
    assert load_payload(tmp_path, 2)["scheduler_state_dict"]["last_epoch"] == 2


def test_wrong_optimizer_scheduler_rejected_before_training():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="scheduler.*optimizer"):
        TrainEngine(ParaScaleConfig()).fit(
            [torch.ones(1, 2)],
            model=model,
            optimizer=optimizer,
            optimizer_builder=lambda wrapped: torch.optim.SGD(
                wrapped.parameters(), lr=0.1
            ),
            scheduler=scheduler,
            loss_fn=lambda output, batch: output.square().mean(),
        )
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name].cpu(), value, rtol=0, atol=0)


def test_checkpoint_rejects_a_different_scheduler_before_writing(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = TrainEngine(ParaScaleConfig())
    engine.fit(
        [torch.ones(1, 2)],
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1),
        loss_fn=lambda output, batch: output.square().mean(),
    )
    replacement = torch.optim.lr_scheduler.StepLR(optimizer, 2)
    with pytest.raises(ValueError, match="scheduler"):
        engine.save_checkpoint(CheckpointManager(str(tmp_path)), scheduler=replacement)
    assert not (tmp_path / "step-00000001" / "manifest.json").exists()


def test_tensor_learning_rate_scheduler_is_supported(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=torch.tensor(0.01))
    engine = TrainEngine(ParaScaleConfig())
    engine.fit(
        [torch.ones(1, 2)],
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5),
        loss_fn=lambda output, batch: output.square().mean(),
    )
    assert optimizer.param_groups[0]["lr"].item() == pytest.approx(0.005)
    engine.save_checkpoint(CheckpointManager(str(tmp_path)))
    assert load_payload(tmp_path, 1)["scheduler_state_dict"]["last_epoch"] == 1


def test_function_scheduler_requires_explicit_semantic_identity():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="scheduler_metadata"):
        TrainEngine(ParaScaleConfig()).fit(
            [torch.ones(1, 2)],
            model=model,
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: 0.9**step
            ),
            loss_fn=lambda output, batch: output.square().mean(),
        )


@pytest.mark.parametrize("accumulation", [1, 2])
def test_skipped_optimizer_updates_do_not_advance_schedule(accumulation):
    from parascale.runtime.backends.base import TrainingBackend

    class SkipFirst(TrainingBackend):
        name = "native"

        def step(self, optimizer=None):
            if not getattr(self, "skipped", False):
                self.skipped = True
                self.optimizer.zero_grad(set_to_none=True)
                return False
            return super().step(optimizer)

    model = torch.nn.Linear(2, 1)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = ParaScaleConfig(gradient_accumulation_steps=accumulation)
    engine = TrainEngine(config, training_backend=SkipFirst(model, optimizer, config))
    engine.fit(
        [torch.ones(1, 2)] * (3 * accumulation),
        max_steps=2,
        model=model,
        optimizer=optimizer,
        scheduler_builder=lambda final: torch.optim.lr_scheduler.StepLR(
            final, 1, gamma=0.5
        ),
        loss_fn=lambda output, batch: output.square().mean(),
    )
    assert engine.state.global_step == engine.scheduler.last_epoch == 2
    assert engine.state.consumed_micro_batches == 3 * accumulation
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.025)


def _scheduler_failure_worker(rank, root, scenario):
    from parascale.optimizers.schedule import SchedulerSpec
    from parascale.runtime.backends.base import TrainingBackend

    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    root = Path(root)
    torch.distributed.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        if scenario == "configuration":
            request = configuration(root / "checkpoints")
            request["parascale"]["training_backend"] = "native_ddp"
            request["hardware_profile"] = {"world_size": 2, "gpus_per_node": 2}
            if rank == 1:
                request["training"]["scheduler"]["total_steps"] = 0
            with pytest.raises(
                (ValueError, RuntimeError), match="scheduler|another rank"
            ):
                run_train_from_config(request)
            return
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        config = ParaScaleConfig(training_backend="native")
        backend = TrainingBackend(model, optimizer, config)
        backend.name = "native"
        spec = SchedulerSpec(
            type="linear", total_steps=4 + (rank if scenario == "identity" else 0)
        )
        engine = TrainEngine(
            config, training_backend=backend, scheduler_metadata=spec.to_metadata()
        )
        if scenario == "function":
            engine.scheduler_metadata = {}
        before = deepcopy(model.state_dict())

        def build(final):
            if scenario == "function":
                return torch.optim.lr_scheduler.LambdaLR(
                    final, lambda step: (0.9 - 0.1 * rank) ** step
                )
            if scenario == "builder" and rank == 1:
                raise ValueError("scheduler builder intentionally failed")
            return spec.build(final)

        if scenario == "save_presence":
            engine.state.global_step = 1
            if rank == 0:
                engine.scheduler = spec.build(optimizer)
            with pytest.raises(
                (ValueError, RuntimeError), match="scheduler|another rank"
            ):
                engine.save_checkpoint(CheckpointManager(str(root / "checkpoints")))
            assert not (
                root / "checkpoints" / "step-00000001" / "manifest.json"
            ).exists()
        else:

            def forbidden_forward(*args):
                raise AssertionError("scheduler mismatch must fail before forward")

            model.register_forward_pre_hook(forbidden_forward)
            with pytest.raises(
                (ValueError, RuntimeError), match="scheduler|another rank"
            ):
                engine.fit(
                    [torch.ones(1, 2)],
                    model=model,
                    optimizer=optimizer,
                    scheduler_builder=build,
                    loss_fn=lambda output, batch: output.square().mean(),
                )
            assert engine.state.global_step == 0
        for name, value in before.items():
            assert torch.equal(model.state_dict()[name], value)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    "scenario", ["builder", "identity", "save_presence", "configuration", "function"]
)
def test_rank_local_scheduler_errors_fail_all_ranks_before_training_or_save(
    tmp_path, monkeypatch, scenario
):
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    context = torch.multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_scheduler_failure_worker, args=(rank, str(tmp_path), scenario)
        )
        for rank in range(2)
    ]
    deadline = time.monotonic() + 60
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
