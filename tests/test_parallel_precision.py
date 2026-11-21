# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Explicit autocast scope and FP32 optimizer-state contracts for stage training."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def trainer_fixture(precision):
    from parascale.parallel import DistributedParallelConfig
    from parascale.parallel.execution import DistributedTrainer

    trainer = object.__new__(DistributedTrainer)
    trainer._precision = precision
    trainer.device = torch.device("cpu")
    trainer._config = DistributedParallelConfig(microbatches=2)
    trainer.module = torch.nn.Linear(4, 4)
    trainer.execution_module = trainer.module
    trainer.loss_fn = torch.nn.MSELoss()
    trainer.summed_loss = None
    trainer.schedule = None
    trainer.pp_rank = 0
    trainer._normalization_count = None
    trainer._prepare_step = lambda inputs, targets: None
    return trainer


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("ambient", [False, True])
def test_explicit_precision_controls_forward_not_backward(precision, ambient):
    trainer = trainer_fixture(precision)
    reference = copy.deepcopy(trainer.module)
    inputs, targets = torch.randn(4, 4), torch.randn(4, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
        losses = [
            torch.nn.functional.mse_loss(reference(x), y) / 2
            for x, y in zip(inputs.chunk(2), targets.chunk(2))
        ]
    for loss in losses:
        loss.backward()
    expected = [parameter.grad.clone() for parameter in reference.parameters()]
    forward, backward = [], []
    trainer.module.register_forward_hook(
        lambda module, args, output: forward.append(output.dtype)
    )
    for parameter in trainer.module.parameters():
        parameter.register_hook(
            lambda grad: backward.append(torch.is_autocast_enabled("cpu"))
        )
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=ambient):
        result = trainer.forward_backward(inputs, targets, return_outputs=True)
        assert torch.is_autocast_enabled("cpu") == ambient
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    assert forward == [dtype, dtype]
    assert result.output.dtype == dtype
    assert backward and not any(backward)
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=0.01, foreach=False)
    for parameter, gradient in zip(trainer.parameters(), expected):
        assert parameter.dtype == parameter.grad.dtype == torch.float32
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    optimizer.step()
    for state in optimizer.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32


def test_exception_restores_ambient_autocast():
    trainer = trainer_fixture("fp32")

    def fail(module, args):
        assert not torch.is_autocast_enabled("cpu")
        raise RuntimeError("injected forward failure")

    trainer.module.register_forward_pre_hook(fail)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="injected"):
            trainer.forward_backward(torch.randn(4, 4), torch.randn(4, 4))
        assert torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize("precision", ["fp16", "auto", True, None])
def test_unsupported_precision_rejected(precision):
    from parascale.parallel.precision import validate_precision

    with pytest.raises(ValueError, match="precision"):
        validate_precision(precision, torch.device("cpu"))


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_pipeline_inference_and_forward_share_precision(monkeypatch, precision):
    from torch.distributed.pipelining import PipelineStage

    from parascale.parallel.pipeline_stage import PrecisionPipelineStage

    def initialize(self, module, index, count, device, example, **kwargs):
        self.submod, self.device = module, device
        with torch.no_grad():
            self.inferred_dtype = module(example).dtype

    monkeypatch.setattr(PipelineStage, "__init__", initialize)
    model = torch.nn.Linear(4, 4)
    inputs = torch.randn(2, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "fp32"):
        stage = PrecisionPipelineStage(
            model, 0, 2, inputs.device, inputs, group=None, precision=precision
        )
        output = stage.forward_maybe_with_nosync(inputs)
    expected = torch.bfloat16 if precision == "bf16" else torch.float32
    assert stage.inferred_dtype == output.dtype == expected
    assert output.requires_grad


def test_pipeline_schedule_disables_ambient_backward_autocast(monkeypatch):
    trainer = trainer_fixture("bf16")
    from parascale.parallel import DistributedParallelConfig

    trainer._config = DistributedParallelConfig(pp_size=2, microbatches=2)
    trainer.dp_rank = trainer.tp_rank = 0
    trainer.mesh = {"pp": SimpleNamespace(get_group=lambda: None)}
    seen = []
    trainer.schedule = SimpleNamespace(
        step=lambda inputs, **kwargs: seen.append(torch.is_autocast_enabled("cpu"))
    )
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *args, **kwargs: None)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        trainer.forward_backward(torch.randn(4, 4), torch.randn(4, 4))
        assert torch.is_autocast_enabled("cpu")
    assert seen == [False]


def test_bf16_does_not_reuse_weight_casts_across_optimizer_steps():
    trainer = trainer_fixture("bf16")
    reference = copy.deepcopy(trainer.module)
    inputs, targets = torch.randn(4, 4), torch.randn(4, 4)
    optimizer = torch.optim.SGD(trainer.parameters(), lr=0.3)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        for _ in range(3):
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
                losses = [
                    torch.nn.functional.mse_loss(reference(x), y) / 2
                    for x, y in zip(inputs.chunk(2), targets.chunk(2))
                ]
            with torch.autocast("cpu", enabled=False):
                for loss in losses:
                    loss.backward()
            trainer.forward_backward(inputs, targets)
            optimizer.step()
            reference_optimizer.step()
            for parameter, expected in zip(
                trainer.parameters(), reference.parameters()
            ):
                torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


def test_precision_cannot_be_reassigned():
    trainer = trainer_fixture("fp32")
    with pytest.raises(AttributeError):
        trainer.precision = "bf16"


def test_bf16_requires_cuda():
    from parascale.parallel.precision import validate_precision

    with pytest.raises(ValueError, match="native CUDA BF16"):
        validate_precision("bf16", torch.device("cpu"))


@pytest.mark.parametrize("supported", [True, False])
def test_bf16_checks_selected_device_without_emulation(monkeypatch, supported):
    from parascale.parallel.precision import validate_precision

    selected, probes = [], []
    monkeypatch.setattr(
        torch.cuda, "device", lambda device: selected.append(device) or nullcontext()
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda **kwargs: probes.append(kwargs) or supported,
    )
    device = torch.device("cuda", 2)
    if supported:
        validate_precision("bf16", device)
    else:
        with pytest.raises(ValueError, match="native CUDA BF16"):
            validate_precision("bf16", device)
    assert selected == [device]
    assert probes == [{"including_emulation": False}]


@pytest.mark.parametrize(
    "stage,precision,accepted",
    [(0, "bf16", False), (1, "fp32", False), (1, "bf16", True)],
)
def test_bf16_stage_inputs_only_allowed_after_first_stage(stage, precision, accepted):
    from parascale.parallel.execution import _validate_example_input

    inputs = torch.zeros(2, 4, dtype=torch.bfloat16)
    if accepted:
        _validate_example_input(inputs, stage, precision)
    else:
        with pytest.raises(ValueError, match="input"):
            _validate_example_input(inputs, stage, precision)


@pytest.mark.parametrize("mismatch", ["context", "rank"])
def test_precision_agreement_precedes_mesh_creation(monkeypatch, mismatch):
    from unittest.mock import Mock

    from parascale.parallel import DistributedParallelConfig, execution

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(execution, "_validate_constructor", lambda *args: None)
    mesh = Mock()
    monkeypatch.setattr(torch.distributed.device_mesh, "init_device_mesh", mesh)

    def gather(records, record):
        records[:] = [record, record]
        if mismatch == "rank":
            records[1] = (None, (*record[1][:-1], "bf16"))

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    context = None
    if mismatch == "context":
        context = dict(model="m", loss="l", data="d", tokenizer="t", precision="bf16")
    with pytest.raises(ValueError, match="precision"):
        execution.DistributedTrainer(
            torch.nn.Linear(2, 2),
            DistributedParallelConfig(dp_size=2),
            torch.zeros(1, 2),
            torch.nn.MSELoss(),
            checkpoint_context=context,
        )
    mesh.assert_not_called()


@pytest.mark.parametrize("dp,tp", [(1, 1), (2, 1), (1, 2)])
@pytest.mark.parametrize("mismatch", [None, "dtype", "shape", "arity"])
def test_pipeline_boundaries_are_checked_on_every_rank(monkeypatch, dp, tp, mismatch):
    from parascale.parallel import DistributedParallelConfig
    from parascale.parallel.pipeline_stage import validate_pipeline_contract

    config = DistributedParallelConfig(dp_size=dp, tp_size=tp, pp_size=2)
    fp32 = ((2, 4), "torch.float32")
    bf16 = ((2, 4), "torch.bfloat16")
    records = []
    for rank in range(config.world_size):
        _, pp, _ = config.coordinates(rank)
        records.append(([fp32 if pp == 0 else bf16], [bf16]))
    bad_rank = next(
        rank
        for rank in range(config.world_size)
        if config.coordinates(rank) == (0, 1, 0)
    )
    if mismatch == "dtype":
        records[bad_rank] = ([fp32], [bf16])
    elif mismatch == "shape":
        records[bad_rank] = ([((2, 5), "torch.bfloat16")], [bf16])
    elif mismatch == "arity":
        records[bad_rank] = ([bf16, bf16], [bf16])
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda outputs, local: outputs.__setitem__(slice(None), records),
    )
    for rank in range(config.world_size):
        inputs, outputs = records[rank]
        stage = SimpleNamespace(
            inputs_meta=tuple(
                torch.empty(shape, dtype=getattr(torch, dtype.split(".")[-1]))
                for shape, dtype in inputs
            ),
            get_outputs_meta=lambda: tuple(
                torch.empty(shape, dtype=getattr(torch, dtype.split(".")[-1]))
                for shape, dtype in outputs
            ),
        )
        if mismatch is None:
            validate_pipeline_contract(stage, config)
        else:
            with pytest.raises(ValueError, match="pipeline boundary"):
                validate_pipeline_contract(stage, config)


def test_pipeline_schedule_agreement_precedes_mesh_creation(monkeypatch):
    from unittest.mock import Mock

    from parascale.parallel import DistributedParallelConfig, execution

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(execution, "_validate_constructor", lambda *args: None)
    mesh = Mock(side_effect=AssertionError("mesh created before schedule agreement"))
    monkeypatch.setattr(torch.distributed.device_mesh, "init_device_mesh", mesh)

    def gather(records, record):
        other = tuple("1f1b" if value == "gpipe" else value for value in record[1])
        records[:] = [record, (None, other)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(ValueError, match="schedule"):
        execution.DistributedTrainer(
            torch.nn.Linear(2, 2),
            DistributedParallelConfig(pp_size=2),
            torch.zeros(1, 2),
            torch.nn.MSELoss(),
        )
    mesh.assert_not_called()


def test_execution_config_cannot_be_replaced_after_schedule_selection():
    from parascale.parallel import DistributedParallelConfig
    from parascale.parallel.execution import DistributedTrainer

    trainer = object.__new__(DistributedTrainer)
    trainer._config = DistributedParallelConfig(pp_size=2)
    with pytest.raises(AttributeError):
        trainer.config = DistributedParallelConfig(pp_size=2, pipeline_schedule="1f1b")
    assert trainer.config.pipeline_schedule == "gpipe"
