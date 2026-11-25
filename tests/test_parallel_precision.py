# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Explicit autocast scope and FP32 optimizer-state contracts for stage training."""

import copy
import json
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def ddp_constructor(monkeypatch):
    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
        bf16_compress_hook,
    )

    from parascale.parallel import execution
    from parascale.parallel.distributed import DistributedParallelConfig

    dist = torch.distributed
    group = object()
    mesh = Mock(return_value={"dp": SimpleNamespace(get_group=lambda: group)})
    gather = Mock(
        side_effect=lambda records, value: records.__setitem__(
            slice(None), [value, value]
        )
    )
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(dist, "get_global_rank", lambda *args: 0)
    monkeypatch.setattr(dist, "broadcast", Mock())
    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setattr(dist.device_mesh, "init_device_mesh", mesh)
    monkeypatch.setattr(execution, "_validate_constructor", Mock())
    monkeypatch.setattr(execution, "validate_precision", Mock())
    monkeypatch.setattr(torch.cuda.nccl, "version", lambda: (2, 10, 0))
    register = Mock()

    class DDP:
        def __init__(self, module, *, device_ids, process_group):
            self.module = module
            self.process_group = process_group
            self._comm_hooks = []

        def register_comm_hook(self, state, hook):
            register(state, hook)
            self._comm_hooks.append((hook, state))

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", DDP)

    def construct(*, config=None, **kwargs):
        return execution.DistributedTrainer(
            torch.nn.Linear(2, 2),
            config or DistributedParallelConfig(dp_size=2),
            SimpleNamespace(
                device=torch.device("cuda", 0), shape=(1, 2), dtype=torch.float32
            ),
            torch.nn.MSELoss(),
            **kwargs,
        )

    return SimpleNamespace(
        construct=construct,
        group=group,
        hook=bf16_compress_hook,
        mesh=mesh,
        gather=gather,
        register=register,
        execution=execution,
    )


@pytest.mark.parametrize("policy", ["auto", "fp16_compress", True, None])
def test_ddp_policy_rejects_non_effective_names_before_mesh(ddp_constructor, policy):
    with pytest.raises(ValueError, match="ddp_comm_hook"):
        ddp_constructor.construct(ddp_comm_hook=policy)
    ddp_constructor.mesh.assert_not_called()


@pytest.mark.parametrize("axes", [(1, 1, 1), (2, 2, 1), (2, 1, 2)])
def test_ddp_compression_requires_pure_dp_before_mesh(ddp_constructor, axes):
    from parascale.parallel.distributed import DistributedParallelConfig

    dp, tp, pp = axes
    config = DistributedParallelConfig(dp_size=dp, tp_size=tp, pp_size=pp)
    with pytest.raises(ValueError, match="pure DP"):
        ddp_constructor.construct(config=config, ddp_comm_hook="bf16_compress")
    ddp_constructor.mesh.assert_not_called()


@pytest.mark.parametrize("failure", ["bf16", "nccl"])
def test_ddp_compression_validates_capability_before_mesh(
    ddp_constructor, monkeypatch, failure
):
    if failure == "bf16":
        ddp_constructor.execution.validate_precision.side_effect = ValueError(
            "no native BF16"
        )
    else:
        monkeypatch.setattr(torch.cuda.nccl, "version", lambda: (2, 9, 6))
    with pytest.raises(ValueError, match="BF16|NCCL"):
        ddp_constructor.construct(ddp_comm_hook="bf16_compress")
    ddp_constructor.mesh.assert_not_called()


def test_ddp_hook_preflight_rejects_peer_policy_mismatch(ddp_constructor):
    def disagree(records, value):
        error, signature = value
        assert error is None and signature[-1] == "bf16_compress"
        records[:] = [value, (None, (*signature[:-1], "none"))]

    ddp_constructor.gather.side_effect = disagree
    with pytest.raises(ValueError, match="preflight.*ddp_comm_hook.*agree"):
        ddp_constructor.construct(ddp_comm_hook="bf16_compress")
    ddp_constructor.mesh.assert_not_called()


@pytest.mark.parametrize("peer", [False, True])
def test_ddp_hook_registration_failure_is_coordinated(ddp_constructor, peer):
    def gather(records, value):
        records[:] = (
            [value, "injected peer failure" if peer else None]
            if isinstance(value, (str, type(None)))
            else [value, value]
        )

    ddp_constructor.gather.side_effect = gather
    if not peer:
        ddp_constructor.register.side_effect = RuntimeError(
            "injected registration failure"
        )
    expected = (
        "DDP hook registration phase failed on another rank:.*injected peer failure"
        if peer
        else "injected registration failure"
    )
    with pytest.raises(RuntimeError, match=expected):
        ddp_constructor.construct(ddp_comm_hook="bf16_compress")
    assert ddp_constructor.gather.call_count == 2
    assert ddp_constructor.gather.call_args.args[1] == (
        None if peer else "injected registration failure"
    )


@pytest.mark.parametrize(
    "drift",
    [
        "module",
        "ddp_module",
        "wrapper",
        "group",
        "hook",
        "state",
        "context",
        "none_hook",
        "clean",
    ],
)
def test_ddp_execution_policy_checks_registered_binding(ddp_constructor, drift):
    policy = "none" if drift == "none_hook" else "bf16_compress"
    context = dict(
        model="linear",
        loss="mse",
        tokenizer="test",
        data="test",
        precision="fp32",
        execution={"torch": "2.9.1"},
    )
    trainer = ddp_constructor.construct(
        ddp_comm_hook=policy, checkpoint_context=context
    )
    assert trainer.checkpoint_context["execution"] == {
        "torch": "2.9.1",
        "ddp_comm_hook": policy,
    }
    assert context["execution"] == {"torch": "2.9.1"}
    assert trainer.ddp_comm_hook == policy
    if policy == "none":
        ddp_constructor.register.assert_not_called()
    else:
        ddp_constructor.register.assert_called_once_with(
            ddp_constructor.group, ddp_constructor.hook
        )
    trainer.validate_execution_policy()
    wrapper = trainer.execution_module
    if drift == "clean":
        assert (
            ddp_constructor.construct(ddp_comm_hook=policy).checkpoint_context is None
        )
        return
    if drift == "module":
        trainer.module = copy.deepcopy(trainer.module)
    elif drift == "ddp_module":
        wrapper.module = copy.deepcopy(wrapper.module)
    elif drift == "wrapper":
        trainer.execution_module = object()
    elif drift == "group":
        wrapper.process_group = object()
    elif drift == "context":
        trainer._checkpoint_context = json.dumps(context)
    elif drift == "hook":
        wrapper._comm_hooks[:] = [(object(), ddp_constructor.group)]
    elif drift == "state":
        wrapper._comm_hooks[:] = [(ddp_constructor.hook, object())]
    else:
        wrapper._comm_hooks.append((ddp_constructor.hook, ddp_constructor.group))
    with pytest.raises(ValueError, match="policy|module|wrapper|group|hook"):
        trainer.validate_execution_policy()


@pytest.mark.parametrize("registry", [None, (), [None], [(object(),)]])
def test_ddp_execution_policy_rejects_unreadable_hook_registry(
    ddp_constructor, registry
):
    trainer = ddp_constructor.construct()
    if registry is None:
        del trainer.execution_module._comm_hooks
    else:
        trainer.execution_module._comm_hooks = registry
    with pytest.raises(ValueError, match="hook registry"):
        trainer.validate_execution_policy()


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
@pytest.mark.parametrize("return_outputs", [False, True])
def test_unused_stage_outputs_are_released_before_backward(
    monkeypatch, precision, return_outputs
):
    trainer = trainer_fixture(precision)
    reference = copy.deepcopy(trainer.module)
    inputs, targets = torch.randn(4, 4), torch.randn(4, 4)

    def loss(output, target):
        return (output.double() - target.double()).square().mean()

    trainer.loss_fn = loss
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
        expected_outputs = [reference(batch) for batch in inputs.chunk(2)]
        expected_losses = [
            loss(output, target) / 2
            for output, target in zip(expected_outputs, targets.chunk(2))
        ]
    for value in expected_losses:
        value.backward()
    references = []
    trainer.module.register_forward_hook(
        lambda module, args, output: references.append(weakref.ref(output))
    )
    backward = torch.Tensor.backward
    calls = []

    def checked_backward(value, *args, **kwargs):
        if not return_outputs:
            assert all(reference() is None for reference in references)
        calls.append(True)
        return backward(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", checked_backward)
    result = trainer.forward_backward(inputs, targets, return_outputs=return_outputs)
    assert len(calls) == 2
    assert (result.output is not None) == return_outputs
    if return_outputs:
        torch.testing.assert_close(
            result.output, torch.cat(expected_outputs), rtol=0, atol=0
        )
        assert not result.output.requires_grad
    for parameter, expected in zip(trainer.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=0, atol=0)


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


@pytest.mark.parametrize("rank", [0, 1])
def test_stage_example_validation_is_coordinated_before_mesh(monkeypatch, rank):
    from unittest.mock import Mock

    from parascale.parallel import DistributedParallelConfig, execution

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(execution, "_validate_constructor", lambda *args: None)
    mesh = Mock(side_effect=AssertionError("mesh created before input validation"))
    monkeypatch.setattr(torch.distributed.device_mesh, "init_device_mesh", mesh)
    module = torch.nn.Linear(2, 2)
    example = torch.zeros(1, 2)
    validator = Mock(side_effect=ValueError("invalid stage example") if rank else None)
    module.validate_example_inputs = validator
    module.forward = Mock(side_effect=AssertionError("forward during preflight"))

    def gather(records, record):
        if rank:
            assert record[0] == "invalid stage example"
            records[:] = [(None, None), record]
        else:
            assert record[0] is None
            records[:] = [record, ("invalid stage example", None)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(ValueError, match="rank 1: invalid stage example"):
        execution.DistributedTrainer(
            module,
            DistributedParallelConfig(pp_size=2),
            example,
            torch.nn.MSELoss(),
        )
    validator.assert_called_once_with(example)
    module.forward.assert_not_called()
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
