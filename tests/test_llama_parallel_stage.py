# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Complete Llama stage parity; pinned operators are exercised on the GPU host."""

import copy

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(
    transformers.__version__ != "4.51.3", reason="pinned Llama operator contract"
)


def model_fixture(*, tied=False, attention="eager"):
    config = transformers.LlamaConfig(
        vocab_size=31,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=tied,
        attention_dropout=0.0,
    )
    config._attn_implementation = attention
    return transformers.LlamaForCausalLM(config).train()


@pytest.mark.parametrize("stages,tied", [(1, False), (1, True), (2, False)])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_complete_model_stage_logits_loss_and_gradients(stages, tied, attention):
    from parascale.parallel import CausalLMLoss
    from parascale.parallel.llama import LlamaStage

    torch.manual_seed(17)
    reference = model_fixture(tied=tied, attention=attention)
    adapted_model = copy.deepcopy(reference)
    adapters = [LlamaStage(adapted_model, index, stages) for index in range(stages)]
    tokens = torch.randint(0, 31, (2, 7))
    targets = tokens.clone()
    targets[0, 2:5] = -100
    expected = reference(tokens, use_cache=False).logits
    actual = tokens
    for adapter in adapters:
        actual = adapter(actual)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    objective = CausalLMLoss(31)
    expected_loss = objective(expected, targets)
    actual_loss = objective(actual, targets)
    native_loss = reference(tokens, labels=targets, use_cache=False).loss
    torch.testing.assert_close(
        actual_loss / objective.normalization_count(targets), native_loss
    )
    expected_loss.backward()
    actual_loss.backward()
    expected_parameters = dict(reference.named_parameters())
    seen = set()
    for adapter in adapters:
        for name, parameter in adapter.named_parameters():
            assert name not in seen
            seen.add(name)
            torch.testing.assert_close(
                parameter.grad, expected_parameters[name].grad, rtol=2e-5, atol=1e-6
            )
    assert seen == set(expected_parameters)
    if tied:
        assert adapters[0].model.embed_tokens.weight is adapters[0].lm_head.weight


def test_adapter_does_not_keep_unowned_parameters():
    from parascale.parallel.llama import LlamaStage

    source = model_fixture()
    first, last = LlamaStage(source, 0, 2), LlamaStage(source, 1, 2)
    assert list(first.model.layers) == ["0", "1"]
    assert list(last.model.layers) == ["2", "3"]
    assert not hasattr(first, "lm_head")
    assert not hasattr(last.model, "embed_tokens")
    assert first.model.layers["0"] is source.model.layers[0]
    assert last.lm_head is source.lm_head


@pytest.mark.parametrize("index,count", [(-1, 2), (2, 2), (True, 2), (0, 0), (0, 3)])
def test_invalid_stage_rejected(index, count):
    from parascale.parallel.llama import LlamaStage

    with pytest.raises(ValueError, match="stage"):
        LlamaStage(model_fixture(), index, count)


def test_cross_stage_ties_rejected_without_mutation():
    from parascale.parallel.llama import LlamaStage

    model = model_fixture(tied=True)
    with pytest.raises(ValueError, match="tied"):
        LlamaStage(model, 0, 2)
    assert model.model.embed_tokens.weight is model.lm_head.weight


@pytest.mark.parametrize(
    "setting,value", [("attention_dropout", 0.1), ("pretraining_tp", 2)]
)
def test_unsupported_model_settings_rejected(setting, value):
    from parascale.parallel.llama import LlamaStage

    model = model_fixture()
    setattr(model.config, setting, value)
    with pytest.raises(ValueError, match=setting):
        LlamaStage(model)


@pytest.mark.parametrize(
    "setting,value",
    [("num_attention_heads", 3), ("num_key_value_heads", 1), ("intermediate_size", 63)],
)
def test_nondivisible_tp_configuration_rejected(setting, value):
    from parascale.parallel.llama import LlamaStage

    model = model_fixture()
    setattr(model.config, setting, value)
    with pytest.raises(ValueError, match="divisible"):
        LlamaStage(model, tp_size=2)


def test_tensor_plan_covers_only_owned_attention_and_mlp():
    from parascale.parallel.llama import LlamaStage

    stage = LlamaStage(model_fixture(), 1, 2, tp_size=2)
    plan = stage.tensor_parallel_plan()
    expected = {
        f"model.layers.{index}.{projection}"
        for index in (2, 3)
        for projection in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    }
    assert set(plan) == expected
    assert set(plan) <= set(dict(stage.named_modules()))
    assert LlamaStage(model_fixture()).tensor_parallel_plan() == {}


def test_wrong_input_contract_and_cache_are_not_silently_accepted():
    from parascale.parallel.llama import LlamaStage

    source = model_fixture()
    first, last = LlamaStage(source, 0, 2), LlamaStage(source, 1, 2)
    with pytest.raises(ValueError, match="int64"):
        first(torch.ones(2, 7))
    with pytest.raises(ValueError, match="hidden"):
        last(torch.ones(2, 7, dtype=torch.int64))
    with pytest.raises(TypeError):
        first(torch.ones(2, 7, dtype=torch.int64), use_cache=True)


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_causal_prefix_is_independent_of_future_tokens(attention):
    from parascale.parallel.llama import LlamaStage

    stage = LlamaStage(model_fixture(attention=attention))
    tokens = torch.tensor([[1, 2, 2, 3, 4]])
    changed = torch.tensor([[1, 2, 2, 9, 9]])
    torch.testing.assert_close(stage(tokens)[:, :3], stage(changed)[:, :3])


def test_sdpa_uses_native_causal_dispatch_without_materializing_mask():
    from parascale.parallel.llama import LlamaStage

    stage = LlamaStage(model_fixture(attention="sdpa"))
    observed = []
    hook = stage.model.layers["0"].register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(kwargs["attention_mask"]),
        with_kwargs=True,
    )
    try:
        stage(torch.ones(1, 5, dtype=torch.int64))
    finally:
        hook.remove()
    assert observed == [None]


def test_invalid_embedding_input_is_rejected_by_trainer_preflight():
    from parascale.parallel import CausalLMLoss, DistributedParallelConfig
    from parascale.parallel.execution import DistributedTrainer
    from parascale.parallel.llama import LlamaStage

    trainer = object.__new__(DistributedTrainer)
    trainer.config = DistributedParallelConfig(microbatches=1)
    trainer.device = torch.device("cpu")
    trainer.input_shape = (1, 3)
    trainer.input_dtype = torch.int64
    trainer.pp_rank = 0
    trainer.module = LlamaStage(model_fixture())
    trainer.summed_loss = CausalLMLoss(31)
    with pytest.raises(ValueError, match="input token"):
        trainer._prepare_step(torch.tensor([[31, 1, 2]]), torch.tensor([[-100, 1, 2]]))
