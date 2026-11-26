# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Physical channel removal, masked-reference equivalence and HF roundtrip."""

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def model():
    config = transformers.LlamaConfig(vocab_size=64, hidden_size=32,
                                      intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = "eager"
    return transformers.LlamaForCausalLM(config).eval()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pruning_matches_masked_model_and_reloads(tmp_path, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from parascale.pruning import prune_llama_mlp

    original = model().to(device=device, dtype=dtype)
    original.config.mlp_bias = True
    for layer in original.model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(layer.mlp, name)
            linear.bias = torch.nn.Parameter(torch.linspace(-0.01, 0.01, linear.out_features, device=device, dtype=dtype))
    before = deepcopy(original.state_dict())
    pruned, report = prune_llama_mlp(original, keep_ratio=0.5, alignment=8)
    assert pruned.config.intermediate_size == 32
    assert original.config.intermediate_size == 64
    assert report["parameters_after"] < report["parameters_before"]
    assert report["parameters_before"] - report["parameters_after"] == 2 * (3 * 32 * 32 + 2 * 32)
    reference = deepcopy(original)
    for layer, record in zip(reference.model.layers, report["layers"]):
        removed = sorted(set(range(64)) - set(record["kept_indices"]))
        with torch.no_grad():
            layer.mlp.down_proj.weight[:, removed] = 0
    tokens = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        expected = reference(tokens).logits
        actual = pruned(tokens).logits
    tolerance = {torch.float32: (1e-5, 1e-6), torch.float16: (2e-3, 2e-4), torch.bfloat16: (2e-2, 2e-3)}[dtype]
    torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    pruned.save_pretrained(tmp_path)
    restored = transformers.LlamaForCausalLM.from_pretrained(tmp_path, attn_implementation="eager", torch_dtype=dtype).to(device).eval()
    assert all(p.dtype == dtype for p in restored.parameters())
    with torch.no_grad():
        torch.testing.assert_close(restored(tokens).logits, actual, rtol=0, atol=0)
        assert restored.generate(tokens, max_new_tokens=2, do_sample=False).shape[1] > tokens.shape[1]
    optimizer = torch.optim.SGD(restored.parameters(), lr=0.1)
    restored.train()
    weight = restored.model.layers[0].mlp.up_proj.weight.detach().clone()
    loss = restored(tokens, labels=tokens).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in restored.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(weight, restored.model.layers[0].mlp.up_proj.weight)


@pytest.mark.parametrize("mutation", ["adapter", "backward_pre_hook", "bias_config"])
def test_unsupported_mutations_rejected(mutation):
    from parascale.pruning import prune_llama_mlp

    source = model()
    if mutation == "adapter":
        projection = source.model.layers[0].self_attn.q_proj
        source.model.layers[0].self_attn.q_proj = torch.nn.Sequential(projection)
        source.eval()
    elif mutation == "backward_pre_hook":
        source.model.layers[0].mlp.up_proj.register_full_backward_pre_hook(lambda *args: None)
    else:
        source.model.layers[0].mlp.up_proj.bias = torch.nn.Parameter(torch.ones(64))
    with pytest.raises(ValueError, match="architecture|hooks"):
        prune_llama_mlp(source, alignment=8)


@pytest.mark.parametrize("ratio", [0, 1, -1, float("nan"), True])
def test_invalid_ratio_rejected(ratio):
    from parascale.pruning import prune_llama_mlp

    with pytest.raises(ValueError, match="keep_ratio"):
        prune_llama_mlp(model(), keep_ratio=ratio, alignment=8)


def test_invalid_alignment_or_too_small_result_rejected():
    from parascale.pruning import prune_llama_mlp

    for alignment in (0, True, 1.5, 128):
        with pytest.raises(ValueError, match="alignment|channels"):
            prune_llama_mlp(model(), keep_ratio=0.5, alignment=alignment)


def test_hooks_nonfinite_weights_and_training_mode_rejected():
    from parascale.pruning import prune_llama_mlp

    source = model()
    source.train()
    with pytest.raises(ValueError, match="eval"):
        prune_llama_mlp(source, alignment=8)
    source.eval()
    handle = source.model.layers[0].mlp.up_proj.register_forward_hook(lambda *args: None)
    with pytest.raises(ValueError, match="hooks"):
        prune_llama_mlp(source, alignment=8)
    handle.remove()
    with torch.no_grad():
        source.model.layers[0].mlp.down_proj.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        prune_llama_mlp(source, alignment=8)


def test_deterministic_ties_and_shared_parameters():
    from parascale.pruning import prune_llama_mlp

    source = model()
    with torch.no_grad():
        for layer in source.model.layers:
            layer.mlp.down_proj.weight.fill_(0.01)
    _, report = prune_llama_mlp(source, keep_ratio=0.5, alignment=8)
    assert all(item["kept_indices"] == list(range(32)) for item in report["layers"])
    source.model.layers[0].mlp.up_proj.weight = source.model.layers[0].mlp.gate_proj.weight
    with pytest.raises(ValueError, match="shared"):
        prune_llama_mlp(source, alignment=8)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_forward_and_bias_shapes(dtype):
    from parascale.pruning import prune_llama_mlp

    source = model().to(dtype=dtype)
    for layer in source.model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(layer.mlp, name)
            linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=dtype))
    source.config.mlp_bias = True
    pruned, _ = prune_llama_mlp(source, keep_ratio=0.5, alignment=8)
    assert pruned.model.layers[0].mlp.up_proj.bias.shape == (32,)
    assert pruned.model.layers[0].mlp.down_proj.bias.shape == (32,)
    with torch.no_grad():
        assert torch.isfinite(pruned(torch.tensor([[1, 2, 3]])).logits).all()
