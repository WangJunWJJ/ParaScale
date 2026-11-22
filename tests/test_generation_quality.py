# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午8:48
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Quality checks must exercise the same autoregressive generation path."""

import copy
import json

import pytest

torch = pytest.importorskip("torch")

from tests.benchmarks.scripts.quantized_generation_quality import (  # noqa: E402
    TeacherForcedScores,
    evaluate_generation,
)


def test_teacher_forced_scores_record_original_distribution_without_mutation():
    targets = torch.tensor([[1, 0], [0, 2]])
    processor = TeacherForcedScores(targets, prompt_length=2)
    scores = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], requires_grad=True)
    original = scores.clone()
    prompt = torch.tensor([[2, 2], [1, 1]])
    sequence = prompt
    expected = []
    for step in range(2):
        forced = processor(sequence, scores)
        assert torch.equal(scores, original)
        assert torch.equal(forced.argmax(-1), targets[:, step])
        expected.append(
            torch.logsumexp(scores, -1)
            - scores.gather(1, targets[:, step : step + 1]).squeeze(1)
        )
        sequence = torch.cat((sequence, targets[:, step : step + 1]), dim=1)
    summary = processor.finish(sequence, prompt)
    assert summary["first_token"]["nll_sum"] == pytest.approx(expected[0].sum().item())
    assert summary["decode"]["nll_sum"] == pytest.approx(expected[1].sum().item())
    assert summary["decode"]["tokens"] == 2
    assert processor.nll.shape == targets.shape
    assert not processor.nll.requires_grad
    assert processor.nll.grad_fn is None


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_forcing_does_not_hide_nonfinite_logits(invalid):
    targets = torch.tensor([[0, 1]])
    processor = TeacherForcedScores(targets, prompt_length=1)
    prompt = torch.tensor([[1]])
    processor(prompt, torch.tensor([[invalid, 1.0]]))
    processor(torch.tensor([[1, 0]]), torch.ones(1, 2))
    with pytest.raises(ValueError, match="nonfinite"):
        processor.finish(torch.tensor([[1, 0, 1]]), prompt)


def test_teacher_forcing_rejects_wrong_position_and_incomplete_sequence():
    processor = TeacherForcedScores(torch.tensor([[0, 1]]), prompt_length=2)
    with pytest.raises(ValueError, match="position"):
        processor(torch.ones(1, 3, dtype=torch.long), torch.ones(1, 2))
    with pytest.raises(ValueError, match="incomplete"):
        processor.finish(
            torch.ones(1, 4, dtype=torch.long), torch.ones(1, 2, dtype=torch.long)
        )


def test_decode_observer_starts_only_after_first_token_scoring():
    calls = []
    processor = TeacherForcedScores(
        torch.tensor([[0, 1]]),
        prompt_length=1,
        on_decode_start=lambda: calls.append(processor.step),
    )
    processor(torch.tensor([[1]]), torch.ones(1, 2))
    processor(torch.tensor([[1, 0]]), torch.ones(1, 2))
    assert calls == [1]


@pytest.mark.parametrize("static_cache", [False, True])
def test_real_llama_generation_nll_matches_full_forward_and_resets_cache(static_cache):
    transformers = pytest.importorskip("transformers")
    from tests.benchmarks.scripts.benchmark_quantized_inference import (
        fixed_length_generation,
    )

    torch.manual_seed(12)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    model.generation_config.forced_bos_token_id = 7
    model.generation_config.forced_eos_token_id = 2
    model.generation_config.suppress_tokens = [0, 2]
    model.generation_config.repetition_penalty = 2.0
    model.generation_config.bad_words_ids = [[1]]
    model.generation_config.renormalize_logits = True
    defaults = copy.deepcopy(model.generation_config.to_dict())
    observed_configs = []
    tokens = torch.tensor([1, 3, 2, 0, 4, 5, 6, 2, 0, 7, 8, 9, 10, 11, 12, 13, 14])

    def generate(prompt, processor):
        return fixed_length_generation(
            model,
            prompt,
            torch.ones_like(prompt),
            2,
            0,
            static_cache=static_cache,
            logits_processor=[processor],
            configuration_observer=observed_configs.append,
        )

    with torch.inference_mode():
        first = evaluate_generation(
            generate, tokens, prompt_length=2, new_tokens=2, batch_size=2, device="cpu"
        )
        again = evaluate_generation(
            generate, tokens, prompt_length=2, new_tokens=2, batch_size=2, device="cpu"
        )
        expected = []
        for block in tokens[:16].reshape(2, 2, 4):
            logits = model(
                block, attention_mask=torch.ones_like(block), use_cache=False
            ).logits.float()
            expected.append(
                torch.nn.functional.cross_entropy(
                    logits[:, 1:3].reshape(-1, 16),
                    block[:, 2:].reshape(-1),
                    reduction="none",
                ).reshape(2, 2)
            )
    expected = torch.cat(expected)
    assert first == again
    assert first["first_token"]["nll_sum"] == pytest.approx(
        expected[:, 0].sum().item(), abs=1e-5
    )
    assert first["decode"]["nll_sum"] == pytest.approx(
        expected[:, 1].sum().item(), abs=1e-5
    )
    assert first["decode"]["tokens"] == 4
    assert first["discarded_tokens"] == 1
    assert first["segments"] == 2
    assert first["all"]["tokens"] == 8
    assert model.generation_config.to_dict() == defaults
    assert json.loads(json.dumps(observed_configs, allow_nan=False)) == observed_configs
    assert observed_configs and all(
        config["suppress_tokens"] is None
        and config["forced_bos_token_id"] is None
        and config["eos_token_id"] is None
        and config["repetition_penalty"] == 1.0
        and config["renormalize_logits"] is False
        for config in observed_configs
    )
