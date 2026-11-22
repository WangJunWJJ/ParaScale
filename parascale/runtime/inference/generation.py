# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午10:25
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Explicit HF greedy generation shared by product and measurement paths."""

import json
from inspect import signature


def generate_causal_tokens(
    model,
    prompt,
    attention,
    new_tokens,
    pad_token_id,
    *,
    eos_token_id,
    static_cache=False,
    compile_static=False,
    min_new_tokens=0,
    logits_processor=None,
    configuration_observer=None,
):
    """Generate with explicit stopping/cache policy, never inherited sampling.

    Args:
        model: Prepared HF causal model; loading/fusion must already be complete.
        prompt: Padded token IDs on the model device.
        attention: Attention mask marking actual prompt tokens.
        new_tokens: Maximum continuation length.
        pad_token_id: Existing vocabulary token used for batch padding.
        eos_token_id: Stop ID(s); explicit None disables EOS for diagnostics only.
        static_cache: Use HF's bounded static cache rather than its dynamic cache.
        compile_static: Enable HF-managed fullgraph max-autotune compilation.
        min_new_tokens: Minimum continuation length; zero for normal generation.
        logits_processor: Optional HF processors for separate quality diagnostics.
        configuration_observer: Optional synchronous resolved-config observer.

    Returns:
        HF sequence tensor, including the padded input prefix.

    Raises:
        ValueError: Compilation requested without a static cache.
        RuntimeError: HF/model/kernel execution fails; there is no fallback.
    """
    from transformers import CompileConfig, GenerationConfig

    if compile_static and not static_cache:
        raise ValueError("Compiled generation requires static cache")
    neutral = {
        "do_sample": False,
        "eos_token_id": eos_token_id,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "repetition_penalty": 1.0,
        "encoder_repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "bad_words_ids": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "sequence_bias": None,
        "watermarking_config": None,
        "renormalize_logits": False,
        "remove_invalid_values": False,
        "token_healing": False,
        "guidance_scale": 1.0,
        "exponential_decay_length_penalty": None,
        "max_time": None,
        "stop_strings": None,
        "output_scores": False,
        "output_logits": False,
        "output_attentions": False,
        "output_hidden_states": False,
    }
    # Empty containers survive newer HF default merging before explicit kwargs.
    config = GenerationConfig(
        **{
            **neutral,
            "bad_words_ids": [],
            "suppress_tokens": [],
            "begin_suppress_tokens": [],
            "sequence_bias": {},
        },
        num_beams=1,
        num_return_sequences=1,
        return_dict_in_generate=False,
        max_new_tokens=new_tokens,
        min_new_tokens=min_new_tokens,
        pad_token_id=pad_token_id,
        use_cache=True,
        cache_implementation="static" if static_cache else None,
        disable_compile=not compile_static,
        compile_config=CompileConfig(mode="max-autotune", fullgraph=True),
    )
    defaults = (
        {"use_model_defaults": False}
        if "use_model_defaults" in signature(model.generate).parameters
        else {}
    )
    if configuration_observer is not None:
        resolved, _ = model._prepare_generation_config(config, **defaults, **neutral)
        configuration_observer(json.loads(resolved.to_json_string(use_diff=False)))
    return model.generate(
        input_ids=prompt,
        attention_mask=attention,
        generation_config=config,
        **defaults,
        **neutral,
        **(
            {"logits_processor": logits_processor}
            if logits_processor is not None
            else {}
        ),
    )
