# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午10:20
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Real local Llama generation through the product inference path."""

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from parascale.commands.run import run_inference_from_config  # noqa: E402
from tests.test_causal_lm_workload import (  # noqa: E402, F401
    configuration,
    prepare,
)
from tests.test_causal_lm_workload import (  # noqa: E402
    model_path as model_path,
)


def infer_config(model_path, tmp_path, **options):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": "one", "prompt": "a b"},
                {"id": "two", "prompt": "a b c d"},
                {"id": "three", "prompt": "c d"},
            ]
        ),
        encoding="utf-8",
    )
    return {
        "model": {"pretrained_model_name_or_path": str(model_path)},
        "data": {"type": "prompts_jsonl", "path": str(path)},
        "runtime": {"device": "cpu", "run_dir": str(tmp_path / "infer-run")},
        "inference": {
            "workload": "causal_lm",
            "precision": "fp32",
            "batch_size": 2,
            "max_input_tokens": 8,
            "max_new_tokens": 3,
            "warmup_steps": 2,
            **options,
        },
    }


def test_real_product_float_generation_matches_hf_greedy(model_path, tmp_path):
    result = run_inference_from_config(infer_config(model_path, tmp_path))
    records = [row for batch in result["outputs"] for row in batch["records"]]
    assert [row["id"] for row in records] == ["one", "two", "three"]
    assert result["metrics"]["requests"] == 3
    assert result["measurement_window"]["measured_batches"] == 2
    assert result["metrics"]["prompt_tokens"] == 8
    assert result["metrics"]["generated_tokens"] == sum(
        row["generated_tokens"] for row in records
    )
    assert result["execution"]["backend"] == "float"
    assert result["execution"]["tokenizer_identity"]
    assert result["mock"] is False and result["synthetic"] is False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, padding_side="left"
    )
    reference = transformers.AutoModelForCausalLM.from_pretrained(model_path).eval()
    for row, prompt in zip(records, ["a b", "a b c d", "c d"]):
        encoded = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
        with torch.inference_mode():
            expected = reference.generate(
                **encoded, do_sample=False, max_new_tokens=3, use_cache=True
            )
        ids = expected[0, encoded["input_ids"].shape[1] :].tolist()
        if 1 in ids:
            ids = ids[: ids.index(1) + 1]
        assert row["generated_token_ids"] == ids
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "precision,dtype",
    [("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)],
)
def test_float_loader_uses_requested_parameter_dtype(
    model_path, tmp_path, monkeypatch, precision, dtype
):
    from parascale.workloads.causal_lm_inference import (
        CausalLmInferenceModel,
        build_causal_lm_inference_components,
    )

    config = infer_config(model_path, tmp_path, precision=precision)
    config["runtime"]["device"] = "cuda:0"
    # Test dtype conversion on CPU without pretending to validate CUDA execution.
    monkeypatch.setattr(
        CausalLmInferenceModel, "resolve_device", lambda self: ("cpu", lambda: None)
    )
    model, _, _ = build_causal_lm_inference_components(config)
    try:
        model.eval()
        assert {parameter.dtype for parameter in model._model.parameters()} == {dtype}
        assert model.execution_metadata["precision"] == precision
    finally:
        model.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_fp16_generation_matches_native_hf(model_path, tmp_path):
    config = infer_config(model_path, tmp_path, precision="fp16")
    config["runtime"]["device"] = "cuda:0"
    result = run_inference_from_config(config)
    records = [row for batch in result["outputs"] for row in batch["records"]]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, padding_side="left"
    )
    reference = (
        transformers.AutoModelForCausalLM.from_pretrained(model_path)
        .eval()
        .to(device="cuda:0", dtype=torch.float16)
    )
    prompts = ["a b", "a b c d", "c d"]
    expected_ids = []
    for start in range(0, len(prompts), 2):
        encoded = tokenizer(
            prompts[start : start + 2],
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to("cuda:0")
        with torch.inference_mode():
            generated = reference.generate(
                **encoded, do_sample=False, max_new_tokens=3, use_cache=True
            )
        for sequence in generated[:, encoded["input_ids"].shape[1] :].tolist():
            expected_ids.append(
                sequence[: sequence.index(1) + 1] if 1 in sequence else sequence
            )
    assert [row["generated_token_ids"] for row in records] == expected_ids
    assert result["execution"]["precision"] == "fp16"
    assert result["execution"]["device"] == "cuda:0"
    assert result["metrics"]["requests"] == 3
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mode", ["ptq", "qat"])
def test_product_export_reloads_without_float_weights_for_inference(
    model_path,
    tmp_path,
    capsys,
    mode,
):
    from parascale.cli import main

    corpus = prepare(tmp_path, model_path, capsys)
    train = configuration(model_path, corpus, tmp_path / "checkpoint")
    train["parascale"]["quantization"] = {
        "enabled": True,
        "mode": mode,
        "bits": 8,
        "calib_batches": 1,
    }
    artifact = tmp_path / "quantized.pt"
    train["training"]["quantized_output"] = str(artifact)
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(train), encoding="utf-8")
    args = ["quantize" if mode == "ptq" else "train", "--config", str(config_path)]
    if mode == "ptq":
        args += ["--output", str(artifact)]
    assert main(args) == 0
    capsys.readouterr()
    for path in model_path.glob("*.safetensors"):
        path.unlink()
    config = infer_config(
        model_path, tmp_path, execution="reference", artifact_path=str(artifact)
    )
    result = run_inference_from_config(config)
    assert result["metrics"]["requests"] == 3
    assert result["execution"]["artifact"]["mode"] == mode
    assert result["execution"]["artifact"]["sha256"]
    assert result["execution"]["backend"] == "reference"


def test_streamed_outputs_preserve_every_request_with_zero_retention(
    model_path, tmp_path
):
    from parascale.cli import main

    path = tmp_path / "generations.jsonl"
    config = infer_config(model_path, tmp_path, output_jsonl=str(path), max_outputs=0)
    source = tmp_path / "infer.json"
    summary = tmp_path / "summary.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    assert main(["infer", "--config", str(source), "--output", str(summary)]) == 0
    result = json.loads(summary.read_text(encoding="utf-8"))
    assert result["outputs"] == []
    assert result["output_stream"]["complete"] is True
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in records] == ["one", "two", "three"]
    with pytest.raises(FileExistsError):
        run_inference_from_config(config)


def test_failed_stream_keeps_partial_and_closes_model(model_path, tmp_path):
    from parascale.runtime.inference.causal_lm import run_causal_lm_inference
    from parascale.workloads.causal_lm_inference import (
        build_causal_lm_inference_components,
    )

    output = tmp_path / "records.jsonl"
    config = infer_config(model_path, tmp_path, output_jsonl=str(output), batch_size=1)
    with (tmp_path / "prompts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\nnot-json\n")
    model, batches, _ = build_causal_lm_inference_components(config)
    with pytest.raises(ValueError, match="Invalid prompt at line 4"):
        run_causal_lm_inference(model, batches)
    assert not output.exists()
    partial = tmp_path / "records.jsonl.partial"
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 3
    assert model._model is None and model._closed


def test_long_prompts_fail_without_silent_truncation(model_path, tmp_path):
    config = infer_config(model_path, tmp_path, max_input_tokens=3)
    with pytest.raises(ValueError, match="max_input_tokens"):
        run_inference_from_config(config)


def test_context_budget_is_checked(model_path, tmp_path):
    config = infer_config(model_path, tmp_path, max_input_tokens=30, max_new_tokens=4)
    with pytest.raises(ValueError, match="position|context"):
        run_inference_from_config(config)


def test_distributed_launch_rejected_before_model_load(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    config = infer_config(tmp_path / "missing", tmp_path)
    with pytest.raises(ValueError, match="single.device|distributed"):
        run_inference_from_config(config)


def test_unavailable_cuda_does_not_fallback_to_cpu(model_path, tmp_path, monkeypatch):
    config = infer_config(model_path, tmp_path, precision="bf16")
    config["runtime"]["device"] = "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        run_inference_from_config(config)


def test_generation_counts_include_first_eos_but_not_fill():
    from parascale.workloads.causal_lm_inference import generation_output

    tokenizer = type(
        "Tokenizer", (), {"decode": lambda self, ids, **kwargs: str(ids)}
    )()
    result = generation_output(
        [[0, 3, 1, 1, 1], [3, 4, 5, 6, 1]],
        input_width=2,
        prompt_lengths=[1, 2],
        ids=["a", "b"],
        tokenizer=tokenizer,
        eos_ids={1},
    )
    assert result["counts"] == {
        "requests": 2,
        "prompt_tokens": 3,
        "generated_tokens": 4,
    }
    assert [row["generated_token_ids"] for row in result["records"]] == [[1], [5, 6, 1]]
    assert all(row["finish_reason"] == "eos" for row in result["records"])


def test_generation_policy_explicitly_keeps_eos_and_disables_implicit_compile():
    from parascale.runtime.inference.generation import generate_causal_tokens

    class Model:
        def generate(self, *, use_model_defaults=None, **kwargs):
            assert use_model_defaults is False
            config = kwargs["generation_config"]
            assert config.max_new_tokens == 4 and config.min_new_tokens == 0
            assert config.disable_compile is True
            assert config.do_sample is False and config.num_beams == 1
            assert kwargs["eos_token_id"] == [1, 2]
            assert kwargs["forced_eos_token_id"] is None
            return "generated"

    assert (
        generate_causal_tokens(Model(), "prompt", "mask", 4, 0, eos_token_id=[1, 2])
        == "generated"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or transformers.__version__ != "4.51.3",
    reason="requires CUDA and qualified Transformers 4.51.3",
)
def test_compiled_static_cache_preserves_unrelated_requests(model_path, tmp_path):
    config = infer_config(
        model_path,
        tmp_path,
        precision="bf16",
        batch_size=1,
        attention_policy="math",
        warmup_steps=2,
    )
    config["runtime"]["device"] = "cuda:0"
    eager = run_inference_from_config(config)
    config["inference"].update(
        cache_implementation="static", compile_mode="max-autotune"
    )
    compiled = run_inference_from_config(config)
    assert compiled["outputs"] == eager["outputs"]
    assert compiled["metrics"]["requests"] == 3
    assert compiled["measurement_window"]["warmup_steps_effective"] == 2
    policy = compiled["execution"]["generation_config"]
    assert policy["cache_implementation"] == "static"
    assert policy["disable_compile"] is False
