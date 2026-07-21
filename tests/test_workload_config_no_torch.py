# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

def test_vlm_lora_fsdp_uses_backend_activation_checkpointing_without_torch():
    from parascale.workloads.vlm_lora import VlmLoraSpec

    config = {
        "parascale": {
            "training_backend": "fsdp",
            "enable_activation_checkpointing": True,
            "batch_size": 1,
        },
        "model": {
            "type": "vlm_lora",
            "activation_checkpointing": True,
        },
        "data": {
            "batch_size": 1,
        },
    }

    spec = VlmLoraSpec.from_config(config)

    assert spec.activation_checkpointing is False

def test_default_workload_registry_resolves_builtin_aliases_without_torch():
    from parascale.runtime import WorkloadRegistry
    from parascale.workloads.registry import default_workload_registry

    registry = default_workload_registry()

    assert isinstance(registry, WorkloadRegistry)
    assert registry.resolve("tiny_clip") == "clip_contrastive"
    assert registry.resolve("vlm_lora_finetune") == "vlm_lora"
    assert registry.resolve("yoloworld") == "yolo_world"
    assert registry.resolve("synthetic_regression") == "torch_tiny"
    assert "clip_contrastive" in registry.names()

def test_workload_specs_are_imported_from_scenario_modules_without_torch():
    from parascale.workloads.specs.vlm_lora import VlmLoraSpec as SpecsVlmLoraSpec
    from parascale.workloads.vlm_lora import VlmLoraSpec as WorkloadVlmLoraSpec

    assert WorkloadVlmLoraSpec is SpecsVlmLoraSpec

def test_vlm_lora_batches_account_for_gradient_accumulation_without_torch():
    from parascale.workloads.vlm_lora import VlmLoraSpec

    config = {
        "parascale": {
            "training_backend": "fsdp",
            "gradient_accumulation_steps": 4,
            "batch_size": 1,
        },
        "training": {
            "max_steps": 150,
            "gradient_accumulation_steps": 4,
        },
        "data": {
            "batch_size": 1,
        },
    }

    spec = VlmLoraSpec.from_config(config)

    assert spec.num_batches == 600

def test_vlm_lora_pipeline_cache_and_worker_preprocess_config_without_torch():
    from parascale.workloads.vlm_lora import VlmLoraSpec

    config = {
        "parascale": {
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": False,
            "dataloader_prefetch_factor": 4,
            "dataloader_persistent_workers": True,
            "dataset_local_cache_dir": ".parascale_cache/dataset",
            "preprocess_in_workers": True,
            "pipeline_cache": True,
            "pipeline_cache_dir": ".parascale_cache/vlm_test",
            "pipeline_cache_max_entries": 8,
            "pipeline_cache_max_bytes": 2048,
            "pipeline_cache_ttl_seconds": 30.0,
            "prompt_template_cache": True,
            "prompt_template_cache_dir": ".parascale_cache/vlm_prompts",
            "cuda_prefetch": True,
        },
        "data": {
            "batch_size": 1,
            "streaming": True,
        },
        "training": {
            "max_steps": 2,
        },
    }

    spec = VlmLoraSpec.from_config(config)

    assert spec.num_workers == 4
    assert spec.pin_memory is False
    assert spec.prefetch_factor == 4
    assert spec.persistent_workers is True
    assert spec.dataset_local_cache_dir == ".parascale_cache/dataset"
    assert spec.preprocess_in_workers is True
    assert spec.pipeline_cache is True
    assert spec.pipeline_cache_dir == ".parascale_cache/vlm_test"
    assert spec.pipeline_cache_max_entries == 8
    assert spec.pipeline_cache_max_bytes == 2048
    assert spec.pipeline_cache_ttl_seconds == 30.0
    assert spec.prompt_template_cache is True
    assert spec.prompt_template_cache_dir == ".parascale_cache/vlm_prompts"
    assert spec.cuda_prefetch is True

def test_yolo_world_dataloader_and_tensor_cache_config_without_torch():
    from parascale.workloads.specs.yolo import YoloWorldSpec

    spec = YoloWorldSpec.from_config(
        {
            "parascale": {
                "dataloader_num_workers": 4,
                "dataloader_pin_memory": False,
                "dataloader_prefetch_factor": 3,
                "dataloader_persistent_workers": True,
                "tensor_cache": True,
                "tensor_cache_dir": ".parascale_cache/yolo_tensor",
            },
            "model": {"path": "/models/yolov8s-worldv2.pt"},
            "data": {
                "type": "objects365_yolo_cache",
                "data_dir": "/dataset/cache/objects365_tiny_yolo",
                "batch_size": 2,
            },
            "training": {"workload": "yolo_world_detection", "max_steps": 5},
        }
    )

    assert spec.num_workers == 4
    assert spec.pin_memory is False
    assert spec.prefetch_factor == 3
    assert spec.persistent_workers is True
    assert spec.tensor_cache is True
    assert spec.tensor_cache_dir == ".parascale_cache/yolo_tensor"
    assert spec.num_batches == 5

def test_vlm_processor_timer_captures_nested_components_without_torch():
    from parascale.workloads.vlm_cache import _timed_vlm_processor_call

    class Component:
        def __call__(self, *_args, **_kwargs):
            return {"ok": True}

    class Processor:
        def __init__(self):
            self.tokenizer = Component()
            self.image_processor = Component()

        def __call__(self, **kwargs):
            self.tokenizer(kwargs["text"])
            self.image_processor(kwargs["images"])
            return {"input_ids": [[1, 2, 3]]}

    processor = Processor()

    encoded, profile = _timed_vlm_processor_call(
        processor, text=["hello"], images=["image"]
    )

    assert encoded["input_ids"] == [[1, 2, 3]]
    assert profile["tokenizer_ms"] >= 0.0
    assert profile["image_processor_ms"] >= 0.0
    assert isinstance(processor.tokenizer, Component)
    assert isinstance(processor.image_processor, Component)

def test_vlm_prompt_disk_cache_without_torch():
    import shutil
    from pathlib import Path

    from parascale.workloads.vlm_cache import (
        _VLM_PROMPT_CACHE,
        _vlm_prompt,
    )
    from parascale.workloads.vlm_lora import VlmLoraSpec

    class Processor:
        pass

    cache_dir = Path(".pytest-parascale") / "prompt-cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = VlmLoraSpec(
        conversation_template="llava_onevision",
        response_template="answer",
        prompt_template_cache=True,
        prompt_template_cache_dir=str(cache_dir),
    )
    sample = {"text": "what is here?"}
    _VLM_PROMPT_CACHE.clear()

    first = _vlm_prompt(Processor(), sample, spec)
    _VLM_PROMPT_CACHE.clear()
    second = _vlm_prompt(Processor(), sample, spec)

    assert first == second
    assert list(cache_dir.glob("*.txt"))

def test_data_samplers_without_torch():
    from parascale.data import (
        LengthBucketSampler,
        MultiModalCollator,
        TokenBudgetBatchSampler,
        estimate_sample_tokens,
    )

    dataset = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [1] * 16},
        {"input_ids": [1] * 8},
    ]

    assert estimate_sample_tokens(dataset[0]).text_tokens == 3
    length_batches = list(LengthBucketSampler(dataset, batch_size=2, shuffle=False))
    token_batches = list(TokenBudgetBatchSampler(dataset, max_tokens=10, shuffle=False))

    assert length_batches
    assert token_batches

    collator = MultiModalCollator(pad_token_id=0, max_length=6)
    batch = collator(
        [{"input_ids": [1, 2], "labels": [1, 2]}, {"input_ids": [3], "labels": [3]}]
    )
    assert batch["input_ids"] == [[1, 2], [3, 0]]
    assert batch["labels"] == [[1, 2], [3, -100]]
