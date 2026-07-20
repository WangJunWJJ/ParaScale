# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

from dataclasses import is_dataclass

import pytest


def test_optimizer_spec_defaults_to_adamw_without_torch():
    from parascale.optimizers.spec import OptimizerSpec

    spec = OptimizerSpec.from_config({"optimizer": {"lr": 0.002}})

    assert spec.type == "adamw"
    assert spec.lr == 0.002


@pytest.mark.parametrize("optimizer_type", ["four_bit_adamw", "four_bit_sgd"])
def test_optimizer_spec_accepts_configured_four_bit_types(optimizer_type):
    from parascale.optimizers.spec import OptimizerSpec

    spec = OptimizerSpec.from_config(
        {
            "optimizer": {
                "type": optimizer_type,
                "lr": 0.01,
                "group_size": 64,
                "compensate_quant_error": True,
                "error_compensation_dtype": "fp32",
            }
        }
    )

    assert spec.type == optimizer_type
    assert spec.group_size == 64
    assert spec.to_metadata()["state_schema_version"] == 1


def test_optimizer_spec_rejects_fields_for_another_optimizer_type():
    from parascale.optimizers.spec import OptimizerSpec

    with pytest.raises(ValueError, match="momentum.*four_bit_adamw"):
        OptimizerSpec.from_config(
            {"optimizer": {"type": "four_bit_adamw", "momentum": 0.9}}
        )


def test_optimizer_spec_accepts_block_scaled_fp16_residuals():
    from parascale.optimizers.spec import OptimizerSpec

    spec = OptimizerSpec.from_config(
        {
            "optimizer": {
                "type": "four_bit_adamw",
                "error_compensation_dtype": "fp16",
                "error_compensation_mode": "block_scaled",
            }
        }
    )

    assert spec.error_compensation_mode == "block_scaled"
    assert spec.to_metadata()["error_compensation_mode"] == "block_scaled"


@pytest.mark.parametrize("backend", ["fsdp", "deepspeed"])
def test_four_bit_optimizer_spec_rejects_sharded_backends(backend):
    from parascale.optimizers.spec import OptimizerSpec

    spec = OptimizerSpec.from_config(
        {"optimizer": {"type": "four_bit_adamw"}}
    )

    with pytest.raises(ValueError, match=backend):
        spec.validate_backend(backend, zero_stage=0)


def test_four_bit_optimizer_spec_rejects_native_zero_stage_one():
    from parascale.optimizers.spec import OptimizerSpec

    spec = OptimizerSpec.from_config(
        {"optimizer": {"type": "four_bit_sgd"}}
    )

    with pytest.raises(ValueError, match="zero_stage=1"):
        spec.validate_backend("native_ddp", zero_stage=1)



def test_backend_config_roundtrip_without_torch():
    from parascale.config import ParaScaleConfig

    config = ParaScaleConfig(
        training_backend="deepspeed",
        zero_optimization=True,
        zero_stage=3,
        zero_offload=True,
        precision="bf16",
        grad_clip_norm=1.0,
        label_keys=["labels"],
        dataset_local_cache_dir=".parascale_cache/dataset",
        tensor_cache=True,
        tensor_cache_dir=".parascale_cache/tensor",
        cuda_prefetch=True,
        cuda_prefetch_device="cuda:0",
        tuner_dataloader_wait_threshold_ms=7.5,
        preprocess_in_workers=True,
        pipeline_cache=True,
        pipeline_cache_dir=".parascale_cache/test",
        pipeline_cache_max_entries=16,
        pipeline_cache_max_bytes=1024,
        pipeline_cache_ttl_seconds=60.0,
        prompt_template_cache=True,
        prompt_template_cache_dir=".parascale_cache/prompts",
        fsdp_state_dict_type="sharded",
        fsdp_activation_checkpointing_policy="transformer_auto",
        fsdp_checkpoint_module_classes=["Qwen2DecoderLayer"],
        allow_world_size_change_on_resume=True,
    )

    restored = ParaScaleConfig.from_dict(config.to_dict())

    assert restored.training_backend == "deepspeed"
    assert restored.zero_stage == 3
    assert restored.zero_offload is True
    assert restored.precision == "bf16"
    assert restored.grad_clip_norm == 1.0
    assert restored.label_keys == ["labels"]
    assert restored.dataset_local_cache_dir == ".parascale_cache/dataset"
    assert restored.tensor_cache is True
    assert restored.tensor_cache_dir == ".parascale_cache/tensor"
    assert restored.cuda_prefetch is True
    assert restored.cuda_prefetch_device == "cuda:0"
    assert restored.tuner_dataloader_wait_threshold_ms == 7.5
    assert restored.preprocess_in_workers is True
    assert restored.pipeline_cache is True
    assert restored.pipeline_cache_dir == ".parascale_cache/test"
    assert restored.pipeline_cache_max_entries == 16
    assert restored.pipeline_cache_max_bytes == 1024
    assert restored.pipeline_cache_ttl_seconds == 60.0
    assert restored.prompt_template_cache is True
    assert restored.prompt_template_cache_dir == ".parascale_cache/prompts"
    assert restored.fsdp_state_dict_type == "sharded"
    assert restored.fsdp_activation_checkpointing_policy == "transformer_auto"
    assert restored.fsdp_checkpoint_module_classes == ["Qwen2DecoderLayer"]
    assert restored.allow_world_size_change_on_resume is True


def test_config_accepts_ascend_native_backend_without_torch():
    from parascale.config import ParaScaleConfig

    config = ParaScaleConfig(training_backend="ascend_native")

    assert config.training_backend == "ascend_native"


def test_device_prefetch_config_is_generic_and_cuda_compatible_without_torch():
    from parascale.config import ParaScaleConfig

    config = ParaScaleConfig(device_prefetch=True, prefetch_device="npu:0")

    assert config.device_prefetch is True
    assert config.prefetch_device == "npu:0"
    assert config.cuda_prefetch is True
    assert config.cuda_prefetch_device == "npu:0"
    assert config.to_dict()["device_prefetch"] is True
    assert config.to_dict()["prefetch_device"] == "npu:0"


def test_legacy_cuda_prefetch_config_populates_generic_fields_without_torch():
    from parascale.config import ParaScaleConfig

    config = ParaScaleConfig.from_dict(
        {"cuda_prefetch": True, "cuda_prefetch_device": "cuda:1"}
    )

    assert config.device_prefetch is True
    assert config.prefetch_device == "cuda:1"
    assert config.cuda_prefetch is True
    assert config.cuda_prefetch_device == "cuda:1"


def test_layered_config_roundtrip_without_torch():
    from parascale.config import LayeredParaScaleConfig, ParaScaleConfig

    config = ParaScaleConfig(
        task_type="multimodal",
        model_family="clip",
        training_backend="deepspeed",
        zero_stage=2,
        dataloader_num_workers=8,
        tensor_cache=True,
        tensor_cache_dir=".parascale_cache/tensor",
        pipeline_cache=True,
        checkpoint_save_path="runs/ckpt",
    )

    layered = config.to_layered()
    layered_dict = config.to_layered_dict()
    restored = ParaScaleConfig.from_dict(layered_dict)

    assert isinstance(layered, LayeredParaScaleConfig)
    assert layered.workload.task_type == "multimodal"
    assert layered.backend.training_backend == "deepspeed"
    assert layered.backend.zero_stage == 2
    assert layered.data.dataloader_num_workers == 8
    assert layered.data.tensor_cache is True
    assert layered.data.tensor_cache_dir == ".parascale_cache/tensor"
    assert layered.data.pipeline_cache is True
    assert layered.training.checkpoint_save_path == "runs/ckpt"
    assert restored.to_dict() == config.to_dict()


def test_config_field_map_covers_layered_and_flat_configs():
    from parascale.config import (
        BackendConfig,
        DataPipelineConfig,
        ParallelConfig,
        ParaScaleConfig,
        TrainingRunConfig,
        WorkloadConfig,
        flat_config_fields_by_section,
    )

    fields_by_section = flat_config_fields_by_section()
    section_types = {
        "workload": WorkloadConfig,
        "parallel": ParallelConfig,
        "backend": BackendConfig,
        "data": DataPipelineConfig,
        "training": TrainingRunConfig,
    }
    flat_fields = set(ParaScaleConfig().to_dict()) - {"quantization"}

    for section, config_type in section_types.items():
        assert set(fields_by_section[section]) == set(config_type().to_dict())

    mapped_fields = {
        field for fields in fields_by_section.values() for field in fields
    }
    assert mapped_fields == flat_fields


def test_parascale_config_defaults_are_derived_from_layered_schema():
    from parascale.config import LayeredParaScaleConfig, ParaScaleConfig

    defaults = LayeredParaScaleConfig().to_flat_dict()
    config = ParaScaleConfig()

    assert not is_dataclass(ParaScaleConfig)
    assert config.to_dict() == defaults
    assert config.to_layered_dict() == LayeredParaScaleConfig().to_dict()


def test_layered_config_copies_nested_resolution_buckets_without_torch():
    from parascale.config import ParaScaleConfig

    config = ParaScaleConfig(resolution_buckets=[[224, 224], [336, 336]])
    layered = config.to_layered()

    config.resolution_buckets[0][0] = 128

    assert layered.data.resolution_buckets == [[224, 224], [336, 336]]


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


def test_deepspeed_checkpoint_path_parser_without_torch():
    import os

    def parse_checkpoint_path(checkpoint_path, tag=None):
        if tag is not None:
            return checkpoint_path, tag
        base = os.path.basename(os.path.normpath(checkpoint_path))
        if base.startswith("checkpoint_"):
            return os.path.dirname(os.path.normpath(checkpoint_path)), base
        return checkpoint_path, None

    load_dir, tag = parse_checkpoint_path("runs/ckpt", tag="global_step10")
    assert load_dir == "runs/ckpt"
    assert tag == "global_step10"

    load_dir, tag = parse_checkpoint_path("runs/ckpt/checkpoint_10")
    assert load_dir.replace("\\", "/") == "runs/ckpt"
    assert tag == "checkpoint_10"


def test_deepspeed_config_merges_parascale_values_without_torch():
    from parascale.config import ParaScaleConfig
    from parascale.runtime.backends.deepspeed import DeepSpeedTrainingBackend

    config = ParaScaleConfig(
        training_backend="deepspeed",
        batch_size=4,
        gradient_accumulation_steps=3,
        precision="bf16",
        zero_stage=3,
        zero_offload=True,
        deepspeed_config={
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": 1},
            "steps_per_print": 5,
        },
    )

    merged = DeepSpeedTrainingBackend(config=config).build_deepspeed_config()

    assert merged["train_micro_batch_size_per_gpu"] == 4
    assert merged["gradient_accumulation_steps"] == 3
    assert merged["zero_optimization"]["stage"] == 3
    assert merged["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert merged["bf16"]["enabled"] is True
    assert merged["steps_per_print"] == 5
    assert merged["_parascale"]["merged_user_config"] is True


def test_deepspeed_config_preserves_zero_stage_zero_without_torch():
    from parascale.config import ParaScaleConfig
    from parascale.runtime.backends.deepspeed import DeepSpeedTrainingBackend

    config = ParaScaleConfig(
        training_backend="deepspeed",
        zero_stage=0,
        batch_size=2,
        gradient_accumulation_steps=1,
    )

    merged = DeepSpeedTrainingBackend(config=config).build_deepspeed_config()

    assert merged["zero_optimization"]["stage"] == 0
    assert merged["_parascale"]["resolved_config"] is True


def test_workload_capability_lives_outside_orchestrator_without_torch(monkeypatch):
    from parascale.workloads.capability import capability_level_for_training

    monkeypatch.setenv("WORLD_SIZE", "1")
    config = {
        "training": {"workload": "clip_contrastive"},
        "data": {"type": "datacomp_wds"},
        "hardware_profile": {"world_size": 2, "gpus_per_node": 1, "num_nodes": 2},
    }

    assert capability_level_for_training(config) == "multi_node_smoke"


def test_benchmark_aggregation_lives_in_reporting_without_torch():
    from parascale.reporting.aggregation import aggregate_stable_metrics

    metrics = aggregate_stable_metrics(
        [
            {"images_per_second": 10.0, "step_time_seconds": 0.2},
            {"images_per_second": 20.0, "step_time_seconds": 0.1},
        ],
        warmup_steps=1,
    )

    assert metrics["measured_steps"] == 1.0
    assert metrics["stable_images_per_second"] == 20.0
    assert metrics["stable_step_time_ms"] == 100.0


def test_benchmark_aggregation_includes_stable_loss_window():
    from parascale.reporting.aggregation import aggregate_stable_metrics

    metrics = aggregate_stable_metrics(
        [{"loss": 9.0}, {"loss": 2.0}, {"loss": 1.0}],
        warmup_steps=1,
    )

    assert metrics["stable_loss"] == 1.5
    assert metrics["stable_min_loss"] == 1.0
    assert metrics["stable_max_loss"] == 2.0


def test_train_runner_does_not_own_workload_capability_or_aggregation_without_torch():
    from pathlib import Path

    source = Path("parascale/runtime/train_runner.py").read_text(encoding="utf-8")

    assert "workload_flags" not in source
    assert "capability_level_for_training" not in source
    assert "def _aggregate_stable_metrics" not in source
    assert "local_native_clip_contrastive" not in source
    assert "local_native_vlm_lora" not in source


def test_synthetic_regression_is_built_through_workload_registry_without_torch():
    from pathlib import Path

    train_runner_source = Path("parascale/runtime/train_runner.py").read_text(
        encoding="utf-8"
    )
    command_source = Path("parascale/commands/run.py").read_text(encoding="utf-8")

    assert "build_synthetic_regression_components" not in train_runner_source
    assert "SyntheticRegressionSpec" not in train_runner_source
    assert 'if flags["synthetic"]:' not in train_runner_source
    assert "build_synthetic_regression_components" not in command_source


def test_optimizer_parameter_selection_filters_frozen_params_without_torch():
    from parascale.workloads.registry import trainable_parameter_stats

    class Param:
        def __init__(self, count, requires_grad):
            self.requires_grad = requires_grad
            self.count = count

        def numel(self):
            return self.count

    class Model:
        def parameters(self):
            return [
                Param(10, False),
                Param(5, True),
                Param(15, True),
            ]

    selected, stats = trainable_parameter_stats(Model())

    assert len(selected) == 2
    assert stats["trainable_params"] == 20
    assert stats["total_params"] == 30
    assert stats["trainable_ratio"] == 20 / 30


def test_optimizer_parameter_selection_rejects_all_frozen_without_torch():
    from parascale.workloads.registry import trainable_parameter_stats

    class Param:
        requires_grad = False

        def numel(self):
            return 10

    class Model:
        def parameters(self):
            return [Param()]

    try:
        trainable_parameter_stats(Model())
    except RuntimeError as exc:
        assert "no trainable parameters" in str(exc)
    else:
        raise AssertionError("optimizer construction must reject all-frozen models")


def test_vlm_lora_uses_shared_trainable_parameter_selection_without_torch():
    from pathlib import Path

    source = Path("parascale/workloads/vlm_lora.py").read_text(encoding="utf-8")

    assert "from .optimizer import build_adamw_optimizer_for_model" in source
    assert "trainable_parameters = [" not in source
    assert "optim.AdamW(trainable_parameters" not in source


def test_clip_workload_dataloader_does_not_own_device_placement_without_torch():
    from pathlib import Path

    source = Path("parascale/workloads/clip.py").read_text(encoding="utf-8")
    forbidden = [
        'batch["pixel_values"] = batch["pixel_values"].to(device)',
        'batch["input_ids"] = batch["input_ids"].to(device)',
        'batch["attention_mask"] = batch["attention_mask"].to(device)',
        "device=device",
    ]

    for snippet in forbidden:
        assert snippet not in source


def test_training_workload_dataloaders_do_not_own_device_placement_without_torch():
    from pathlib import Path

    forbidden_by_file = {
        "parascale/workloads/vision.py": [
            'batch["pixel_values"].to(device)',
            'batch["labels"].to(device)',
        ],
        "parascale/workloads/tiny.py": [
            "x.to(device)",
            "y.to(device)",
        ],
        "parascale/workloads/yolo.py": [
            "torch.stack(images, dim=0).to(device)",
        ],
        "parascale/workloads/vlm_lora.py": [
            "_vlm_move_batch_to_device(",
            'batch_device = "cpu" if spec.cuda_prefetch else device',
            'batch["pixel_values"] = batch["pixel_values"].to(batch_device)',
            'batch["input_ids"] = batch["input_ids"].to(batch_device)',
            'batch["attention_mask"] = batch["attention_mask"].to(batch_device)',
        ],
    }

    for path, snippets in forbidden_by_file.items():
        source = Path(path).read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet not in source


def test_training_workloads_do_not_own_model_placement_without_torch():
    from pathlib import Path

    forbidden_by_file = {
        "parascale/workloads/vision.py": [
            "TinyPatchClassifier().to(device)",
        ],
        "parascale/workloads/tiny.py": [
            "TinyTorchMLP().to(device)",
        ],
        "parascale/workloads/yolo.py": [
            "yolo.model.to(device)",
        ],
        "parascale/workloads/vlm_lora.py": [
            "TinyVlmLora().to(device)",
            "HfVlmLoraWrapper(peft_model).to(device)",
            ").to(device)",
            "AutoModel.from_pretrained(spec.pretrained_model_name_or_path).to(device)",
            "HFClipLoraAdapter(base_model).to(device)",
        ],
    }

    for path, snippets in forbidden_by_file.items():
        source = Path(path).read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet not in source


def test_dependency_metadata_matches_runtime_capabilities_without_torch():
    from pathlib import Path

    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "torch>=2.4.0" in requirements
    assert "torchvision>=0.19.0" in requirements
    assert "[project.optional-dependencies]" in pyproject
    assert "deepspeed = [" in pyproject
    assert "datacomp = [" in pyproject
    assert "vlm = [" in pyproject
    assert "yolo = [" in pyproject
    assert "ascend = [" in pyproject
    assert '"deepspeed>=' in pyproject
    assert '"transformers>=' in pyproject
    assert '"peft>=' in pyproject
    assert '"webdataset>=' in pyproject
    assert '"pandas>=' in pyproject
    assert '"pyarrow>=' in pyproject
    assert '"pillow>=' in pyproject
    assert '"ultralytics>=' in pyproject
    assert '"torch-npu>=' in pyproject
