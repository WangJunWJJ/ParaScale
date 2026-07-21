# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

from dataclasses import is_dataclass


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
