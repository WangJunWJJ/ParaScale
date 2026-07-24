# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Typed configuration objects for ParaScale."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

try:
    from .quantization.base import QuantizationConfig
except ImportError:

    @dataclass
    class QuantizationConfig:
        enabled: bool = False
        mode: Literal["qat", "ptq"] = "qat"
        bits: int = 8
        scheme: Literal["symmetric", "asymmetric"] = "symmetric"
        per_channel: bool = True
        observer_type: Literal["minmax", "moving_average"] = "minmax"
        moving_average_ratio: float = 0.9
        fuse_modules: bool = True
        qat_epochs: int = 10
        calib_batches: int = 100
        backend: Literal["fbgemm", "qnnpack"] = "fbgemm"

        def to_dict(self) -> Dict[str, Any]:
            return dict(self.__dict__)

        @classmethod
        def from_dict(cls, config_dict: Dict[str, Any]) -> "QuantizationConfig":
            fields = cls.__dataclass_fields__
            return cls(
                **{key: value for key, value in config_dict.items() if key in fields}
            )


@dataclass
class WorkloadConfig:
    task_type: Literal["generic", "llm", "vision", "multimodal"] = "generic"
    model_family: str = "unknown"
    target_scale: Literal["local", "single_node", "small_cluster", "sub_100_gpus"] = (
        "local"
    )
    optimize_for: Literal["throughput", "memory", "latency", "balanced"] = "balanced"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ParallelConfig:
    data_parallel_size: int = 1
    model_parallel_size: int = 1
    tensor_parallel_size: int = 1
    tensor_parallel_mode: Literal["row", "column"] = "row"
    pipeline_parallel_size: int = 1
    pipeline_parallel_chunks: int = 1
    max_tensor_parallel_size: int = 8
    max_pipeline_parallel_size: int = 16

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class BackendConfig:
    training_backend: Literal[
        "native", "native_ddp", "fsdp", "deepspeed", "ascend_native", "auto"
    ] = "native"
    zero_optimization: bool = False
    zero_stage: int = 0
    zero_offload: bool = False
    fsdp_sharding_strategy: Literal[
        "full_shard", "shard_grad_op", "no_shard", "hybrid_shard"
    ] = "full_shard"
    fsdp_cpu_offload: bool = False
    fsdp_auto_wrap: bool = False
    fsdp_min_num_params: int = 100000000
    fsdp_state_dict_type: Literal["full", "sharded", "local"] = "full"
    fsdp_use_orig_params: bool = True
    fsdp_activation_checkpointing_policy: Literal[
        "none", "transformer_auto", "size_based"
    ] = "transformer_auto"
    fsdp_checkpoint_module_classes: List[str] = field(default_factory=list)
    ddp_find_unused_parameters: bool = False
    ddp_gradient_as_bucket_view: bool = True
    ddp_static_graph: bool = False
    ddp_comm_hook: Literal["auto", "none", "fp16_compress", "bf16_compress"] = "auto"
    ddp_bucket_cap_mb: Optional[int] = None
    deepspeed_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["fsdp_checkpoint_module_classes"] = list(
            self.fsdp_checkpoint_module_classes
        )
        return data


@dataclass
class DataPipelineConfig:
    batching_strategy: Literal["sample", "length_bucket", "token_budget"] = "sample"
    max_tokens_per_batch: Optional[int] = None
    max_patch_tokens_per_batch: Optional[int] = None
    resolution_buckets: List[List[int]] = field(default_factory=list)
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    dataloader_persistent_workers: bool = True
    dataloader_drop_last: bool = False
    dataset_local_cache_dir: Optional[str] = None
    tensor_cache: bool = False
    tensor_cache_dir: Optional[str] = None
    device_prefetch: bool = False
    prefetch_device: Optional[str] = None
    cuda_prefetch: bool = False
    cuda_prefetch_device: Optional[str] = None
    tuner_dataloader_wait_threshold_ms: float = 20.0
    preprocess_in_workers: bool = False
    pipeline_cache: bool = False
    pipeline_cache_dir: Optional[str] = None
    pipeline_cache_max_entries: int = 4096
    pipeline_cache_max_bytes: int = 20_000_000_000
    pipeline_cache_ttl_seconds: float = 0.0
    prompt_template_cache: bool = False
    prompt_template_cache_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["resolution_buckets"] = [
            list(bucket) for bucket in self.resolution_buckets
        ]
        return data


@dataclass
class TrainingRunConfig:
    strategy_memory_margin: float = 0.9
    enable_activation_checkpointing: bool = False
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.001
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    grad_clip_norm: Optional[float] = None
    log_interval: int = 100
    label_keys: List[str] = field(
        default_factory=lambda: ["labels", "label", "targets", "target", "y"]
    )
    seed: int = 42
    checkpoint_save_path: str = "./checkpoints"
    checkpoint_save_interval: int = 1000
    adapter_only_checkpoint: bool = False
    allow_world_size_change_on_resume: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["label_keys"] = list(self.label_keys)
        return data


_CONFIG_SECTION_TYPES = {
    "workload": WorkloadConfig,
    "parallel": ParallelConfig,
    "backend": BackendConfig,
    "data": DataPipelineConfig,
    "training": TrainingRunConfig,
}


def flat_config_fields_by_section() -> Dict[str, tuple[str, ...]]:
    """Return the canonical mapping from layered sections to flat config fields."""

    return {
        section: tuple(config_type().to_dict().keys())
        for section, config_type in _CONFIG_SECTION_TYPES.items()
    }


def flat_config_field_names() -> tuple[str, ...]:
    """Return every flat config field owned by a layered config section."""

    fields: list[str] = []
    for section_fields in flat_config_fields_by_section().values():
        fields.extend(section_fields)
    return tuple(fields)


def _copy_config_value(value: Any) -> Any:
    if isinstance(value, list):
        return [list(item) if isinstance(item, list) else item for item in value]
    if isinstance(value, dict):
        return dict(value)
    return value


def _section_kwargs_from_config(
    config: "ParaScaleConfig", section: str
) -> Dict[str, Any]:
    return {
        field_name: _copy_config_value(getattr(config, field_name))
        for field_name in flat_config_fields_by_section()[section]
    }


@dataclass
class LayeredParaScaleConfig:
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    data: DataPipelineConfig = field(default_factory=DataPipelineConfig)
    training: TrainingRunConfig = field(default_factory=TrainingRunConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload": self.workload.to_dict(),
            "parallel": self.parallel.to_dict(),
            "backend": self.backend.to_dict(),
            "data": self.data.to_dict(),
            "training": self.training.to_dict(),
            "quantization": self.quantization.to_dict(),
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for section in [
            self.workload,
            self.parallel,
            self.backend,
            self.data,
            self.training,
        ]:
            data.update(section.to_dict())
        data["quantization"] = self.quantization.to_dict()
        return data

    def to_config(self) -> "ParaScaleConfig":
        return ParaScaleConfig.from_dict(self.to_flat_dict())

    @classmethod
    def from_config(cls, config: "ParaScaleConfig") -> "LayeredParaScaleConfig":
        return cls(
            workload=WorkloadConfig(**_section_kwargs_from_config(config, "workload")),
            parallel=ParallelConfig(**_section_kwargs_from_config(config, "parallel")),
            backend=BackendConfig(**_section_kwargs_from_config(config, "backend")),
            data=DataPipelineConfig(**_section_kwargs_from_config(config, "data")),
            training=TrainingRunConfig(
                **_section_kwargs_from_config(config, "training")
            ),
            quantization=config.quantization,
        )


class ParaScaleConfig:
    """Flat runtime view derived from the layered ParaScale schema."""

    def __init__(self, **config_dict: Any) -> None:
        explicit_fields = set(config_dict)
        defaults = LayeredParaScaleConfig().to_flat_dict()
        quantization = config_dict.pop("quantization", defaults.pop("quantization"))
        unknown = sorted(set(config_dict) - set(defaults))
        if unknown:
            raise TypeError(
                "unexpected ParaScaleConfig fields: " + ", ".join(unknown)
            )
        values = {**defaults, **config_dict}
        for key, value in values.items():
            setattr(self, key, _copy_config_value(value))
        if isinstance(quantization, QuantizationConfig):
            self.quantization = quantization
        elif isinstance(quantization, dict):
            self.quantization = QuantizationConfig.from_dict(quantization)
        else:
            raise TypeError("quantization must be a QuantizationConfig or mapping.")
        self._explicit_fields = frozenset(explicit_fields)
        self._normalize_device_prefetch_aliases()
        self._validate()

    def _normalize_device_prefetch_aliases(self) -> None:
        self.device_prefetch = bool(self.device_prefetch or self.cuda_prefetch)
        self.cuda_prefetch = bool(self.cuda_prefetch or self.device_prefetch)
        if self.prefetch_device is None and self.cuda_prefetch_device is not None:
            self.prefetch_device = self.cuda_prefetch_device
        if self.cuda_prefetch_device is None and self.prefetch_device is not None:
            self.cuda_prefetch_device = self.prefetch_device

    def _validate(self) -> None:
        if self.task_type not in ["generic", "llm", "vision", "multimodal"]:
            raise ValueError(
                f"task_type must be generic, llm, vision, or multimodal, got {self.task_type}"
            )
        if self.target_scale not in [
            "local",
            "single_node",
            "small_cluster",
            "sub_100_gpus",
        ]:
            raise ValueError(f"unsupported target_scale: {self.target_scale}")
        if self.optimize_for not in ["throughput", "memory", "latency", "balanced"]:
            raise ValueError(f"unsupported optimize_for: {self.optimize_for}")
        if self.data_parallel_size < 1:
            raise ValueError(
                f"data_parallel_size must be >= 1, got {self.data_parallel_size}"
            )
        if self.model_parallel_size < 1:
            raise ValueError(
                f"model_parallel_size must be >= 1, got {self.model_parallel_size}"
            )
        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}"
            )
        if self.pipeline_parallel_size < 1:
            raise ValueError(
                f"pipeline_parallel_size must be >= 1, got {self.pipeline_parallel_size}"
            )
        if self.pipeline_parallel_chunks < 1:
            raise ValueError(
                f"pipeline_parallel_chunks must be >= 1, got {self.pipeline_parallel_chunks}"
            )
        if self.zero_stage not in [0, 1, 2, 3]:
            raise ValueError(f"zero_stage must be 0, 1, 2, or 3, got {self.zero_stage}")
        if self.training_backend not in [
            "native",
            "native_ddp",
            "fsdp",
            "deepspeed",
            "ascend_native",
            "auto",
        ]:
            raise ValueError(
                "training_backend must be native, native_ddp, fsdp, deepspeed, "
                f"ascend_native, or auto, got {self.training_backend}"
            )
        if self.fsdp_sharding_strategy not in [
            "full_shard",
            "shard_grad_op",
            "no_shard",
            "hybrid_shard",
        ]:
            raise ValueError(
                f"unsupported fsdp_sharding_strategy: {self.fsdp_sharding_strategy}"
            )
        if self.fsdp_min_num_params < 1:
            raise ValueError(
                f"fsdp_min_num_params must be >= 1, got {self.fsdp_min_num_params}"
            )
        if self.fsdp_state_dict_type not in ["full", "sharded", "local"]:
            raise ValueError(
                f"fsdp_state_dict_type must be full, sharded, or local, got {self.fsdp_state_dict_type}"
            )
        if self.fsdp_activation_checkpointing_policy not in [
            "none",
            "transformer_auto",
            "size_based",
        ]:
            raise ValueError(
                "fsdp_activation_checkpointing_policy must be none, "
                "transformer_auto, or size_based, got "
                f"{self.fsdp_activation_checkpointing_policy}"
            )
        if self.ddp_comm_hook not in [
            "auto",
            "none",
            "fp16_compress",
            "bf16_compress",
        ]:
            raise ValueError(
                "ddp_comm_hook must be auto, none, fp16_compress, or "
                f"bf16_compress, got {self.ddp_comm_hook}"
            )
        if self.ddp_bucket_cap_mb is not None and self.ddp_bucket_cap_mb < 1:
            raise ValueError(
                "ddp_bucket_cap_mb must be >= 1 when set, got "
                f"{self.ddp_bucket_cap_mb}"
            )
        if not 0 < self.strategy_memory_margin <= 1:
            raise ValueError(
                f"strategy_memory_margin must be in (0, 1], got {self.strategy_memory_margin}"
            )
        if self.max_tensor_parallel_size < 1:
            raise ValueError(
                f"max_tensor_parallel_size must be >= 1, got {self.max_tensor_parallel_size}"
            )
        if self.max_pipeline_parallel_size < 1:
            raise ValueError(
                f"max_pipeline_parallel_size must be >= 1, got {self.max_pipeline_parallel_size}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.precision not in ["fp32", "fp16", "bf16"]:
            raise ValueError(
                f"precision must be fp32, fp16, or bf16, got {self.precision}"
            )
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive when set, got {self.grad_clip_norm}"
            )
        if self.log_interval < 1:
            raise ValueError(f"log_interval must be >= 1, got {self.log_interval}")
        if self.batching_strategy not in ["sample", "length_bucket", "token_budget"]:
            raise ValueError(
                f"batching_strategy must be sample, length_bucket, or token_budget, got {self.batching_strategy}"
            )
        if self.max_tokens_per_batch is not None and self.max_tokens_per_batch < 1:
            raise ValueError(
                f"max_tokens_per_batch must be >= 1 when set, got {self.max_tokens_per_batch}"
            )
        if (
            self.max_patch_tokens_per_batch is not None
            and self.max_patch_tokens_per_batch < 1
        ):
            raise ValueError(
                f"max_patch_tokens_per_batch must be >= 1 when set, got {self.max_patch_tokens_per_batch}"
            )
        if self.dataloader_num_workers < 0:
            raise ValueError(
                f"dataloader_num_workers must be >= 0, got {self.dataloader_num_workers}"
            )
        if self.dataloader_prefetch_factor < 1:
            raise ValueError(
                f"dataloader_prefetch_factor must be >= 1, got {self.dataloader_prefetch_factor}"
            )
        if self.tuner_dataloader_wait_threshold_ms < 0:
            raise ValueError(
                "tuner_dataloader_wait_threshold_ms must be >= 0, got "
                f"{self.tuner_dataloader_wait_threshold_ms}"
            )
        if self.pipeline_cache_max_entries < 1:
            raise ValueError(
                "pipeline_cache_max_entries must be >= 1, got "
                f"{self.pipeline_cache_max_entries}"
            )
        if self.pipeline_cache_max_bytes < 1:
            raise ValueError(
                "pipeline_cache_max_bytes must be >= 1, got "
                f"{self.pipeline_cache_max_bytes}"
            )
        if self.pipeline_cache_ttl_seconds < 0:
            raise ValueError(
                "pipeline_cache_ttl_seconds must be >= 0, got "
                f"{self.pipeline_cache_ttl_seconds}"
            )
        if self.checkpoint_save_interval < 1:
            raise ValueError(
                f"checkpoint_save_interval must be >= 1, got {self.checkpoint_save_interval}"
            )

    def update(self, config_dict: Dict[str, Any]) -> "ParaScaleConfig":
        explicit_fields = set(getattr(self, "_explicit_fields", frozenset()))
        for key, value in config_dict.items():
            if key == "quantization":
                if isinstance(value, QuantizationConfig):
                    self.quantization = value
                elif isinstance(value, dict):
                    self.quantization = QuantizationConfig.from_dict(value)
                continue
            if hasattr(self, key):
                setattr(self, key, value)
                explicit_fields.add(key)
        self._explicit_fields = frozenset(explicit_fields)
        self._normalize_device_prefetch_aliases()
        self._validate()
        return self

    def to_layered(self) -> LayeredParaScaleConfig:
        return LayeredParaScaleConfig.from_config(self)

    def to_layered_dict(self) -> Dict[str, Any]:
        return self.to_layered().to_dict()

    def to_dict(self) -> Dict[str, Any]:
        data = {
            field_name: _copy_config_value(getattr(self, field_name))
            for field_name in flat_config_field_names()
        }
        data["resolution_buckets"] = [
            list(bucket) for bucket in self.resolution_buckets
        ]
        data["label_keys"] = list(self.label_keys)
        data["fsdp_checkpoint_module_classes"] = list(
            self.fsdp_checkpoint_module_classes
        )
        data["deepspeed_config"] = (
            dict(self.deepspeed_config) if self.deepspeed_config is not None else None
        )
        data["quantization"] = self.quantization.to_dict()
        return data

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ParaScaleConfig":
        normalized = dict(config_dict)
        if any(
            key in normalized
            for key in ["workload", "parallel", "backend", "data", "training"]
        ):
            return cls.from_layered_dict(normalized)
        quantization = normalized.get("quantization")
        if isinstance(quantization, dict):
            normalized["quantization"] = QuantizationConfig.from_dict(quantization)
        return cls(**normalized)

    @classmethod
    def from_layered_dict(cls, config_dict: Dict[str, Any]) -> "ParaScaleConfig":
        layered_keys = list(_CONFIG_SECTION_TYPES)
        flat: Dict[str, Any] = {}
        for key in layered_keys:
            section = config_dict.get(key, {})
            if isinstance(section, dict):
                flat.update(section)
        if "quantization" in config_dict:
            flat["quantization"] = config_dict["quantization"]
        for key, value in config_dict.items():
            if key not in layered_keys and key != "quantization":
                flat[key] = value
        return cls.from_dict(flat)

    def __str__(self) -> str:
        return f"ParaScaleConfig(data_parallel_size={self.data_parallel_size}, model_parallel_size={self.model_parallel_size}, tensor_parallel_size={self.tensor_parallel_size}, tensor_parallel_mode={self.tensor_parallel_mode}, pipeline_parallel_size={self.pipeline_parallel_size}, zero_optimization={self.zero_optimization}, zero_stage={self.zero_stage})"
