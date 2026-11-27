# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Pure native-parallel planning and resolver provenance regressions."""

import copy
import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


def settings():
    return {
        "parascale": {"backend": {"training_backend": "native_parallel"}},
        "training": {
            "workload": "causal_lm",
            "max_steps": 3,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
        },
        "model": {"pretrained_model_name_or_path": "/missing/local/model"},
        "data": {"type": "token_corpus", "path": "/missing/local/corpus"},
    }


def build(data):
    from parascale.runtime.stage_plan import build_stage_training_plan

    return build_stage_training_plan(data)


def test_evaluation_plan_is_optional_explicit_and_roundtrips():
    data = settings()
    assert build(data).evaluation is None
    data["evaluation"] = {"data_path": "/local/validation", "max_batches": 3}
    plan = build(data)
    assert plan.evaluation.data_path == "/local/validation"
    assert plan.to_dict()["evaluation"] == data["evaluation"]
    assert build(plan.config_data).evaluation == plan.evaluation


@pytest.mark.parametrize("hook", [None, "auto", "none", "bf16_compress"])
@pytest.mark.parametrize("dp", [2, 4])
def test_ddp_policy_is_explicit_and_roundtrips(hook, dp):
    data = settings()
    data["parascale"]["parallel"] = {"data_parallel_size": dp}
    if hook is not None:
        data["parascale"]["backend"]["ddp_comm_hook"] = hook
    original = copy.deepcopy(data)
    plan = build(data)
    expected = "bf16_compress" if hook == "bf16_compress" else "none"
    assert plan.ddp_comm_hook == plan.to_dict()["ddp_comm_hook"] == expected
    assert build(plan.config_data).ddp_comm_hook == expected
    assert data == original
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.ddp_comm_hook = "changed"


@pytest.mark.parametrize("dp,tp,pp", [(1, 1, 1), (1, 2, 1), (2, 2, 1), (2, 1, 2)])
def test_compressed_stage_policy_rejects_unqualified_layouts(dp, tp, pp):
    data = settings()
    data["parascale"]["parallel"] = {
        "data_parallel_size": dp,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
    }
    data["parascale"]["backend"]["ddp_comm_hook"] = "bf16_compress"
    with pytest.raises(ValueError, match="pure DP"):
        build(data)


def test_stage_policy_rejects_fp16_compression():
    data = settings()
    data["parascale"]["parallel"] = {"data_parallel_size": 2}
    data["parascale"]["backend"]["ddp_comm_hook"] = "fp16_compress"
    with pytest.raises(ValueError, match="bf16_compress"):
        build(data)


@pytest.mark.parametrize(
    "dp,tp,pp",
    [
        (1, 1, 1),
        (2, 1, 1),
        (4, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
        (2, 2, 1),
        (2, 1, 2),
        (1, 2, 2),
    ],
)
def test_supported_layouts(dp, tp, pp):
    data = settings()
    original = copy.deepcopy(data)
    data["parascale"]["parallel"] = {
        "data_parallel_size": dp,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
    }
    plan = build(data)
    payload = plan.to_dict()
    assert plan.topology.world_size == dp * tp * pp
    assert payload["world_size"] == dp * tp * pp
    assert payload["global_batch_size"] == 8 * dp
    assert plan.config.checkpoint_format == payload["checkpoint_format"] == "rank"
    assert plan.workload.batch_size == plan.config.batch_size == 2
    assert plan.timeout_seconds == 180
    assert plan.config.pipeline_schedule == "gpipe"
    assert build(plan.config_data).to_dict() == payload
    assert json.loads(json.dumps(payload)) == payload
    assert data["training"] == original["training"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.max_steps = 10


@pytest.mark.parametrize("dp,pp", [(2, 1), (4, 1), (2, 2)])
def test_dcp_checkpoint_format_in_resolved_plan(dp, pp):
    data = settings()
    data["parascale"]["parallel"] = {
        "data_parallel_size": dp,
        "pipeline_parallel_size": pp,
    }
    data["training"]["checkpoint_format"] = "dcp"

    plan = build(data)

    assert plan.config.checkpoint_format == plan.to_dict()["checkpoint_format"] == "dcp"
    assert build(plan.config_data).to_dict() == plan.to_dict()


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_parallel_size", 1),
        ("pipeline_parallel_chunks", 1),
        ("fsdp_use_orig_params", True),
        ("fsdp_cpu_offload", False),
        ("ddp_gradient_as_bucket_view", True),
        ("zero_stage", 0),
        ("zero_offload", False),
        ("enable_activation_checkpointing", True),
        ("cuda_prefetch", True),
        ("device_prefetch", True),
        ("grad_clip_norm", 1.0),
        ("allow_world_size_change_on_resume", True),
        ("precision", "fp16"),
    ],
)
def test_reject_explicit_unsupported_and_normalized(key, value):
    from parascale.configuration.resolver import normalize_execution_config

    data = settings()
    data["parascale"] = {"training_backend": "native_parallel", key: value}
    normalized, _ = normalize_execution_config(data)
    for candidate in (data, normalized):
        with pytest.raises(ValueError):
            build(candidate)


@pytest.mark.parametrize(
    "key,minimum",
    [
        ("max_steps", 1),
        ("checkpoint_keep_last", 2),
        ("checkpoint_save_interval", 1),
        ("batch_size", 1),
        ("gradient_accumulation_steps", 1),
    ],
)
@pytest.mark.parametrize("bad", [True, 1.5, "2", None, 0])
def test_strict_training_budgets(key, minimum, bad):
    data = settings()
    data["training"][key] = bad
    with pytest.raises((ValueError, TypeError)):
        build(data)


@pytest.mark.parametrize("bad", [True, 0, -1, 1.5, "180", None])
def test_strict_timeout(bad):
    data = settings()
    data["runtime"] = {"distributed_timeout_seconds": bad}
    with pytest.raises(ValueError, match="distributed_timeout_seconds"):
        build(data)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("training", "workload", "tiny"),
        ("runtime", "device", "cpu"),
        ("runtime", "device", "npu"),
        ("model", "activation_checkpointing", True),
        ("model", "pretrained_model_name_or_path", 42),
        ("model", "pretrained_model_name_or_path", "https://example.com/model"),
        ("data", "path", ["corpus"]),
        ("data", "path", "s3://bucket/data"),
        ("data", "type", "text"),
        ("distillation", "enabled", True),
        ("deepspeed_config", "train_batch_size", 8),
    ],
)
def test_unsupported_workload_and_runtime(section, key, value):
    data = settings()
    data.setdefault(section, {})[key] = value
    with pytest.raises(ValueError):
        build(data)


def test_missing_step_budget_and_quantization():
    data = settings()
    del data["training"]["max_steps"]
    with pytest.raises(ValueError, match="max_steps"):
        build(data)
    data = settings()
    data["parascale"]["quantization"] = {"enabled": True}
    with pytest.raises(ValueError, match="quantization|QAT"):
        build(data)


def test_aliases_and_specs():
    from parascale.runtime.stage_plan import is_native_parallel

    data = settings()
    data.pop("parascale")
    data["training"].update(backend="native_parallel", lr=0.02)
    data["data"]["batch_size"] = 3
    data["optimizer"] = {"type": "adamw", "lr": 0.03}
    data["training"]["scheduler"] = {"type": "linear", "total_steps": 20}
    assert is_native_parallel(data)
    assert not is_native_parallel({})
    plan = build(data)
    assert plan.optimizer_spec.lr == 0.03
    assert plan.scheduler_spec.total_steps == 20
    assert plan.workload.batch_size == 3
    assert plan.to_dict()["global_batch_size"] == 12


def test_torch_import_is_blocked_in_fresh_process():
    script = """
import importlib.abc, json, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("Torch import attempted")
sys.meta_path.insert(0, BlockTorch())
from parascale.runtime.stage_plan import build_stage_training_plan
plan = build_stage_training_plan(json.loads(sys.argv[1]))
json.dumps(plan.to_dict())
assert "torch" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(settings())],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_json_safe_tuple_optimizer_input():
    data = settings()
    data["optimizer"] = {"betas": (0.8, 0.95)}
    payload = build(data).to_dict()
    assert json.loads(json.dumps(payload)) == payload


def test_override_provenance_and_disabled_features():
    from parascale.configuration.resolver import normalize_execution_config

    data = settings()
    data["training"]["enable_activation_checkpointing"] = True
    normalized, _ = normalize_execution_config(
        data, cli_overrides={"training.enable_activation_checkpointing": False}
    )
    with pytest.raises(ValueError, match="activation_checkpointing"):
        build(normalized)
    data = settings()
    data["training"].update(
        enable_activation_checkpointing=False, allow_world_size_change_on_resume=False
    )
    build(data)


@pytest.mark.parametrize(
    "schedule,pp,microbatches,valid",
    [
        ("1f1b", 2, 2, True),
        ("1f1b", 1, 2, False),
        ("1f1b", 2, 1, False),
        ("invalid", 2, 2, False),
    ],
)
def test_schedule_constraints(schedule, pp, microbatches, valid):
    data = settings()
    data["parascale"]["parallel"] = {
        "pipeline_parallel_size": pp,
        "pipeline_schedule": schedule,
    }
    data["training"]["gradient_accumulation_steps"] = microbatches
    if valid:
        assert build(data).topology.pipeline_schedule == schedule
    else:
        with pytest.raises(ValueError):
            build(data)


@pytest.mark.parametrize(
    "parallel",
    [
        {"data_parallel_size": 3},
        {"tensor_parallel_size": 4},
        {
            "data_parallel_size": 2,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
        },
        {"data_parallel_size": True},
        {"tensor_parallel_size": 2.0},
    ],
)
def test_unqualified_topologies(parallel):
    data = settings()
    data["parascale"]["parallel"] = parallel
    with pytest.raises(ValueError):
        build(data)


def test_retention_minimum():
    data = settings()
    data["training"]["checkpoint_keep_last"] = 1
    with pytest.raises(ValueError):
        build(data)


def test_backend_selection_uses_resolver_overrides():
    from parascale.runtime.stage_plan import is_native_parallel

    data = settings()
    data["_resolution"] = {"cli_overrides": {"backend.training_backend": "native"}}
    assert not is_native_parallel(data)
    with pytest.raises(ValueError, match="training_backend"):
        build(data)


@pytest.mark.parametrize(
    "relative", ["parascale/runtime/stage_plan.py", "tests/test_stage_plan_no_torch.py"]
)
def test_python_header_timestamp(relative):
    lines = (
        (Path(__file__).resolve().parents[1] / relative)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert re.fullmatch(
        r"# @Time : \d{4}/\d{1,2}/\d{1,2} (上午|下午)\d{1,2}:\d{2}", lines[1]
    )


def test_zero_metrics_history_is_supported_and_roundtrips():
    data = settings()
    data["training"]["metrics_history_limit"] = 0
    plan = build(data)
    assert plan.metrics_history_limit == 0
    assert build(plan.config_data).metrics_history_limit == 0


@pytest.mark.parametrize("bad", [True, -1, 1.5, "0", None])
def test_metrics_history_rejects_invalid_budget(bad):
    data = settings()
    data["training"]["metrics_history_limit"] = bad
    with pytest.raises((ValueError, TypeError)):
        build(data)


@pytest.mark.parametrize("optimizer", ["four_bit_adamw", "four_bit_sgd", "adamw_4bit"])
@pytest.mark.parametrize("tp", [1, 2])
def test_four_bit_optimizers_not_qualified(optimizer, tp):
    data = settings()
    data["optimizer"] = {"type": optimizer}
    data["parascale"]["parallel"] = {"tensor_parallel_size": tp}
    with pytest.raises(ValueError, match="optimizer"):
        build(data)


def test_workload_batch_size_controls_global_batch():
    data = settings()
    data["parascale"]["training"] = {"batch_size": 5}
    data["parascale"]["parallel"] = {"data_parallel_size": 2, "tensor_parallel_size": 2}
    data["data"]["batch_size"] = 3
    plan = build(data)
    assert plan.workload.batch_size == plan.config.batch_size == 3
    assert plan.to_dict()["global_batch_size"] == 3 * 4 * 2
    assert build(plan.config_data).to_dict() == plan.to_dict()


@pytest.mark.parametrize("accelerator", ["cpu", "npu", "gpu", "mps", None, True])
def test_reject_unsupported_accelerator(accelerator):
    data = settings()
    data["runtime"] = {"device": "auto", "accelerator": accelerator}
    with pytest.raises(ValueError, match="runtime.accelerator"):
        build(data)


@pytest.mark.parametrize("accelerator", ["auto", "cuda"])
def test_supported_accelerators(accelerator):
    data = settings()
    data["runtime"] = {"accelerator": accelerator}
    assert build(data).config_data["runtime"]["accelerator"] == accelerator


@pytest.mark.parametrize("section", ["launch", "distributed"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("nnodes", 2),
        ("nnodes", True),
        ("nnodes", "1"),
        ("nproc_per_node", 1),
        ("nproc_per_node", 2.0),
        ("world_size", 4),
        ("node_rank", 1),
        ("node_rank", False),
    ],
)
def test_explicit_launch_must_match_topology(section, key, value):
    data = settings()
    data["parascale"]["parallel"] = {"data_parallel_size": 2}
    data[section] = {key: value}
    with pytest.raises(ValueError, match=f"{section}.{key}"):
        build(data)


def test_matching_single_node_launch_roundtrip():
    data = settings()
    data["parascale"]["parallel"] = {"data_parallel_size": 2}
    for section in ("launch", "distributed"):
        data[section] = {
            "nnodes": 1,
            "nproc_per_node": 2,
            "world_size": 2,
            "node_rank": 0,
        }
    plan = build(data)
    assert build(plan.config_data).to_dict() == plan.to_dict()


@pytest.mark.parametrize(
    "alias,value,canonical",
    [
        ("checkpoint_dir", "ignored/path", "checkpoint_save_path"),
        ("checkpoint_interval", 1, "checkpoint_save_interval"),
        ("skip_final_checkpoint", True, "final checkpoint"),
        ("skip_final_checkpoint", False, "final checkpoint"),
    ],
)
def test_reject_legacy_checkpoint_aliases(alias, value, canonical):
    from parascale.configuration.resolver import normalize_execution_config

    data = settings()
    data["training"][alias] = value
    normalized, _ = normalize_execution_config(data)
    for candidate in (data, normalized):
        with pytest.raises(ValueError, match=canonical):
            build(candidate)


@pytest.mark.parametrize("section", ["data", "training", "parascale.data"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("batching_strategy", "token_budget"),
        ("batching_strategy", "length_bucket"),
        ("max_tokens_per_batch", 1024),
        ("max_patch_tokens_per_batch", 1024),
        ("resolution_buckets", [[224, 224]]),
        ("dataloader_drop_last", True),
    ],
)
def test_reject_nonfixed_batch_requests_with_provenance(section, key, value):
    from parascale.configuration.resolver import normalize_execution_config

    data = settings()
    target = (
        data["parascale"].setdefault("data", {})
        if section == "parascale.data"
        else data[section]
    )
    target[key] = value
    normalized, _ = normalize_execution_config(data)
    for candidate in (data, normalized):
        with pytest.raises(ValueError, match=key):
            build(candidate)


def test_batch_policy_provenance_survives_override():
    from parascale.configuration.resolver import normalize_execution_config

    data = settings()
    data["parascale"]["data"] = {"batching_strategy": "token_budget"}
    normalized, _ = normalize_execution_config(
        data, cli_overrides={"data.batching_strategy": "sample"}
    )
    with pytest.raises(ValueError, match="batching_strategy"):
        build(normalized)


def test_canonical_precision_owner():
    data = settings()
    data["parascale"]["training"] = {"precision": "bf16"}
    assert build(data).config.precision == "bf16"


@pytest.mark.parametrize("section", ["data", "training"])
def test_reject_drop_last_alias(section):
    data = settings()
    data[section]["drop_last"] = True
    with pytest.raises(ValueError, match="drop_last"):
        build(data)


@pytest.mark.parametrize("optimizer", [{"type": "adamw", "lr": 0.05}, None])
def test_reject_misplaced_training_optimizer(optimizer):
    from parascale.configuration.resolver import normalize_execution_config

    data = settings()
    data["training"]["optimizer"] = optimizer
    normalized, _ = normalize_execution_config(data)
    for candidate in (data, normalized):
        with pytest.raises(ValueError, match="training.optimizer.*root.*optimizer"):
            build(candidate)
