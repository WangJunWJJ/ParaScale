# -*- coding: utf-8 -*-
# @Time : 2026/6/16 下午4:19
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import pytest


def test_distributed_topology_rejects_world_size_mismatch(monkeypatch):
    from parascale.runtime.lifecycle import validate_distributed_topology

    monkeypatch.setenv("WORLD_SIZE", "4")

    with pytest.raises(ValueError, match="WORLD_SIZE=4.*nnodes=2.*nproc_per_node=1"):
        validate_distributed_topology(
            {"distributed": {"nnodes": 2, "nproc_per_node": 1}}
        )

def test_distributed_component_config_uses_rank_specific_data_seed():
    from parascale.runtime.train_runner import _rank_component_config

    config = {"training": {"seed": 42}}

    ranked = _rank_component_config(config, rank=3, distributed=True)

    assert ranked["training"]["seed"] == 45
    assert config["training"]["seed"] == 42

def test_accumulated_pipeline_cache_hit_is_normalized_without_torch():
    from parascale.runtime.training.metrics import merge_pipeline_profiles

    profiles = [
        {"cache_hit": 1.0, "prompt_template_ms": 0.1},
        {"cache_hit": 1.0, "prompt_template_ms": 0.2},
        {"cache_hit": 1.0, "prompt_template_ms": 0.3},
        {"cache_hit": 1.0, "prompt_template_ms": 0.4},
    ]

    merged = merge_pipeline_profiles(profiles)

    assert merged["cache_hit"] == 1.0
    assert merged["cache_hit_count"] == 4.0
    assert merged["cache_sample_count"] == 4.0
    assert merged["prompt_template_ms"] == 1.0

def test_accumulated_pipeline_cache_hit_supports_partial_hits_without_torch():
    from parascale.runtime.training.metrics import merge_pipeline_profiles

    profiles = [
        {"cache_hit": 1.0},
        {"cache_hit": 0.0},
        {"cache_hit": 0.5},
        {"cache_hit": 1.0},
    ]

    merged = merge_pipeline_profiles(profiles)

    assert merged["cache_hit"] == 0.625
    assert merged["cache_hit_count"] == 2.5
    assert merged["cache_sample_count"] == 4.0

def test_multinode_capability_level_is_marked_as_smoke_without_torch():
    from parascale.runtime.train_runner import _capability_level_for_scope

    config_data = {
        "hardware_profile": {
            "world_size": 2,
            "gpus_per_node": 1,
            "num_nodes": 2,
        }
    }

    capability = _capability_level_for_scope(
        "local_native_clip_contrastive_datacomp_wds", config_data
    )

    assert capability == "multi_node_smoke"
