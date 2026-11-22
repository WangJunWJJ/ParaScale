# -*- coding: utf-8 -*-
# @Time : 2026/11/23 上午12:50
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import pytest

torch = pytest.importorskip("torch")


def test_recovery_comparison_preserves_exact_and_tolerant_gates():
    from tests.benchmarks.scripts.validate_clip_prefetch_recovery import compare_values

    value = {"tensor": torch.tensor([1.0]), "step": 50}
    assert compare_values(value, value, exact=True)["passed"]
    nearby = {"tensor": torch.tensor([1.00001]), "step": 50}
    assert compare_values(value, nearby)["passed"]
    assert not compare_values(value, nearby, exact=True)["passed"]
    assert not compare_values(value, {"tensor": torch.tensor([1.01]), "step": 50})[
        "passed"
    ]
    assert not compare_values(
        value, {"tensor": torch.tensor([float("nan")]), "step": 50}
    )["passed"]


def test_recovery_requires_exact_metadata_and_supports_rng_trees():
    from tests.benchmarks.scripts.validate_clip_prefetch_recovery import compare_values

    value = {"identity": "source", "rng": ("MT19937", [1, 2], None), "lr": 0.001}
    assert compare_values(value, value, exact=True)["passed"]
    assert not compare_values(value, {**value, "lr": 0.001001})["passed"]
    assert not compare_values(value, {**value, "identity": "other"})["passed"]


@pytest.mark.parametrize("steps", [4, 50])
@pytest.mark.parametrize(
    "bad", [None, "empty", "step", "moment", "missing_moment", "duplicate_id", "nan"]
)
def test_optimizer_budget_rejects_equal_but_incomplete_states(bad, steps):
    from tests.benchmarks.scripts.validate_clip_prefetch_recovery import (
        validate_optimizer_budget,
    )

    state = {
        "model_state_dict": {"weight": torch.ones(2)},
        "optimizer_state_dict": {
            "param_groups": [{"params": [0]}],
            "state": {
                0: {
                    "step": torch.tensor(float(steps)),
                    "exp_avg": torch.ones(2),
                    "exp_avg_sq": torch.ones(2),
                }
            },
        },
    }
    if bad == "empty":
        state["optimizer_state_dict"]["state"].clear()
    elif bad == "step":
        state["optimizer_state_dict"]["state"][0]["step"] = torch.tensor(steps - 0.001)
    elif bad == "moment":
        state["optimizer_state_dict"]["state"][0]["exp_avg"] = torch.ones(1)
    elif bad == "missing_moment":
        del state["optimizer_state_dict"]["state"][0]["exp_avg_sq"]
    elif bad == "duplicate_id":
        state["model_state_dict"]["bias"] = torch.ones(2)
        state["optimizer_state_dict"]["param_groups"][0]["params"] = [0, 0]
    elif bad == "nan":
        state["optimizer_state_dict"]["state"][0]["exp_avg"][0] = float("nan")
    assert validate_optimizer_budget(state, steps=steps)["passed"] == (bad is None)


@pytest.mark.parametrize("position", [1600, 1599])
def test_runtime_budget_requires_expected_consumption(position):
    from tests.benchmarks.scripts.validate_clip_prefetch_recovery import (
        validate_runtime_budget,
    )

    runtime = {
        "version": 1,
        "rank": 2,
        "world_size": 4,
        "data_state": {
            "consumed_micro_batches": 50,
            "resume_mode": "state_dict",
            "target": "dataloader",
            "state": {
                "position": position,
                "identity": {"rank": 2, "world_size": 4, "batch_size": 32},
            },
        },
    }
    assert validate_runtime_budget(runtime, 2)["passed"] == (position == 1600)


@pytest.mark.parametrize("bad", [None, "split", "model", "extra", "hash"])
def test_input_binding_rejects_wrong_training_sources(tmp_path, monkeypatch, bad):
    import hashlib
    import json

    from tests.benchmarks.scripts import validate_clip_prefetch_recovery as gate

    data = tmp_path / "data"
    (data / "train").mkdir(parents=True)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.bin").write_bytes(b"weights")
    (data / "train/00000.tar").write_bytes(b"shard")
    manifest = {
        "splits": {"train": [{"shard": "train/00000.tar"}]},
        "shards_sha256": {"train/00000.tar": hashlib.sha256(b"shard").hexdigest()},
    }
    (data / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(gate, "validate_manifest", lambda value: None)
    config = {
        "model": {"type": "hf_clip", "pretrained_model_name_or_path": str(model)},
        "data": {
            "type": "datacomp_wds",
            "streaming": True,
            "num_workers": 0,
            "data_dir": str(data / "train"),
        },
    }
    if bad == "split":
        config["data"]["data_dir"] = str(data / "validation")
    elif bad == "model":
        config["model"]["pretrained_model_name_or_path"] = str(tmp_path / "other")
    elif bad == "extra":
        (data / "train/extra.tar").write_bytes(b"extra")
    elif bad == "hash":
        (data / "train/00000.tar").write_bytes(b"changed")
    if bad:
        with pytest.raises(ValueError):
            gate.validate_input_binding(config, data, model)
    else:
        assert gate.validate_input_binding(config, data, model)["model_assets"]
