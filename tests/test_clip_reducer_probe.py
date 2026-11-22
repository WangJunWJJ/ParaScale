# -*- coding: utf-8 -*-
# @Time : 2026/11/23 下午2:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Small numerical controls for the bounded DDP recovery diagnostic."""

import pytest

torch = pytest.importorskip("torch")


def test_fingerprint_covers_tensor_metadata_and_nested_values():
    from tests.benchmarks.scripts.probe_clip_reducer_recovery import fingerprint

    value = {"x": torch.arange(4), "metadata": [None, True, "clip"]}
    assert fingerprint(value) == fingerprint(value)
    assert fingerprint(value) != fingerprint({**value, "x": value["x"].reshape(2, 2)})
    assert fingerprint(value) != fingerprint({**value, "x": value["x"].float()})
    assert fingerprint(value) != fingerprint(
        {**value, "metadata": [None, False, "clip"]}
    )


def test_tensor_differences_reports_small_errors_without_hiding_them():
    from tests.benchmarks.scripts.probe_clip_reducer_recovery import tensor_differences

    left = {"weight": torch.tensor([1.0, 0.0]), "step": torch.tensor(25)}
    right = {"weight": torch.tensor([1.0, 1e-6]), "step": torch.tensor(25)}
    result = tensor_differences(left, right)
    assert result["differing_elements"] == 1
    assert result["outside_tolerance"] == 0
    assert result["max_abs"] == pytest.approx(1e-6)
    assert result["changed"][0]["name"] == "weight"


def test_tensor_differences_rejects_missing_or_invalid_tensors():
    from tests.benchmarks.scripts.probe_clip_reducer_recovery import tensor_differences

    with pytest.raises(AssertionError):
        tensor_differences({"x": torch.ones(2)}, {})
    with pytest.raises(AssertionError):
        tensor_differences({"x": torch.ones(2)}, {"x": torch.ones(1)})
    with pytest.raises(AssertionError):
        tensor_differences({"x": torch.ones(1)}, {"x": torch.tensor([float("nan")])})


@pytest.mark.parametrize("field", ["input_sha256", "rng_before", "step"])
def test_step_control_rejects_changed_input_or_rng(field):
    from tests.benchmarks.scripts.probe_clip_reducer_recovery import (
        validate_step_control,
    )

    cold = {"step": 26, "input_sha256": "a", "rng_before": "b"}
    validate_step_control(cold, dict(cold))
    with pytest.raises(AssertionError):
        validate_step_control(cold, {**cold, field: "changed"})


def test_warm_preflight_requires_tensor_evidence_and_identical_execution(tmp_path):
    import json

    from tests.benchmarks.scripts.probe_clip_reducer_recovery import (
        validate_warm_inputs,
    )

    report = {
        "initial": {"backend": "a"},
        "checkpoint_sha256": "b",
        "attention_classes": ["SDPA"],
        "sdpa": {"math": True},
        "deterministic_algorithms": False,
    }
    (tmp_path / "cold-rank0.json").write_text(json.dumps(report))
    with pytest.raises(FileNotFoundError):
        validate_warm_inputs(tmp_path, 0, report)
    for filename in ("cold-step26.pt", "cold-step27.pt", "cold-local-rank0.pt"):
        (tmp_path / filename).touch()
    assert validate_warm_inputs(tmp_path, 0, report) == report
    with pytest.raises(AssertionError):
        validate_warm_inputs(tmp_path, 0, {**report, "attention_classes": ["Eager"]})
