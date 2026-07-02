# -*- coding: utf-8 -*-
# @Time : 2026/7/2 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Strict doctor behavior without hardware runtime dependencies."""

import importlib
import json

from parascale.cli import main


def _evaluate(payload, requirements):
    diagnostics = importlib.import_module("parascale.commands.diagnostics")
    return diagnostics.evaluate_diagnostics(payload, requirements)


def _payload(**overrides):
    payload = {
        "python": {"version": "3.12.0"},
        "dependencies": {
            "torch": False,
            "torch_npu": False,
            "deepspeed": False,
            "yaml": True,
        },
        "torch_runtime": {"available": False},
        "distributed_runtime": {"available": False},
        "ascend_runtime": {"available": False},
    }
    payload.update(overrides)
    return payload


def test_strict_core_and_torch_report_missing_torch():
    report = _evaluate(_payload(), ["core", "torch"])

    assert report.ok is False
    assert report.failed == ("torch",)
    assert report.to_dict()["checks"]["torch"]["required"] is True


def test_cuda_requirement_needs_visible_cuda_device():
    report = _evaluate(
        _payload(
            dependencies={"torch": True, "yaml": True, "deepspeed": False},
            torch_runtime={"available": True, "cuda_available": False},
            distributed_runtime={"available": True},
        ),
        ["cuda"],
    )

    assert report.ok is False
    assert report.failed == ("cuda",)


def test_npu_requirement_needs_torch_npu_and_visible_device():
    report = _evaluate(
        _payload(
            dependencies={"torch": True, "torch_npu": True, "yaml": True},
            torch_runtime={"available": True},
            distributed_runtime={"available": True},
            ascend_runtime={"available": True, "device_count": 8},
        ),
        ["npu"],
    )

    assert report.ok is True
    assert report.failed == ()


def test_requirements_preserve_order_and_remove_duplicates():
    report = _evaluate(_payload(), ["torch", "core", "torch"])

    assert report.requirements == ("torch", "core")
    assert tuple(report.to_dict()["checks"]) == ("torch", "core")


def test_torch_requirement_rejects_runtime_import_error():
    report = _evaluate(
        _payload(
            dependencies={"torch": True, "yaml": True},
            torch_runtime={"available": True, "error": "missing shared library"},
        ),
        ["torch"],
    )

    assert report.ok is False


def test_deepspeed_requirement_needs_successful_import():
    report = _evaluate(
        _payload(
            dependencies={"deepspeed": True, "yaml": True},
            deepspeed_runtime={"available": False, "error": "version mismatch"},
        ),
        ["deepspeed"],
    )

    assert report.ok is False


def test_cli_strict_doctor_writes_report_and_returns_two(monkeypatch, tmp_path):
    output = tmp_path / "doctor.json"
    monkeypatch.setattr(
        "parascale.commands.doctor.build_doctor_payload",
        lambda: _payload(),
    )

    assert main(["doctor", "--strict", "--output", str(output)]) == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["requirements"] == ["core", "torch"]
    assert payload["checks"]["torch"]["ok"] is False


def test_cli_diagnostic_only_doctor_remains_successful(monkeypatch, tmp_path):
    output = tmp_path / "doctor.json"
    monkeypatch.setattr(
        "parascale.commands.doctor.build_doctor_payload",
        lambda: _payload(),
    )

    assert main(["doctor", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["requirements"] == []
