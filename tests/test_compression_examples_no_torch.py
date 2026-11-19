# -*- coding: utf-8 -*-
# @Time : 2026/11/19 下午8:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Compression example launchers reuse validators without importing torch."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("example_005_llama_ptq", "validate_llama_quantization.py", "--mode ptq"),
    ("example_006_llama_qat", "validate_llama_quantization.py", "--mode qat"),
    ("example_007_llama_distillation", "validate_llama_distillation.py", "--teacher-model"),
    ("example_008_yolo_distillation", "validate_detection_distillation.py", "--image-dir"),
]


def _bash():
    if os.name == "nt":
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        if candidate.is_file():
            return str(candidate)
        pytest.skip("Git Bash is required for Windows shell launcher tests")
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("Bash is unavailable")
    return executable


@pytest.mark.parametrize("name,validator,contract", CASES)
def test_compression_example_layout(name, validator, contract):
    directory = ROOT / "examples/gpu" / name
    script = (directory / "run.sh").read_text(encoding="utf-8")
    readme = (directory / "README.md").read_text(encoding="utf-8")
    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert f"tests/benchmarks/scripts/{validator}" in script
    assert '"$@"' in script
    assert contract in script or contract in readme
    assert "OUTPUT_DIR" in script and "DEVICE" in script
    assert "report.json" in readme or "summary.json" in readme
    assert "--tiny" not in script
    assert not (directory / "runs").exists()
    assert not (directory / "config.json").exists()


@pytest.mark.parametrize("name,validator,contract", CASES)
def test_compression_launcher_help_from_other_directory(tmp_path, name, validator, contract):
    script = ROOT / "examples/gpu" / name / "run.sh"
    result = subprocess.run(
        [_bash(), script.as_posix(), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHON": Path(sys.executable).as_posix()},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--device" in result.stdout
    assert "--output" in result.stdout
    assert ("--model" if "quantization" in validator else contract) in result.stdout


@pytest.mark.parametrize("name,validator,contract", CASES)
def test_compression_launcher_requires_explicit_inputs(tmp_path, name, validator, contract):
    script = ROOT / "examples/gpu" / name / "run.sh"
    output = tmp_path / "output with spaces"
    result = subprocess.run(
        [_bash(), script.as_posix()],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHON": Path(sys.executable).as_posix(),
            "OUTPUT_DIR": str(output),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert "required" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("name", [case[0] for case in CASES[:2]])
def test_quantization_example_preserves_existing_results(tmp_path, name):
    output = tmp_path / "existing results"
    output.mkdir()
    report = output / "report.json"
    report.write_text('{"status": "passed"}', encoding="utf-8")
    checkpoint = output / "model.pt"
    checkpoint.write_bytes(b"existing-checkpoint")
    result = subprocess.run(
        [
            _bash(),
            (ROOT / "examples/gpu" / name / "run.sh").as_posix(),
            "--tiny",
            "--output",
            output.as_posix(),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHON": Path(sys.executable).as_posix()},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert "empty" in result.stderr
    assert report.read_text(encoding="utf-8") == '{"status": "passed"}'
    assert checkpoint.read_bytes() == b"existing-checkpoint"
