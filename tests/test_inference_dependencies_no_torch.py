# -*- coding: utf-8 -*-
# @Time : 2026/11/22 上午7:30
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Pinned inference dependencies must remain covered by clean resolution CI."""

from pathlib import Path

import pytest
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "extra,backend",
    [("inference-int8", "gemlite==0.5.1"), ("inference-int4", "torchao==0.15.0")],
)
def test_inference_extra_pins_matching_runtime(extra, backend):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = project["project"]["optional-dependencies"][extra]
    assert "torch==2.9.1" in requirements
    assert "torchvision==0.24.1" in requirements
    assert backend in requirements
    if extra == "inference-int8":
        assert "triton==3.5.1; platform_system == 'Linux'" in requirements


@pytest.mark.parametrize("extra", ["inference-int8", "inference-int4"])
def test_ci_resolves_inference_extras_without_installed_dependencies(extra):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["extras-resolution"]
    assert extra in job["strategy"]["matrix"]["extra"]
    resolution = [
        step["run"] for step in job["steps"] if "--dry-run" in step.get("run", "")
    ]
    assert len(resolution) == 1
    assert "--ignore-installed" in resolution[0]
    assert "--no-deps" not in resolution[0]
    assert "matrix.extra" in resolution[0]
