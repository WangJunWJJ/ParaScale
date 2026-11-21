# -*- coding: utf-8 -*-
# @Time : 2026/11/21 下午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""CI must run Torch-dependent contract tests, not only collect-free tests."""

from pathlib import Path

import yaml


def test_ci_pins_runtime_and_runs_complete_cpu_suite():
    workflow = yaml.safe_load(
        (Path(__file__).parents[1] / ".github/workflows/ci.yml").read_text(
            encoding="utf-8"
        )
    )
    job = workflow["jobs"]["runtime-tests"]
    commands = [step.get("run", "") for step in job["steps"]]
    assert any(
        "torch==2.9.1" in command and "/whl/cpu" in command for command in commands
    )
    assert any("transformers==4.51.3" in command for command in commands)
    assert "python tests/run_tests.py" in commands
    assert job["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert (
        "parallel"
        in workflow["jobs"]["extras-resolution"]["strategy"]["matrix"]["extra"]
    )
