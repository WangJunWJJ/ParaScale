# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午7:45
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Failure classification and post-interruption checkpoint validation."""

import json
import os
import sys

import pytest

from parascale.checkpoint import CheckpointManager, CheckpointManifest
from parascale.commands import launcher


@pytest.mark.parametrize(
    "message,expected",
    [
        ("ncclUnhandledCudaError: unhandled cuda error", False),
        (
            "ncclUnhandledCudaError: CUDA error: an illegal memory access was encountered",
            False,
        ),
        ("ncclUnhandledCudaError: CUDA error: out of memory", True),
        ("torch.OutOfMemoryError: CUDA out of memory", True),
    ],
)
def test_generic_nccl_cuda_error_is_not_evidence_for_oom_retry(message, expected):
    result = {"returncode": 1, "log_tail": message}
    assert launcher.matrix_result_is_oom(result) is expected
    failure = launcher.classify_launcher_failure(message, returncode=1)
    assert (failure["failure_type"] == "oom") is expected


def test_launcher_reports_checkpoint_integrity_failure_after_kill(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    manager = CheckpointManager(str(checkpoint_root))
    payload = manager.payload_path(1, "state.pt")
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"good")
    manager.write_manifest(CheckpointManifest(step=1, files=[{"path": "state.pt"}]))
    payload.write_bytes(b"evil")
    assert launcher._checkpoint_is_complete(checkpoint_root, 1)
    assert not launcher._checkpoint_is_valid(checkpoint_root, 1)
    result = launcher.run_matrix_command_until_checkpoint(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        env=dict(os.environ),
        backend="native_ddp",
        run_id="invalid_checkpoint",
        error_path=tmp_path / "error.json",
        log_path=tmp_path / "output.log",
        checkpoint_root=checkpoint_root,
        checkpoint_step=1,
        timeout_seconds=5.0,
    )
    assert result["status"] == "error"
    assert result["checkpoint_ok"] is False
    assert result["returncode"] != 0
    assert result["failure_type"] == "checkpoint_failure"
    assert "validation" in result["error"]
    assert (
        json.loads((tmp_path / "error.json").read_text())["failure_type"]
        == "checkpoint_failure"
    )
