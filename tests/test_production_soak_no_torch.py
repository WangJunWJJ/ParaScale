# -*- coding: utf-8 -*-
# @Time : 2026/11/20 下午3:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.benchmarks.scripts import production_soak
from tests.benchmarks.scripts.production_soak import (
    check_progress,
    check_training_record,
    hf_state_from_training,
)


def test_soak_maps_ddp_and_native_weights_without_accepting_mixed_keys():
    for prefix in ("module.wrapped_model.", "wrapped_model."):
        assert hf_state_from_training({prefix + "weight": 1}) == {"weight": 1}
    with pytest.raises(ValueError):
        hf_state_from_training({"weight": 1, "module.wrapped_model.weight": 2})


def test_soak_rejects_nonfinite_loss_and_unbounded_history():
    check_training_record({"loss": 0.1}, 1000, 1000)
    with pytest.raises(ValueError):
        check_training_record({"loss": float("nan")}, 1, 1000)
    with pytest.raises(ValueError):
        check_training_record({"loss": 0.1}, 1001, 1000)


def test_soak_watchdog_requires_recent_progress_and_disk_headroom():
    check_progress({"time": 100}, 200, 30 * 1024**3)
    with pytest.raises(RuntimeError):
        check_progress({"time": 100}, 1000, 30 * 1024**3)
    with pytest.raises(RuntimeError):
        check_progress({"time": 100}, 200, 1024**3)


def test_soak_rejects_worker_exit_at_stop_boundary():
    with pytest.raises(RuntimeError, match="Unexpected worker exit"):
        production_soak.require_running([SimpleNamespace(poll=lambda: 1)])


@pytest.mark.parametrize("racing_exit", [False, True])
def test_soak_cleans_groups_even_when_leader_already_exited(monkeypatch, racing_exit):
    calls = []

    def killpg(pid, sig):
        calls.append((pid, sig))
        if racing_exit:
            raise ProcessLookupError()

    monkeypatch.setattr(production_soak.os, "killpg", killpg, raising=False)
    monkeypatch.setattr(production_soak.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(production_soak, "group_alive", lambda pid: False)
    process = SimpleNamespace(pid=123, poll=lambda: 0, wait=lambda timeout: 0)
    production_soak.stop_process(process)
    assert len(calls) == 2
    production_soak.stop_process(process)
    assert len(calls) == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_soak_kills_surviving_child_after_leader_exit():
    child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"
    parent = f'import subprocess,sys; subprocess.Popen([sys.executable,"-c",{child!r}]); print("ready",flush=True)'
    process = subprocess.Popen(
        [sys.executable, "-c", parent],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        process.wait(timeout=10)
        production_soak.stop_process(process)
        assert not production_soak.group_alive(process.pid)
    finally:
        production_soak.stop_process(process)
        process.stdout.close()
