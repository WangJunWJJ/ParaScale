# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded subprocesses, including torchrun workers in separate sessions."""

import os
import subprocess
import time
from pathlib import Path


class ProcessCleanupError(RuntimeError):
    """Process cleanup could not be confirmed; do not launch more work."""


def run_process(command, cwd, env, timeout, log):
    """Retain logs and terminate tracked descendants on timeout/interruption."""
    import psutil

    def capture(parent, tracked):
        if parent is not None:
            try:
                tracked.update({child.pid: child for child in parent.children(recursive=True)})
            except psutil.NoSuchProcess:
                pass

    def alive(child):
        try:
            return child.is_running() and child.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    def stop(process, parent, tracked):
        capture(parent, tracked)
        # Let torchrun forward termination to workers before forcing shutdown.
        if parent is not None:
            try:
                parent.terminate()
            except psutil.NoSuchProcess:
                pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            capture(parent, tracked)
        children = [child for child in tracked.values() if alive(child)]
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, remaining = psutil.wait_procs(children, timeout=3)
        for child in remaining:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        _, remaining = psutil.wait_procs([child for child in remaining if alive(child)], timeout=3)
        if any(alive(child) for child in remaining):
            raise RuntimeError("failed to terminate launched descendants")

    def cleanup(process, parent, tracked):
        try:
            stop(process, parent, tracked)
        except Exception as exc:
            raise ProcessCleanupError(f"process cleanup failed: {exc}") from exc

    started = time.monotonic()
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with Path(f"{log}.stdout.log").open("w", encoding="utf-8") as stdout, Path(f"{log}.stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=cwd, env={**os.environ, **env}, stdout=stdout, stderr=stderr, **options)
        try:
            parent = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            parent = None
        tracked, timed_out = {}, False
        try:
            while process.poll() is None:
                capture(parent, tracked)
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    cleanup(process, parent, tracked)
                    break
                try:
                    process.wait(timeout=min(0.2, remaining))
                except subprocess.TimeoutExpired:
                    pass
            if any(alive(child) for child in tracked.values()):
                cleanup(process, parent, tracked)
        except ProcessCleanupError:
            raise
        except BaseException:
            cleanup(process, parent, tracked)
            raise
    return {"returncode": process.returncode, "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - started,
            "stdout": str(Path(f"{log}.stdout.log")), "stderr": str(Path(f"{log}.stderr.log"))}
