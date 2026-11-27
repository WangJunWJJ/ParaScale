# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Pytest parses PYTEST_ADDOPTS using POSIX shlex on every platform."""

import shlex
from pathlib import PurePosixPath, PureWindowsPath

import pytest

from tests import run_tests


@pytest.mark.parametrize("root", [PureWindowsPath("E:/ParaScale data/tests"), PurePosixPath("/tmp/ParaScale data/tests")])
def test_pytest_options_preserve_absolute_path(root):
    assert shlex.split(run_tests.pytest_options(root)) == [
        f"--basetemp={(root / 'pytest').as_posix()}", "-p", "no:cacheprovider"
    ]
