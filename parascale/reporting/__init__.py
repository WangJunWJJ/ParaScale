# -*- coding: utf-8 -*-
# @Time : 2026/6/23 下午5:26
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Reporting namespace for benchmark and tuner outputs."""

from .benchmark import BenchmarkResult
from .matrix import build_report as build_backend_matrix_report
from .matrix import write_markdown as write_backend_matrix_markdown

__all__ = [
    "BenchmarkResult",
    "build_backend_matrix_report",
    "write_backend_matrix_markdown",
]
