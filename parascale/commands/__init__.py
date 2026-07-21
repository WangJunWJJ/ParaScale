# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:56
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""CLI command helpers for ParaScale."""

from .benchmark_matrix import register_benchmark_matrix_parser
from .checkpoint import register_checkpoint_parser
from .common import emit_json, load_config_file
from .configuration import register_config_parser
from .doctor import build_doctor_payload, register_doctor_parser
from .plan import register_plan_parser
from .run import register_run_parsers
from .smoke import register_smoke_parser
from .stability import register_stability_parser
from .vision import register_vision_profile_parser


def register_command_parsers(subparsers) -> None:
    register_config_parser(subparsers)
    register_plan_parser(subparsers)
    register_doctor_parser(subparsers)
    register_smoke_parser(subparsers)
    register_run_parsers(subparsers)
    register_benchmark_matrix_parser(subparsers)
    register_stability_parser(subparsers)
    register_vision_profile_parser(subparsers)
    register_checkpoint_parser(subparsers)


__all__ = [
    "build_doctor_payload",
    "emit_json",
    "load_config_file",
    "register_command_parsers",
]
