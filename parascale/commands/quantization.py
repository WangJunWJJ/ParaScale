# -*- coding: utf-8 -*-
# @Time : 2026/11/13 上午10:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""PTQ export for the existing workload/configuration boundary."""

from __future__ import annotations

import argparse
from typing import Any

from parascale.commands.common import emit_json, load_config_file
from parascale.config import ParaScaleConfig
from parascale.runtime.lifecycle import is_distributed_launch
from parascale.workloads import build_training_components


def register_quantization_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "quantize", help="Calibrate and export a PTQ workload."
    )
    parser.add_argument("--config", required=True, help="Workload JSON/YAML config.")
    parser.add_argument("--output", required=True, help="Quantized checkpoint path.")
    parser.set_defaults(func=cmd_quantize)


def run_ptq_from_config(config_data: dict[str, Any], output: str) -> dict[str, Any]:
    config = ParaScaleConfig.from_dict(config_data.get("parascale", {}))
    if not config.quantization.enabled or config.quantization.mode != "ptq":
        raise ValueError("quantize requires quantization.enabled=true and mode=ptq.")
    if config.training_backend != "native" or is_distributed_launch():
        raise ValueError(
            "PTQ export currently requires a single-process native backend."
        )
    from parascale.quantization import PostTrainingQuantization

    model, _, batches, _ = build_training_components(config_data)
    quantizer = PostTrainingQuantization(model, config.quantization)
    quantizer.prepare()
    quantizer.calibrate(batches)
    quantizer.convert()
    quantizer.export(output)
    return {
        "mode": "ptq",
        "bits": config.quantization.bits,
        "checkpoint": output,
        "execution": "dequantize_float_reference",
    }


def cmd_quantize(args: argparse.Namespace) -> int:
    emit_json(run_ptq_from_config(load_config_file(args.config), args.output))
    return 0
