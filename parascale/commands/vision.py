# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午6:03
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Vision data pipeline command implementation."""

from __future__ import annotations

import argparse
from typing import Any, Dict

from parascale.commands.common import emit_json
from parascale.data import profile_image_folder


def register_vision_profile_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "vision-profile", help="Profile a real image folder data pipeline."
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing real image files."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Images per profiling batch."
    )
    parser.add_argument(
        "--max-batches", type=int, default=4, help="Maximum batches to profile."
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize image edge used for tensor batches.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Patch size used for patch-token accounting.",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, etc."
    )
    parser.add_argument(
        "--output", help="Optional path to write the profile JSON."
    )
    parser.set_defaults(func=cmd_vision_profile)


def build_vision_profile_payload(args: argparse.Namespace) -> Dict[str, Any]:
    profile = profile_image_folder(
        args.data_dir,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        image_size=args.image_size,
        patch_size=args.patch_size,
        device=args.device,
    )
    return {
        "mode": "vision_profile",
        "runtime_status": "real_data_profile",
        "profile": profile.to_dict(),
    }


def cmd_vision_profile(args: argparse.Namespace) -> int:
    emit_json(build_vision_profile_payload(args), args.output)
    return 0
