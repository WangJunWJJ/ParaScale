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
