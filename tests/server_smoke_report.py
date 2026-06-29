#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/5/8 下午3:01
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Run a compact server smoke flow and write a JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parascale.cli import build_smoke_report


def main() -> int:
    parser = argparse.ArgumentParser(description="ParaScale server smoke report")
    parser.add_argument("--config", default="configs/server_tiny_torch.json")
    parser.add_argument("--output", default="runs/server_smoke_report.json")
    parser.add_argument(
        "--skip-real", action="store_true", help="Only run doctor and plan."
    )
    args = parser.parse_args()

    report = build_smoke_report(args.config, skip_real=args.skip_real)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
