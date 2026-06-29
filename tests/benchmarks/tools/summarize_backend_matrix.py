# -*- coding: utf-8 -*-
# @Time : 2026/6/15 下午5:22
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Compatibility CLI for backend matrix summaries.

Prefer `parascale benchmark-matrix` for new runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from parascale.reporting.matrix import build_report, write_markdown


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--workload-label", required=True)
    parser.add_argument(
        "--optimize-for",
        choices=["throughput", "memory", "balanced"],
        default="balanced",
    )
    parser.add_argument("--throughput-tolerance", type=float, default=0.05)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = build_report(
        Path(args.input),
        title=args.title,
        workload_label=args.workload_label,
        optimize_for=args.optimize_for,
        throughput_tolerance=args.throughput_tolerance,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report, markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
