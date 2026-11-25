# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Prepare immutable token data through the unified CLI."""

from __future__ import annotations

import argparse

from .common import emit_json


def register_text_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prepare-text", help="Pack a local UTF-8 text corpus."
    )
    parser.add_argument(
        "--input", required=True, help="UTF-8 file, one document per line."
    )
    parser.add_argument(
        "--tokenizer", required=True, help="Local HuggingFace tokenizer directory."
    )
    parser.add_argument(
        "--output", required=True, help="New corpus directory; must not exist."
    )
    parser.add_argument("--max-line-bytes", type=int, default=1024 * 1024)
    parser.set_defaults(func=cmd_prepare_text)


def cmd_prepare_text(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from parascale.data.text import prepare_token_corpus

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, use_fast=True, local_files_only=True, trust_remote_code=False
    )
    emit_json(
        prepare_token_corpus(
            args.input, args.output, tokenizer, max_line_bytes=args.max_line_bytes
        )
    )
    return 0
