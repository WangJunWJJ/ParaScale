# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Immutable causal token corpora and resumable data-parallel batches."""

from .artifact import TokenCorpus, prepare_token_corpus, tokenizer_fingerprint
from .batches import TokenBatchCursor

__all__ = [
    "TokenBatchCursor",
    "TokenCorpus",
    "prepare_token_corpus",
    "tokenizer_fingerprint",
]
