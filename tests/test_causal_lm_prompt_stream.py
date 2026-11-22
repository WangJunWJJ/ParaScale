# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午10:14
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded, lossless local JSONL prompt input."""

import json

import pytest


def stream(*args, **kwargs):
    from parascale.data.text.prompts import iter_prompt_batches

    return iter_prompt_batches(*args, **kwargs)


def test_prompt_stream_preserves_order_ids_and_short_tail(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"prompt": "a b", "id": "first"},
                {"prompt": "c"},
                {"prompt": "d"},
            ]
        ),
        encoding="utf-8",
    )
    batches = list(stream(path, batch_size=2, max_line_bytes=128))
    assert batches == [
        {"prompts": ["a b", "c"], "ids": ["first", "2"]},
        {"prompts": ["d"], "ids": ["3"]},
    ]


def test_prompt_limit_does_not_read_invalid_tail(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_bytes(b'{"prompt":"one"}\ninvalid\n')
    assert list(stream(path, batch_size=3, max_line_bytes=128, max_requests=1)) == [
        {"prompts": ["one"], "ids": ["1"]},
    ]


@pytest.mark.parametrize(
    "line",
    [
        b"\xff\n",
        b"null\n",
        b"{}\n",
        b"[]\n",
        b"\n",
        b'{"prompt":" "}\n',
        b'{"prompt":"one","id":1}\n',
        b'{"prompt":"one","extra":true}\n',
        b'{"prompt":"one","prompt":"two"}\n',
    ],
)
def test_bad_records_report_line_number(tmp_path, line):
    path = tmp_path / "prompts.jsonl"
    path.write_bytes(b'{"prompt":"good"}\n' + line)
    batches = stream(path, batch_size=1, max_line_bytes=128)
    assert next(batches)["prompts"] == ["good"]
    with pytest.raises(ValueError, match="line 2"):
        next(batches)


def test_stream_uses_bounded_binary_readline(tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "prompts.jsonl"
    path.write_bytes(b'{"prompt":"' + b"a" * 10000 + b'"}\n')
    original = Path.open
    sizes = []

    class Guarded:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def readline(self, size=-1):
            sizes.append(size)
            assert size == 65
            return self.handle.readline(size)

    def guarded(self, *args, **kwargs):
        return Guarded(original(self, *args, **kwargs))

    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="line 1.*64"):
        list(stream(path, batch_size=1, max_line_bytes=64))
    assert sizes == [65]


@pytest.mark.parametrize(
    "option,value",
    [
        ("batch_size", 0),
        ("batch_size", True),
        ("max_line_bytes", 0),
        ("max_requests", -1),
        ("max_requests", True),
        ("batch_size", 1.5),
    ],
)
def test_invalid_limits_fail_before_io(tmp_path, option, value):
    options = {"batch_size": 1, "max_line_bytes": 128, option: value}
    with pytest.raises(ValueError, match=option):
        list(stream(tmp_path / "missing", **options))
