# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午10:16
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded local JSONL input for generation workloads."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def iter_prompt_batches(
    path: str | Path,
    *,
    batch_size: int,
    max_line_bytes: int,
    max_requests: int | None = None,
) -> Iterator[dict[str, list[str]]]:
    """Yield ordered prompt/ID batches without materializing the file.

    Args:
        path: Trusted local UTF-8 JSONL file, one prompt and optional string ID
            per line. Missing IDs use one-based line numbers.
        batch_size: Maximum records in a yielded batch.
        max_line_bytes: Maximum encoded line length, including its delimiter.
        max_requests: Optional positive request limit, checked before each read.

    Raises:
        ValueError: Invalid limits, nonregular input, or malformed/oversized line.
        OSError: Input cannot be opened or read.
    """
    for name, value in (
        ("batch_size", batch_size),
        ("max_line_bytes", max_line_bytes),
        ("max_requests", max_requests),
    ):
        if name == "max_requests" and value is None:
            continue
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Prompt input must be a regular file: {path}")
    prompts, ids = [], []
    with path.open("rb") as stream:
        line_number = 0
        while max_requests is None or line_number < max_requests:
            raw = stream.readline(max_line_bytes + 1)
            if not raw:
                break
            line_number += 1
            if len(raw) > max_line_bytes:
                raise ValueError(
                    f"Prompt line {line_number} exceeds {max_line_bytes} bytes"
                )
            try:
                record = json.loads(
                    raw.decode("utf-8"), object_pairs_hook=_unique_object
                )
                if not isinstance(record, dict) or set(record) - {"prompt", "id"}:
                    raise ValueError("expected prompt and optional id fields")
                prompt = record.get("prompt")
                identifier = record.get("id", str(line_number))
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("prompt must be a nonempty string")
                if not isinstance(identifier, str) or not identifier:
                    raise ValueError("id must be a nonempty string")
            except (ValueError, UnicodeError) as error:
                raise ValueError(
                    f"Invalid prompt at line {line_number}: {error}"
                ) from error
            prompts.append(prompt)
            ids.append(identifier)
            if len(prompts) == batch_size:
                yield {"prompts": prompts, "ids": ids}
                prompts, ids = [], []
    if prompts:
        yield {"prompts": prompts, "ids": ids}
