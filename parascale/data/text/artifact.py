# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Bounded-memory preparation and verified read-only token storage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from parascale.checkpoint.io import atomic_binary_writer

FORMAT = "parascale.tokens.v1"


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Bind the fast tokenizer's vocabulary, normalization and EOS policy."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not callable(getattr(backend, "to_str", None)):
        raise ValueError("Token corpus preparation requires a fast tokenizer")
    description = {
        "backend": json.loads(backend.to_str()),
        "eos_token_id": tokenizer.eos_token_id,
        "vocabulary_size": len(tokenizer),
    }
    payload = json.dumps(description, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_token_corpus(
    source: str | Path,
    output: str | Path,
    tokenizer: Any,
    *,
    max_line_bytes: int = 1024 * 1024,
) -> dict[str, Any]:
    """Pack nonempty UTF-8 lines as documents terminated by one EOS token.

    Publish a new directory only after both payload and manifest are complete.
    Existing outputs are never updated. Long lines fail rather than silently
    truncating a document or allocating an unbounded tokenization buffer.
    """
    import numpy as np

    if (
        isinstance(max_line_bytes, bool)
        or not isinstance(max_line_bytes, int)
        or max_line_bytes < 1
    ):
        raise ValueError("max_line_bytes must be a positive integer")
    eos = tokenizer.eos_token_id
    vocabulary = len(tokenizer)
    if (
        isinstance(eos, bool)
        or not isinstance(eos, int)
        or not 0 <= eos < min(vocabulary, 2**32)
    ):
        raise ValueError("Tokenizer must define a valid EOS token")
    fingerprint = tokenizer_fingerprint(tokenizer)
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash, token_hash = hashlib.sha256(), hashlib.sha256()
    count, documents, maximum = 0, 0, eos
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        with (
            source.open("rb") as text,
            atomic_binary_writer(staging / "tokens.bin") as binary,
        ):
            while raw := text.readline(max_line_bytes + 1):
                if len(raw) > max_line_bytes:
                    raise ValueError(
                        f"Input document exceeds max_line_bytes={max_line_bytes}"
                    )
                source_hash.update(raw)
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                ids = list(tokenizer.encode(line, add_special_tokens=False)) + [eos]
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value < min(vocabulary, 2**32)
                    for value in ids
                ):
                    raise ValueError("Tokenizer returned an invalid token ID")
                payload = np.asarray(ids, dtype="<u4").tobytes()
                binary.write(payload)
                token_hash.update(payload)
                count += len(ids)
                documents += 1
                maximum = max(maximum, max(ids))
        if not documents:
            raise ValueError("Text corpus contains no nonempty documents")
        manifest = {
            "format": FORMAT,
            "dtype": "<u4",
            "token_count": count,
            "document_count": documents,
            "max_token_id": maximum,
            "sha256": token_hash.hexdigest(),
            "source_sha256": source_hash.hexdigest(),
            "tokenizer": fingerprint,
            "eos_token_id": eos,
            "document_policy": "nonempty_utf8_lines_eos",
        }
        with atomic_binary_writer(staging / "manifest.json") as stream:
            stream.write(json.dumps(manifest, sort_keys=True).encode("utf-8"))
        os.rename(staging, output)
        if os.name == "posix":
            descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    return manifest


class TokenCorpus:
    """Validate an immutable corpus once, then expose its read-only memory map."""

    def __init__(
        self, path: str | Path, *, expected_tokenizer: str | None = None
    ) -> None:
        import numpy as np

        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT or manifest.get("dtype") != "<u4":
            raise ValueError("Unsupported token corpus format")
        count = manifest.get("token_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Invalid token_count")
        if (
            expected_tokenizer is not None
            and manifest.get("tokenizer") != expected_tokenizer
        ):
            raise ValueError("Token corpus tokenizer identity mismatch")
        payload = path / "tokens.bin"
        if payload.stat().st_size != count * 4:
            raise ValueError("Token corpus payload length mismatch")
        digest = hashlib.sha256()
        maximum = 0
        with payload.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                maximum = max(maximum, int(np.frombuffer(chunk, dtype="<u4").max()))
        if digest.hexdigest() != manifest.get("sha256"):
            raise ValueError("Token corpus checksum mismatch")
        if (
            type(manifest.get("max_token_id")) is not int
            or manifest["max_token_id"] != maximum
        ):
            raise ValueError("Token corpus maximum token ID mismatch")
        if manifest.get("document_policy") != "nonempty_utf8_lines_eos":
            raise ValueError("Unsupported token corpus document policy")
        self.manifest = manifest
        self.tokens = np.memmap(payload, dtype="<u4", mode="r", shape=(count,))
        self.identity = {
            key: manifest[key]
            for key in (
                "format",
                "sha256",
                "token_count",
                "tokenizer",
                "document_policy",
            )
        }
