# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Immutable token corpus preparation and identity contracts."""

import json

import pytest

pytest.importorskip("numpy")

from parascale.data.text import TokenCorpus, prepare_token_corpus  # noqa: E402


class Tokenizer:
    eos_token_id = 7
    backend_tokenizer = None

    def __init__(self):
        self.backend_tokenizer = self

    def to_str(self):
        return '{"tokenizer":"fixture-v1"}'

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [int(token) for token in text.split()]

    def __len__(self):
        return 8


def test_corpus_packs_nonempty_lines_with_eos_and_opens_readonly(tmp_path):
    source = tmp_path / "train.txt"
    source.write_text("1 2\n\n3 4 5\n", encoding="utf-8")
    output = tmp_path / "corpus"
    manifest = prepare_token_corpus(source, output, Tokenizer())
    corpus = TokenCorpus(output)
    assert corpus.tokens.tolist() == [1, 2, 7, 3, 4, 5, 7]
    assert manifest["token_count"] == 7
    assert manifest["document_count"] == 2
    assert not corpus.tokens.flags.writeable
    with pytest.raises(FileExistsError):
        prepare_token_corpus(source, output, Tokenizer())


@pytest.mark.parametrize(
    "fault", ["checksum", "length", "format", "tokenizer", "maximum"]
)
def test_corpus_rejects_modified_data_or_contract(tmp_path, fault):
    source = tmp_path / "train.txt"
    source.write_text("1 2 3\n", encoding="utf-8")
    output = tmp_path / "corpus"
    prepare_token_corpus(source, output, Tokenizer())
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if fault == "checksum":
        (output / "tokens.bin").write_bytes(bytes(16))
    elif fault == "length":
        manifest["token_count"] += 1
    elif fault == "format":
        manifest["format"] = "unknown"
    elif fault == "maximum":
        manifest["max_token_id"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        TokenCorpus(
            output, expected_tokenizer="wrong" if fault == "tokenizer" else None
        )


@pytest.mark.parametrize("text", ["", " \n\n", "99\n", "-1\n", "1" * 40])
def test_invalid_preparation_does_not_publish_output(tmp_path, text):
    source = tmp_path / "train.txt"
    source.write_text(text, encoding="utf-8")
    output = tmp_path / "corpus"
    with pytest.raises(ValueError):
        prepare_token_corpus(source, output, Tokenizer(), max_line_bytes=32)
    assert not output.exists()
    assert list(tmp_path.glob(".corpus-*")) == []
