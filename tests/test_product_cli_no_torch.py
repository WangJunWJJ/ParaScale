# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午7:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Product entrypoints reuse command contracts without exposing other products."""

import json
import subprocess
import sys

import pytest

from parascale import cli
from parascale._version import __version__


@pytest.mark.parametrize("product", ["train", "infer"])
def test_product_version_and_help(product, capsys):
    entry = getattr(cli, f"{product}_main")
    with pytest.raises(SystemExit) as result:
        entry(["--version"])
    assert result.value.code == 0
    assert capsys.readouterr().out.strip() == f"parascale-{product} {__version__}"
    with pytest.raises(SystemExit) as result:
        entry(["--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert f"parascale-{product}" in help_text
    assert "run" in help_text and "doctor" in help_text
    assert "serve" not in help_text


@pytest.mark.parametrize("product,forbidden", [("train", "quantize"), ("infer", "prepare-text"), ("infer", "checkpoint")])
def test_product_rejects_cross_product_commands(product, forbidden):
    with pytest.raises(SystemExit) as result:
        getattr(cli, f"{product}_main")([forbidden])
    assert result.value.code == 2


@pytest.mark.parametrize("product", ["train", "infer"])
def test_product_run_reuses_existing_handler(product, monkeypatch):
    called = []

    def execute(args):
        called.append(args)
        return 0

    monkeypatch.setattr(f"parascale.commands.run.cmd_{product}", execute)
    assert getattr(cli, f"{product}_main")(["run", "--config", "input.json"]) == 0
    assert len(called) == 1 and called[0].config == "input.json"


@pytest.mark.parametrize("product", ["train", "infer"])
def test_product_errors_use_shared_json_contract(product, monkeypatch, capsys):
    monkeypatch.delenv("PARASCALE_DEBUG", raising=False)

    def execute(args):
        raise ValueError("invalid product config")

    monkeypatch.setattr(f"parascale.commands.run.cmd_{product}", execute)
    assert getattr(cli, f"{product}_main")(["run", "--config", "input.json"]) != 0
    error = json.loads(capsys.readouterr().err)
    assert "invalid product config" in json.dumps(error)


def test_product_help_does_not_import_optional_engines():
    code = (
        "import sys; from parascale.cli import infer_main; "
        "assert not {'torch', 'deepspeed', 'transformers', 'torch_npu'} & sys.modules.keys(); "
        "infer_main(['--help'])"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
