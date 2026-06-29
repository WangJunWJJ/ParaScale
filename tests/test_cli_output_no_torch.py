# -*- coding: utf-8 -*-
# @Time : 2026/6/25 上午11:02
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

from parascale.commands.common import emit_json


def test_emit_json_skips_nonzero_distributed_rank(tmp_path, monkeypatch):
    output = tmp_path / "payload.json"
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")

    emit_json({"rank": 1}, str(output))

    assert not output.exists()
