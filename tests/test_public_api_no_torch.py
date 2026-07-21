# -*- coding: utf-8 -*-
# @Time : 2026/7/3 下午4:53
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Frozen ParaScale 0.2 public API tests without Torch."""

from pathlib import Path

import parascale


def test_public_api_matches_v0_2_snapshot():
    snapshot = Path(__file__).with_name("public_api_v0_2.txt")
    expected = [line for line in snapshot.read_text(encoding="utf-8").splitlines()]

    assert sorted(parascale.__all__) == expected


def test_config_schema_contract_is_public():
    assert parascale.PUBLIC_API_VERSION == "0.2"
    assert parascale.CURRENT_CONFIG_SCHEMA_VERSION == 1
    assert callable(parascale.validate_config_schema)
    assert callable(parascale.migrate_config_schema)
