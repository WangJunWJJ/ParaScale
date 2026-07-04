# -*- coding: utf-8 -*-
# @Time : 2026/7/3 下午4:45
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Configuration schema and migration tests without Torch."""

import json
from pathlib import Path

import pytest

from parascale.cli import main
from parascale.commands.common import load_config_file
from parascale.configuration import (
    CURRENT_CONFIG_SCHEMA_VERSION,
    migrate_config_schema,
    validate_config_schema,
)

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_config_is_valid_and_reported_as_migratable():
    config = {"parascale": {"training_backend": "native"}}

    report = validate_config_schema(config)

    assert report["ok"] is True
    assert report["schema_version"] == 0
    assert report["current_schema_version"] == 1
    assert report["legacy"] is True
    assert report["migration_required"] is True


def test_migrate_config_schema_is_non_mutating_and_idempotent():
    config = {"parascale": {"training_backend": "native"}}

    migrated = migrate_config_schema(config)
    migrated_again = migrate_config_schema(migrated)

    assert config == {"parascale": {"training_backend": "native"}}
    assert migrated["schema_version"] == CURRENT_CONFIG_SCHEMA_VERSION
    assert migrated_again == migrated


def test_future_config_schema_is_rejected_before_runtime():
    config = {"schema_version": CURRENT_CONFIG_SCHEMA_VERSION + 1}

    with pytest.raises(ValueError, match="newer than supported"):
        validate_config_schema(config)


def test_config_loader_rejects_future_schema(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")

    with pytest.raises(ValueError, match="newer than supported"):
        load_config_file(str(path))


def test_config_validate_cli_reports_legacy_schema(tmp_path):
    source = tmp_path / "legacy.json"
    output = tmp_path / "validation.json"
    source.write_text(json.dumps({"parascale": {}}), encoding="utf-8")

    exit_code = main(
        [
            "config",
            "validate",
            "--config",
            str(source),
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["mode"] == "config_validate"
    assert payload["validation"]["legacy"] is True


def test_config_migrate_cli_writes_v1_without_overwriting_source(tmp_path):
    source = tmp_path / "legacy.json"
    output = tmp_path / "migrated.json"
    original = {"parascale": {"training_backend": "native"}}
    source.write_text(json.dumps(original), encoding="utf-8")

    exit_code = main(
        [
            "config",
            "migrate",
            "--config",
            str(source),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(source.read_text(encoding="utf-8")) == original
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1


def test_shipped_configs_declare_current_schema_version():
    paths = list((ROOT / "configs").rglob("*.json"))
    paths.extend((ROOT / "configs").rglob("*.yaml"))
    paths.extend((ROOT / "examples").rglob("config.json"))

    missing = []
    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix == ".json":
            config = json.loads(text)
        else:
            import yaml

            config = yaml.safe_load(text)
        if config.get("schema_version") != CURRENT_CONFIG_SCHEMA_VERSION:
            missing.append(str(path.relative_to(ROOT)))
    assert missing == []
