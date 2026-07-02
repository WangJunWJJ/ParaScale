# -*- coding: utf-8 -*-
# @Time : 2026/7/2 下午4:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Packaging metadata tests that do not require Torch."""

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by Python 3.10 CI
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def _source_version() -> str:
    source = (ROOT / "parascale" / "_version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"$', source, re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_package_metadata_has_one_version_source():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "version" not in data["project"]
    assert data["project"]["dynamic"] == ["version"]
    assert data["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "parascale._version.__version__"
    }
    assert _source_version() == "0.1.0"


def test_setup_py_is_only_a_compatibility_shim():
    source = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert "setup()" in source
    for duplicate in ("version=", "install_requires=", "extras_require="):
        assert duplicate not in source


def test_pyproject_defines_build_and_console_entrypoint():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    assert data["build-system"]["requires"][0] == "setuptools>=77"
    assert data["project"]["scripts"]["parascale"] == "parascale.cli:main"
    assert data["project"]["requires-python"] == ">=3.10"
    assert data["project"]["license"] == "MIT"
    assert data["project"]["license-files"] == ["LICENSE"]
    assert 'tomli>=2.0; python_version < "3.11"' in data["project"][
        "optional-dependencies"
    ]["dev"]


def test_clean_install_verifier_uses_only_public_entrypoints():
    source = (
        ROOT / "tests" / "packaging" / "verify_clean_install.py"
    ).read_text(encoding="utf-8")

    assert "from parascale" not in source
    assert "python -m parascale.cli" not in source
    assert '"parascale"' in source
    assert "checkpoint" in source
    assert "validate" in source
    assert "PYTHONPATH" not in source


def test_ci_covers_supported_python_and_clean_install():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    for version in ("3.10", "3.11", "3.12"):
        assert version in workflow
    assert "python -m build" in workflow
    assert "verify_clean_install.py" in workflow
    assert "python tests/run_tests.py" in workflow
    assert "python -m ruff check" in workflow
