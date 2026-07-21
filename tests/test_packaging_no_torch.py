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
    assert "pillow>=10.0.0" in data["project"]["optional-dependencies"]["dev"]


def test_ascend_extra_is_vendor_managed_outside_public_pypi_matrix():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert data["project"]["optional-dependencies"]["ascend"] == [
        "torch-npu>=2.4.0"
    ]
    assert "extra: [gpu, deepspeed, vlm]" in workflow
    assert "torch-npu wheels are vendor-managed" in workflow


def test_release_package_boundary_excludes_non_product_assets():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_find = data["tool"]["setuptools"]["packages"]["find"]

    assert package_find["include"] == ["parascale*"]
    assert package_find["exclude"] == ["tests*", "examples*"]
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for forbidden in (
        ".superpowers/",
        "dist/",
        "runs/",
        "*.pt",
        "*.pth",
        "*.tar.gz",
    ):
        assert forbidden in gitignore


def test_benchmark_report_has_one_review_entrypoint():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert not (ROOT / "tests" / "UNIFIED_TEST_BENCHMARK_REPORT.md").exists()
    assert (
        "tests/benchmarks/reports/BENCHMARK_REPORT.md"
        in readme
    )
    assert "tests/UNIFIED_TEST_BENCHMARK_REPORT.md" not in readme


def test_trial_release_changelog_declares_version_and_limitations():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "## 0.1.0" in changelog
    for section in ("Added", "Changed", "Fixed", "Validation", "Known Limitations"):
        assert f"### {section}" in changelog


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
    assert '"--version"' in source
    assert '"--strict"' in source
    assert '"--require"' in source
    assert '"torch"' in source
    assert source.count('"config"') >= 2
    assert '"migrate"' in source
    assert "migrated_config" in source
    assert "configs" not in source
    assert "repo-root" not in source


def test_ci_covers_supported_python_and_clean_install():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    for version in ("3.10", "3.11", "3.12"):
        assert version in workflow
    assert "python -m build" in workflow
    assert "verify_clean_install.py" in workflow
    assert "verify_clean_install.py --repo-root" not in workflow
    assert "python tests/run_tests.py" in workflow
    assert "python -m ruff check" in workflow
    assert "pillow>=10.0.0" in workflow
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow
    assert "permissions:\n  contents: read" in workflow


def test_github_issue_forms_cover_trial_release_feedback():
    template_dir = ROOT / ".github" / "ISSUE_TEMPLATE"
    expected = {
        "bug_report.yml": ("ParaScale version", "ResolvedConfig", "Reproduction"),
        "performance_regression.yml": (
            "Baseline",
            "Measurement window",
            "ResolvedConfig",
        ),
        "workload_request.yml": ("Model", "Dataset", "Acceptance criteria"),
    }

    for filename, required_text in expected.items():
        text = (template_dir / filename).read_text(encoding="utf-8")
        assert "name:" in text
        assert "description:" in text
        assert "body:" in text
        for value in required_text:
            assert value in text
    config = (template_dir / "config.yml").read_text(encoding="utf-8")
    assert "blank_issues_enabled: false" in config
