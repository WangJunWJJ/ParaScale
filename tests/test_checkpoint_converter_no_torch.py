# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import tempfile
from pathlib import Path

from parascale import CheckpointConverter, CheckpointManager, CheckpointManifest


def test_checkpoint_converter_validates_format_matrix():
    converter = CheckpointConverter()
    plan = converter.build_plan(
        "hf", target_format="serve_manifest", source_path="missing"
    )

    assert plan.source_format == "hf"
    assert plan.target_format == "serve_manifest"
    assert plan.metadata["requires_weight_rewrite"] is True
    assert plan.metadata["source_exists"] is False
    try:
        converter.build_plan("unknown")
    except ValueError as exc:
        assert "unsupported checkpoint source format" in str(exc)
    else:
        raise AssertionError("unknown checkpoint source format must be rejected")

def test_checkpoint_converter_emits_serve_manifest_for_parascale_manifest():
    root = Path(tempfile.gettempdir()) / "parascale-test-runs" / "checkpoint_converter"
    manager = CheckpointManager(str(root))
    manifest = CheckpointManifest(
        step=3,
        backend="native",
        files=[
            {"path": "backend_state.pt", "role": "backend_state", "format": "torch"}
        ],
        metadata={"note": "source"},
    )
    source = manager.write_manifest(manifest)
    target = root / "serve" / "manifest.json"

    converter = CheckpointConverter()
    plan = converter.build_plan(
        "parascale",
        target_format="serve_manifest",
        source_path=str(source),
        target_path=str(target),
    )
    result = converter.convert(plan)

    assert result["converted"] is True
    assert result["target_manifest"] == str(target)
    converted = CheckpointManifest.from_dict(
        __import__("json").loads(target.read_text(encoding="utf-8"))
    )
    assert converted.format == "parascale_serve_manifest_v1"
    assert converted.metadata["serve_ready"] is True
    assert converted.metadata["conversion_target"] == "serve_manifest"

def test_checkpoint_converter_inspects_hf_checkpoint_directory():
    root = Path(tempfile.gettempdir()) / "parascale-test-runs" / "hf_converter"
    source = root / "hf"
    target = root / "converted" / "manifest.json"
    source.mkdir(parents=True, exist_ok=True)
    (source / "config.json").write_text(
        '{"model_type": "tiny", "hidden_size": 8}\n', encoding="utf-8"
    )
    (source / "model.safetensors").write_bytes(b"tiny-weights")

    converter = CheckpointConverter()
    plan = converter.build_plan(
        "hf",
        target_format="serve_manifest",
        source_path=str(source),
        target_path=str(target),
    )
    result = converter.convert(plan)

    assert result["converted"] is True
    assert result["weight_files"] == 1
    assert result["weight_rewrite_performed"] is False
    converted = CheckpointManifest.from_dict(
        __import__("json").loads(target.read_text(encoding="utf-8"))
    )
    assert converted.format == "parascale_serve_manifest_v1"
    assert converted.backend == "hf"
    assert converted.files[0]["format"] == "safetensors"
    assert converted.metadata["hf_config"]["model_type"] == "tiny"
    assert converted.metadata["serve_layout"]["loader"] == "hf_reference"
