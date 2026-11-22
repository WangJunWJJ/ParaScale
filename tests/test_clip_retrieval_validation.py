# -*- coding: utf-8 -*-
# @Time : 2026/11/23 上午12:50
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Retrieval acceptance uses explicit positive ownership, not diagonal labels."""

import pytest

torch = pytest.importorskip("torch")


def test_retrieval_accepts_multiple_shuffled_positive_captions():
    from tests.benchmarks.scripts.validate_clip_retrieval import retrieval_metrics

    scores = torch.tensor([[0.1, 0.9, 0.8, 0.2], [0.8, 0.1, 0.2, 0.9]])
    result = retrieval_metrics(scores, [1, 0, 0, 1])
    assert all(value == 1.0 for value in result.values())


def test_retrieval_ties_use_stable_gallery_order():
    from tests.benchmarks.scripts.validate_clip_retrieval import retrieval_metrics

    result = retrieval_metrics(torch.zeros(2, 4), [1, 0, 0, 1])
    assert result["image_to_text_r1"] == 0.5
    assert result["text_to_image_r1"] == 0.5
    assert result["image_to_text_r5"] == 1.0
    assert result["text_to_image_r10"] == 1.0


@pytest.mark.parametrize("owners", [[0, 2], [0, 0], [0], []])
def test_retrieval_rejects_invalid_positive_ownership(owners):
    from tests.benchmarks.scripts.validate_clip_retrieval import retrieval_metrics

    with pytest.raises(ValueError):
        retrieval_metrics(torch.eye(2), owners)


def test_retrieval_rejects_nonfinite_scores():
    from tests.benchmarks.scripts.validate_clip_retrieval import retrieval_metrics

    with pytest.raises(ValueError, match="finite"):
        retrieval_metrics(torch.tensor([[float("nan")]]), [0])


def test_data_identity_rejects_cross_split_and_within_split_duplicates():
    from tests.benchmarks.tools.prepare_flickr8k_retrieval import check_unique

    seen = set()
    check_unique("train", "a", "pixels", seen)
    with pytest.raises(ValueError, match="duplicate"):
        check_unique("validation", "b", "pixels", seen)
    with pytest.raises(ValueError, match="duplicate"):
        check_unique("train", "a", "different", seen)


@pytest.mark.parametrize("problem", [None, "duplicate", "overlap", "revision"])
def test_evaluation_rejects_corrupt_manifest_before_model_loading(tmp_path, problem):
    import json
    from types import SimpleNamespace

    from tests.benchmarks.scripts.validate_clip_retrieval import (
        evaluate,
        validate_manifest,
    )
    from tests.benchmarks.tools.prepare_flickr8k_retrieval import FILES, REVISION

    manifest = {
        "schema_version": 1,
        "source": "jxie/flickr8k",
        "revision": REVISION,
        "source_sha256": FILES,
        "splits": {},
    }
    for split, count, offset in (("train", 5999, 0), ("validation", 1000, 6000)):
        manifest["splits"][split] = [
            {
                "key": f"{i:06d}",
                "shard": f"{split}/{i // 1500:05d}.tar",
                "image_sha256": f"{i + offset:064x}",
                "pixel_sha256": f"{i + offset + 10000:064x}",
                "captions": ["text"] * (10 if split == "train" and i == 1027 else 5),
                "source_rows": (
                    [1027, 1359]
                    if split == "train" and i == 1027
                    else [i + int(split == "train" and i >= 1359)]
                ),
            }
            for i in range(count)
        ]
    if problem is None:
        assert len(validate_manifest(manifest)) == 1000
        return
    if problem == "revision":
        manifest["revision"] = "wrong"
    elif problem == "duplicate":
        manifest["splits"]["validation"][1] = manifest["splits"]["validation"][0]
    else:
        manifest["splits"]["validation"][0]["image_sha256"] = manifest["splits"][
            "train"
        ][0]["image_sha256"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate(SimpleNamespace(data=tmp_path, model=tmp_path / "missing"))


def test_evaluation_memory_backend_matches_selected_device(monkeypatch):
    from tests.benchmarks.scripts.validate_clip_retrieval import select_accelerator

    calls = []
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda value: calls.append(str(value))
    )
    assert select_accelerator("cpu") is None
    assert not calls
    assert select_accelerator("cuda:1") is torch.cuda
    assert calls == ["cuda:1"]
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert select_accelerator("cuda") is torch.cuda
    assert calls[-1] == "cuda:2"


def test_training_scalar_storage_restores_native_hf_shape():
    from tests.benchmarks.scripts.production_soak import hf_state_from_training

    stored = {"module.wrapped_model.logit_scale": torch.tensor([2.0])}
    target = {"logit_scale": torch.tensor(0.0)}
    restored = hf_state_from_training(stored, target)
    assert restored["logit_scale"].shape == torch.Size([])
    assert restored["logit_scale"].item() == 2.0
    assert stored["module.wrapped_model.logit_scale"].shape == (1,)


def test_quality_gate_requires_same_inputs_and_one_percentage_point_limit():
    from tests.benchmarks.scripts.validate_clip_retrieval import compare_quality

    baseline = {
        "status": "passed",
        "manifest_sha256": "data",
        "evaluation_identity": {"processor": "fixed"},
        "model_sha256": "initial",
        "checkpoint": None,
        "metrics": {"image_to_text_r1": 0.6, "text_to_image_r1": 0.4},
    }
    current = {"image_to_text_r1": 0.59, "text_to_image_r1": 0.39}
    checks = compare_quality(
        current, baseline, "data", {"processor": "fixed"}, "initial"
    )
    assert all(check["passed"] for check in checks.values())
    current["image_to_text_r1"] = 0.589
    assert not compare_quality(
        current, baseline, "data", {"processor": "fixed"}, "initial"
    )["image_to_text_r1"]["passed"]
    with pytest.raises(ValueError, match="identity"):
        compare_quality(current, baseline, "data", {"processor": "changed"}, "initial")
    with pytest.raises(ValueError, match="identity"):
        compare_quality(current, baseline, "data", {"processor": "fixed"}, "different")
    baseline["checkpoint"] = {"step": 50}
    with pytest.raises(ValueError, match="identity"):
        compare_quality(current, baseline, "data", {"processor": "fixed"}, "initial")
