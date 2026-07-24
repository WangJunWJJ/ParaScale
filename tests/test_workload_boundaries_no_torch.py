# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com



def test_workload_capability_lives_outside_orchestrator_without_torch(monkeypatch):
    from parascale.workloads.capability import capability_level_for_training

    monkeypatch.setenv("WORLD_SIZE", "1")
    config = {
        "training": {"workload": "clip_contrastive"},
        "data": {"type": "datacomp_wds"},
        "hardware_profile": {"world_size": 2, "gpus_per_node": 1, "num_nodes": 2},
    }

    assert capability_level_for_training(config) == "multi_node_smoke"


def test_detection_workloads_have_specific_capability_levels_without_torch(monkeypatch):
    from parascale.workloads.capability import describe_workload

    monkeypatch.setenv("WORLD_SIZE", "1")

    yolo = describe_workload(
        {
            "training": {"workload": "yolo_world"},
            "data": {"type": "objects365_cached"},
        }
    )
    ground = describe_workload(
        {
            "training": {"workload": "ground_dino"},
            "data": {"type": "phrase_grounding"},
        }
    )

    assert yolo.flags["yolo_world"] is True
    assert yolo.capability_level == "local_native_yolo_world_objects365"
    assert ground.flags["ground_dino"] is True
    assert ground.capability_level == "local_native_ground_dino_phrase_grounding"

def test_benchmark_aggregation_lives_in_reporting_without_torch():
    from parascale.reporting.aggregation import aggregate_stable_metrics

    metrics = aggregate_stable_metrics(
        [
            {"images_per_second": 10.0, "step_time_seconds": 0.2},
            {"images_per_second": 20.0, "step_time_seconds": 0.1},
        ],
        warmup_steps=1,
    )

    assert metrics["measured_steps"] == 1.0
    assert metrics["stable_images_per_second"] == 20.0
    assert metrics["stable_step_time_ms"] == 100.0

def test_benchmark_aggregation_includes_stable_loss_window():
    from parascale.reporting.aggregation import aggregate_stable_metrics

    metrics = aggregate_stable_metrics(
        [{"loss": 9.0}, {"loss": 2.0}, {"loss": 1.0}],
        warmup_steps=1,
    )

    assert metrics["stable_loss"] == 1.5
    assert metrics["stable_min_loss"] == 1.0
    assert metrics["stable_max_loss"] == 2.0

def test_train_runner_does_not_own_workload_capability_or_aggregation_without_torch():
    from pathlib import Path

    source = Path("parascale/runtime/train_runner.py").read_text(encoding="utf-8")

    assert "workload_flags" not in source
    assert "capability_level_for_training" not in source
    assert "def _aggregate_stable_metrics" not in source
    assert "local_native_clip_contrastive" not in source
    assert "local_native_vlm_lora" not in source

def test_synthetic_regression_is_built_through_workload_registry_without_torch():
    from pathlib import Path

    train_runner_source = Path("parascale/runtime/train_runner.py").read_text(
        encoding="utf-8"
    )
    command_source = Path("parascale/commands/run.py").read_text(encoding="utf-8")

    assert "build_synthetic_regression_components" not in train_runner_source
    assert "SyntheticRegressionSpec" not in train_runner_source
    assert 'if flags["synthetic"]:' not in train_runner_source
    assert "build_synthetic_regression_components" not in command_source

def test_clip_workload_dataloader_does_not_own_device_placement_without_torch():
    from pathlib import Path

    source = Path("parascale/workloads/clip.py").read_text(encoding="utf-8")
    forbidden = [
        'batch["pixel_values"] = batch["pixel_values"].to(device)',
        'batch["input_ids"] = batch["input_ids"].to(device)',
        'batch["attention_mask"] = batch["attention_mask"].to(device)',
        "device=device",
    ]

    for snippet in forbidden:
        assert snippet not in source

def test_training_workload_dataloaders_do_not_own_device_placement_without_torch():
    from pathlib import Path

    forbidden_by_file = {
        "parascale/workloads/vision.py": [
            'batch["pixel_values"].to(device)',
            'batch["labels"].to(device)',
        ],
        "parascale/workloads/tiny.py": [
            "x.to(device)",
            "y.to(device)",
        ],
        "parascale/workloads/yolo.py": [
            "torch.stack(images, dim=0).to(device)",
        ],
        "parascale/workloads/vlm_lora.py": [
            "_vlm_move_batch_to_device(",
            'batch_device = "cpu" if spec.cuda_prefetch else device',
            'batch["pixel_values"] = batch["pixel_values"].to(batch_device)',
            'batch["input_ids"] = batch["input_ids"].to(batch_device)',
            'batch["attention_mask"] = batch["attention_mask"].to(batch_device)',
        ],
    }

    for path, snippets in forbidden_by_file.items():
        source = Path(path).read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet not in source

def test_training_workloads_do_not_own_model_placement_without_torch():
    from pathlib import Path

    forbidden_by_file = {
        "parascale/workloads/vision.py": [
            "TinyPatchClassifier().to(device)",
        ],
        "parascale/workloads/tiny.py": [
            "TinyTorchMLP().to(device)",
        ],
        "parascale/workloads/yolo.py": [
            "yolo.model.to(device)",
        ],
        "parascale/workloads/vlm_lora.py": [
            "TinyVlmLora().to(device)",
            "HfVlmLoraWrapper(peft_model).to(device)",
            ").to(device)",
            "AutoModel.from_pretrained(spec.pretrained_model_name_or_path).to(device)",
            "HFClipLoraAdapter(base_model).to(device)",
        ],
    }

    for path, snippets in forbidden_by_file.items():
        source = Path(path).read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet not in source

def test_dependency_metadata_matches_runtime_capabilities_without_torch():
    from pathlib import Path

    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "torch>=2.4.0" in requirements
    assert "torchvision>=0.19.0" in requirements
    assert "[project.optional-dependencies]" in pyproject
    assert "deepspeed = [" in pyproject
    assert "datacomp = [" in pyproject
    assert "vlm = [" in pyproject
    assert "yolo = [" in pyproject
    assert "ascend = [" in pyproject
    assert '"deepspeed>=' in pyproject
    assert '"transformers>=' in pyproject
    assert '"peft>=' in pyproject
    assert '"webdataset>=' in pyproject
    assert '"pandas>=' in pyproject
    assert '"pyarrow>=' in pyproject
    assert '"pillow>=' in pyproject
    assert '"ultralytics>=' in pyproject
    assert '"torch-npu>=' in pyproject
