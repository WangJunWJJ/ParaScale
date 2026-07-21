# -*- coding: utf-8 -*-
# @Time : 2026/6/10 下午3:15
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

def test_deepspeed_checkpoint_path_parser_without_torch():
    import os

    def parse_checkpoint_path(checkpoint_path, tag=None):
        if tag is not None:
            return checkpoint_path, tag
        base = os.path.basename(os.path.normpath(checkpoint_path))
        if base.startswith("checkpoint_"):
            return os.path.dirname(os.path.normpath(checkpoint_path)), base
        return checkpoint_path, None

    load_dir, tag = parse_checkpoint_path("runs/ckpt", tag="global_step10")
    assert load_dir == "runs/ckpt"
    assert tag == "global_step10"

    load_dir, tag = parse_checkpoint_path("runs/ckpt/checkpoint_10")
    assert load_dir.replace("\\", "/") == "runs/ckpt"
    assert tag == "checkpoint_10"

def test_deepspeed_config_merges_parascale_values_without_torch():
    from parascale.config import ParaScaleConfig
    from parascale.runtime.backends.deepspeed import DeepSpeedTrainingBackend

    config = ParaScaleConfig(
        training_backend="deepspeed",
        batch_size=4,
        gradient_accumulation_steps=3,
        precision="bf16",
        zero_stage=3,
        zero_offload=True,
        deepspeed_config={
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": 1},
            "steps_per_print": 5,
        },
    )

    merged = DeepSpeedTrainingBackend(config=config).build_deepspeed_config()

    assert merged["train_micro_batch_size_per_gpu"] == 4
    assert merged["gradient_accumulation_steps"] == 3
    assert merged["zero_optimization"]["stage"] == 3
    assert merged["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert merged["bf16"]["enabled"] is True
    assert merged["steps_per_print"] == 5
    assert merged["_parascale"]["merged_user_config"] is True

def test_deepspeed_config_preserves_zero_stage_zero_without_torch():
    from parascale.config import ParaScaleConfig
    from parascale.runtime.backends.deepspeed import DeepSpeedTrainingBackend

    config = ParaScaleConfig(
        training_backend="deepspeed",
        zero_stage=0,
        batch_size=2,
        gradient_accumulation_steps=1,
    )

    merged = DeepSpeedTrainingBackend(config=config).build_deepspeed_config()

    assert merged["zero_optimization"]["stage"] == 0
    assert merged["_parascale"]["resolved_config"] is True
