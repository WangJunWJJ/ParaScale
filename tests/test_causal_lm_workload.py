# -*- coding: utf-8 -*-
# @Time : 2026/11/22 下午5:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Offline real Transformers execution through ParaScale's public commands."""

import json
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from parascale.cli import main  # noqa: E402
from parascale.runtime.train_runner import (  # noqa: E402
    _rank_component_config,
    run_train_from_config,
)
from parascale.workloads import build_training_components  # noqa: E402


@pytest.fixture
def model_path(tmp_path):
    torch.manual_seed(47)
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"<pad>": 0, "<eos>": 1, "<unk>": 2, "a": 3, "b": 4, "c": 5, "d": 6},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=7,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    path = tmp_path / "model"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def prepare(tmp_path, model_path, capsys):
    source = tmp_path / "text.txt"
    source.write_text("a b c d a b c\n" * 12, encoding="utf-8")
    output = tmp_path / "tokens"
    assert (
        main(
            [
                "prepare-text",
                "--input",
                str(source),
                "--tokenizer",
                str(model_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["token_count"] == 96
    return output


def configuration(model_path, data_path, checkpoint):
    return {
        "parascale": {"training_backend": "native", "precision": "fp32"},
        "training": {
            "workload": "causal_lm",
            "max_steps": 2,
            "seed": 42,
            "checkpoint_dir": str(checkpoint),
        },
        "model": {"pretrained_model_name_or_path": str(model_path)},
        "data": {
            "type": "token_corpus",
            "path": str(data_path),
            "batch_size": 2,
            "sequence_length": 8,
            "seed": 9,
            "shuffle": True,
            "repeat": True,
        },
        "optimizer": {"type": "adamw", "lr": 0.001},
    }


def test_cli_causal_lm_training_and_resume_matches_continuous(
    tmp_path, model_path, capsys
):
    data_path = prepare(tmp_path, model_path, capsys)

    def train(config, resume=False):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        args = ["train", "--config", str(config_path)]
        if resume:
            args += ["--resume-step", "2"]
        assert main(args) == 0
        return json.loads(capsys.readouterr().out)

    continuous = configuration(model_path, data_path, tmp_path / "continuous")
    continuous["training"]["max_steps"] = 4
    expected = train(continuous)
    split = configuration(model_path, data_path, tmp_path / "split")
    train(split)
    actual = train(split, resume=True)
    assert actual["global_step"] == expected["global_step"] == 4
    for key in ("continuous", "split"):
        path = tmp_path / key / "step-00000004" / "backend_state.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)[
            "backend_state"
        ]
        if key == "continuous":
            expected_weights = payload["model_state_dict"]
        else:
            for name, value in expected_weights.items():
                torch.testing.assert_close(
                    payload["model_state_dict"][name], value, rtol=0, atol=0
                )


def test_rank_seed_does_not_change_global_data_permutation(
    tmp_path, model_path, capsys, monkeypatch
):
    data_path = prepare(tmp_path, model_path, capsys)
    config = configuration(model_path, data_path, tmp_path / "checkpoints")
    blocks = []
    for rank in range(2):
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", "2")
        rank_config = _rank_component_config(config, rank=rank, distributed=True)
        _, _, data, _ = build_training_components(rank_config)
        blocks.append(set(next(iter(data))["block_ids"]))
        assert data.seed == 9
    assert not blocks[0].intersection(blocks[1])


@pytest.mark.parametrize("mode", ["ptq", "qat"])
def test_causal_lm_compression_uses_product_commands(
    tmp_path, model_path, capsys, mode
):
    data_path = prepare(tmp_path, model_path, capsys)
    config = configuration(model_path, data_path, tmp_path / "checkpoints")
    config["parascale"]["quantization"] = {
        "enabled": True,
        "mode": mode,
        "bits": 8,
        "calib_batches": 1,
    }
    config["training"]["quantized_output"] = str(tmp_path / "quantized.pt")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    args = ["quantize" if mode == "ptq" else "train", "--config", str(path)]
    if mode == "ptq":
        args += ["--output", str(tmp_path / "quantized.pt")]
    assert main(args) == 0
    capsys.readouterr()
    assert (tmp_path / "quantized.pt").is_file()
    assert torch.load(tmp_path / "quantized.pt", map_location="cpu", weights_only=True)


def test_resume_rejects_changed_effective_model_configuration(
    tmp_path, model_path, capsys
):
    data_path = prepare(tmp_path, model_path, capsys)
    config = configuration(model_path, data_path, tmp_path / "checkpoints")
    run_train_from_config(config)
    path = model_path / "config.json"
    model_config = json.loads(path.read_text(encoding="utf-8"))
    model_config["attention_dropout"] = 0.2
    path.write_text(json.dumps(model_config), encoding="utf-8")
    with pytest.raises(ValueError, match="objective"):
        run_train_from_config(config, resume_step=2)


def test_data_identity_is_checked_before_backend_state_is_loaded(
    tmp_path, model_path, capsys, monkeypatch
):
    from parascale.runtime.training.checkpointing import CheckpointController

    data_path = prepare(tmp_path, model_path, capsys)
    config = configuration(model_path, data_path, tmp_path / "checkpoints")
    run_train_from_config(config)
    calls = []
    original = CheckpointController._load_portable_backend_state

    def observe(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CheckpointController, "_load_portable_backend_state", observe)
    config["data"]["seed"] += 1
    with pytest.raises(ValueError, match="identity"):
        run_train_from_config(config, resume_step=2)
    assert not calls


def _distributed_resume_worker(rank, root, model_path, data_path):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    root = Path(root)
    torch.distributed.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        for label in ("continuous", "split"):
            config = configuration(model_path, data_path, root / label)
            config["parascale"]["training_backend"] = "native_ddp"
            config["hardware_profile"] = {"world_size": 2, "gpus_per_node": 2}
            config["training"]["max_steps"] = 4 if label == "continuous" else 2
            result = run_train_from_config(config)
            if label == "split":
                result = run_train_from_config(config, resume_step=2)
            assert result["global_step"] == 4
        if rank == 0:
            states = [
                torch.load(
                    root / label / "step-00000004" / "backend_state.pt",
                    map_location="cpu",
                    weights_only=True,
                )["backend_state"]
                for label in ("continuous", "split")
            ]
            for name, value in states[0]["model_state_dict"].items():
                torch.testing.assert_close(
                    states[1]["model_state_dict"][name], value, rtol=0, atol=0
                )
        from parascale.checkpoint import CheckpointManager
        from parascale.config import ParaScaleConfig
        from parascale.runtime.training import TrainEngine

        config = configuration(model_path, data_path, root / "split")
        config["data"]["seed"] += rank  # Only rank 1 has an incompatible input stream.
        model, optimizer, data, loss = build_training_components(config)
        engine = TrainEngine(
            ParaScaleConfig(training_backend="native_ddp"),
            objective_metadata={"workload": loss.objective_metadata},
        )
        before = {key: value.clone() for key, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with pytest.raises((ValueError, RuntimeError), match="identity|another rank"):
            engine.load_checkpoint(
                CheckpointManager(str(root / "split")),
                2,
                model=model,
                optimizer=optimizer,
                dataloader=data,
            )
        assert torch.equal(torch.get_rng_state(), rng)
        assert not optimizer.state
        assert data.position == 0
        for key, value in before.items():
            assert torch.equal(model.state_dict()[key], value)
    finally:
        torch.distributed.destroy_process_group()


def test_real_two_rank_causal_lm_checkpoint_resume(
    tmp_path, model_path, capsys, monkeypatch
):
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable")
    data_path = prepare(tmp_path, model_path, capsys)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    context = torch.multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_distributed_resume_worker,
            args=(rank, str(tmp_path), str(model_path), str(data_path)),
        )
        for rank in range(2)
    ]
    deadline = time.monotonic() + 120
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
