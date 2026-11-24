# -*- coding: utf-8 -*-
# @Time : 2026/11/24 下午9:18
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Diffusion job lifecycle plus optional real Diffusers numerical integration."""

import json
import os
import sys
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from tests.test_diffusion_inference_no_torch import config  # noqa: E402


def prepare(tmp_path, **options):
    (tmp_path / "model").mkdir()
    (tmp_path / "prompts.jsonl").write_text(
        '{"id":"../../outside","prompt":"a red cube"}\n'
        '{"id":"second","prompt":"a blue cube"}\n',
        encoding="utf-8",
    )
    result = config(
        tmp_path,
        precision="fp32",
        height=32,
        width=32,
        num_inference_steps=2,
        **options,
    )
    result["runtime"]["device"] = "cpu"
    return result


class Pipeline:
    """Only substitutes the external denoiser; product job and PNG I/O are real."""

    def __init__(self):
        self.calls = []
        self.scheduler = SimpleNamespace(config={"name": "fixed"})
        self.safety_checker = None
        self.tokenizer = SimpleNamespace(model_max_length=77)
        self.tokenizer = type(
            "Tokenizer",
            (),
            {
                "model_max_length": 77,
                "__call__": lambda self, prompts, **kw: {
                    "input_ids": [[0] * (80 if "long" in p else 5) for p in prompts]
                },
            },
        )()

    def __call__(self, **kwargs):
        seeds = [generator.initial_seed() for generator in kwargs["generator"]]
        self.calls.append(seeds)
        images = np.stack(
            [np.full((32, 32, 3), seed / 255, dtype=np.float32) for seed in seeds]
        )
        return SimpleNamespace(images=images, nsfw_content_detected=None)


@pytest.fixture
def pipeline(monkeypatch):
    from parascale.workloads import diffusion

    pipeline = Pipeline()
    monkeypatch.setattr(
        diffusion,
        "load_diffusion_pipeline",
        lambda spec: (pipeline, {"test_backend": True}),
    )
    return pipeline


def test_job_publishes_all_images_with_replay_stable_seeds(tmp_path, pipeline):
    from parascale.commands.run import run_inference_from_config

    result = run_inference_from_config(prepare(tmp_path, warmup_steps=2))
    assert pipeline.calls == [[42], [42], [42], [43]]
    assert result["metrics"]["images"] == result["metrics"]["requests"] == 2
    assert result["outputs"] == []
    assert result["output_stream"]["complete"] is True
    output = tmp_path / "run" / "images"
    records = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text().splitlines()
    ]
    assert [row["seed"] for row in records] == [42, 43]
    assert [row["file"] for row in records] == ["00000000.png", "00000001.png"]
    assert all(row["sha256"] for row in records)
    assert np.array(Image.open(output / records[0]["file"])).min() == 42
    assert not (output / "manifest.jsonl.partial").exists()
    json.dumps(result, allow_nan=False)


def test_cli_creates_separate_summary_parent_before_generation(tmp_path, pipeline):
    from parascale.cli import main

    data = prepare(tmp_path, warmup_steps=0)
    source = tmp_path / "source.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    summary = tmp_path / "reports" / "new" / "summary.json"
    assert main(["infer", "--config", str(source), "--output", str(summary)]) == 0
    result = json.loads(summary.read_text(encoding="utf-8"))
    assert result["output_stream"]["complete"] is True
    assert result["output_stream"]["records"] == 2


def test_invalid_summary_parent_rejects_before_generation(tmp_path, pipeline):
    from parascale.cli import main

    data = prepare(tmp_path)
    source = tmp_path / "source.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    parent = tmp_path / "reports"
    parent.write_text("existing file", encoding="utf-8")
    assert (
        main(
            ["infer", "--config", str(source), "--output", str(parent / "summary.json")]
        )
        == 2
    )
    assert not pipeline.calls
    assert not (tmp_path / "run").exists()


def test_failed_job_keeps_partial_manifest_and_closes_model(tmp_path, pipeline):
    from parascale.runtime.inference.diffusion import run_diffusion_inference
    from parascale.workloads.diffusion import build_diffusion_inference_components

    data = prepare(tmp_path, warmup_steps=0)
    with (tmp_path / "prompts.jsonl").open("a") as stream:
        stream.write("invalid-json\n")
    model, batches, _ = build_diffusion_inference_components(data)
    with pytest.raises(ValueError, match="Invalid prompt"):
        run_diffusion_inference(model, batches)
    assert model._pipeline is None and model._closed
    output = tmp_path / "run" / "images"
    assert not (output / "manifest.jsonl").exists()
    assert len((output / "manifest.jsonl.partial").read_text().splitlines()) == 2


def test_prompt_truncation_is_rejected(tmp_path, pipeline):
    from parascale.commands.run import run_inference_from_config

    data = prepare(tmp_path)
    (tmp_path / "prompts.jsonl").write_text('{"prompt":"long"}\n')
    with pytest.raises(ValueError, match="token|length"):
        run_inference_from_config(data)
    assert not pipeline.calls


@pytest.mark.parametrize("failure", ["nan", "shape", "count"])
def test_invalid_images_do_not_publish_success(tmp_path, monkeypatch, failure):
    from parascale.commands.run import run_inference_from_config
    from parascale.workloads import diffusion

    class Invalid(Pipeline):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            if failure == "nan":
                result.images[:] = np.nan
            elif failure == "shape":
                result.images = result.images[:, :16]
            else:
                result.images = np.concatenate([result.images, result.images])
            return result

    monkeypatch.setattr(
        diffusion, "load_diffusion_pipeline", lambda spec: (Invalid(), {})
    )
    with pytest.raises(ValueError, match="image"):
        run_inference_from_config(prepare(tmp_path, warmup_steps=0))
    assert not (tmp_path / "run" / "images" / "manifest.jsonl").exists()


@pytest.mark.parametrize("custom", ["external_library", "local_module"])
def test_custom_components_are_rejected_before_loading(tmp_path, monkeypatch, custom):
    from parascale.workloads.diffusion import load_diffusion_pipeline
    from parascale.workloads.specs.diffusion import DiffusionInferenceSpec

    data = prepare(tmp_path)
    model = tmp_path / "model"
    library = "custom_package" if custom == "external_library" else "diffusers"
    (model / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusionPipeline",
                "unet": [library, "UNet2DConditionModel"],
            }
        )
    )
    if custom == "local_module":
        (model / "unet").mkdir()
        (model / "unet" / "diffusers.py").write_text(
            "raise RuntimeError('must not import')"
        )

    def forbidden(*args, **kwargs):
        pytest.fail("custom component reached pipeline loader")

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            __version__="0.35.1",
            StableDiffusionPipeline=SimpleNamespace(from_pretrained=forbidden),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(__version__="4.51.3")
    )
    with pytest.raises(ValueError, match="custom|component"):
        load_diffusion_pipeline(DiffusionInferenceSpec.from_config(data))


@pytest.mark.parametrize(
    "device,compile_mode",
    [
        ("cpu", "off"),
        ("cuda:0", "off"),
        pytest.param(
            "cuda:0",
            "max-autotune",
            marks=pytest.mark.skipif(
                os.environ.get("PARASCALE_TEST_DIFFUSION_COMPILE") != "1",
                reason="explicit CUDA compiler qualification only",
            ),
        ),
    ],
)
def test_real_diffusers_pipeline_matches_native_pixels(tmp_path, device, compile_mode):
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.51.3" or diffusers.__version__ != "0.35.1":
        pytest.skip("requires the pinned parascale[diffusion] dependencies")
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    from parascale.commands.run import run_inference_from_config

    data = prepare(tmp_path)
    data["runtime"]["device"] = device
    data["inference"]["compile_mode"] = compile_mode
    if device.startswith("cuda"):
        data["inference"]["precision"] = "fp16"
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(
        str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt"), model_max_length=16
    )
    with torch.random.fork_rng():
        torch.manual_seed(123)
        pipeline = diffusers.StableDiffusionPipeline(
            vae=diffusers.AutoencoderKL(
                block_out_channels=(32, 64),
                down_block_types=("DownEncoderBlock2D",) * 2,
                up_block_types=("UpDecoderBlock2D",) * 2,
                latent_channels=4,
                norm_num_groups=8,
            ),
            text_encoder=CLIPTextModel(
                CLIPTextConfig(
                    vocab_size=3,
                    hidden_size=32,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    max_position_embeddings=16,
                )
            ),
            tokenizer=tokenizer,
            unet=diffusers.UNet2DConditionModel(
                sample_size=16,
                in_channels=4,
                out_channels=4,
                layers_per_block=1,
                block_out_channels=(32, 64),
                down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                cross_attention_dim=32,
                attention_head_dim=8,
                norm_num_groups=8,
            ),
            scheduler=diffusers.DDIMScheduler(
                num_train_timesteps=10, steps_offset=1, clip_sample=False
            ),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
    pipeline.save_pretrained(tmp_path / "model", safe_serialization=True)
    pipeline.to(
        device, dtype=torch.float16 if device.startswith("cuda") else torch.float32
    )
    (tmp_path / "prompts.jsonl").write_text('{"prompt":"a"}\n')
    result = run_inference_from_config(data)
    expected = pipeline(
        "a",
        height=32,
        width=32,
        num_inference_steps=2,
        guidance_scale=7.5,
        negative_prompt="",
        generator=torch.Generator().manual_seed(42),
        output_type="np",
    ).images[0]
    actual = np.array(Image.open(tmp_path / "run" / "images" / "00000000.png"))
    reference_pixels = (expected * 255).round().astype("uint8")
    if compile_mode == "off":
        assert np.array_equal(actual, reference_pixels)
    else:
        error = np.abs(actual.astype("int16") - reference_pixels.astype("int16"))
        assert error.max() <= 2 and error.mean() <= 0.1
    assert result["execution"]["model_assets_sha256"]
    assert result["execution"]["safety_checker_enabled"] is False
