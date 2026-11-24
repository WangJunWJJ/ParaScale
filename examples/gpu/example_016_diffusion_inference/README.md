# Local Diffusion Generation

Generate PNG images with a local `StableDiffusionPipeline` through the product
CLI. This is single-device offline text-to-image inference, not a service, SDXL,
DiT or diffusion training. The prompts are reproducible examples, not a quality
evaluation dataset. No image-quality or throughput advantage is claimed here.

## Environment and Assets

Install `python -m pip install -e '.[diffusion]'` in a dedicated environment.
The optional extra pins Diffusers 0.35.1 and Transformers 4.51.3. CUDA compilation
also requires Torch 2.9.1; choose the CUDA wheel compatible with your driver.
The CPU functional path uses FP32 and no compilation. CUDA supports FP16/FP32.

Provide a trusted local Diffusers-format Stable Diffusion model directory with
`model_index.json`, tokenizer, scheduler, text encoder, UNet, VAE and any
configured safety components. All model components must have safetensors weights.
The tool does not download weights, load custom pipelines or override the model's
safety checker. A model without a safety checker remains explicitly marked as
such in execution metadata; this is not a content-safety certification.

## Run

```bash
export PARASCALE_MODEL=/data/models/stable-diffusion-v1-5
export PARASCALE_PROMPTS="$PWD/examples/gpu/example_016_diffusion_inference/prompts.jsonl"
export PARASCALE_RUN_DIR=/data/runs/diffusion-eager-01
CUDA_VISIBLE_DEVICES=0 python -m parascale.cli infer \
  --config examples/gpu/example_016_diffusion_inference/eager.json \
  --output "$PARASCALE_RUN_DIR/summary.json"
```

Each invocation needs a fresh image output directory. Inputs must be outside the
run directory. PNG filenames use request ordinals, never user-provided IDs.
`images/manifest.jsonl` records each ID, seed, filename, SHA256 and safety flag.
Failed jobs retain `manifest.jsonl.partial`; only a final manifest denotes success.
The summary retains no image arrays, so memory does not grow with request count.

Prompt and negative-prompt token overflow is rejected instead of truncated.
Seeds are `(seed + zero-based request ordinal) modulo 2**63`, with one CPU
generator per request. Warmup replays the first batch without consuming requests
or advancing their seeds. Grouping can still change floating-point numerics.

## Matched Optimization Comparison

Run `compiled.json` with a different `PARASCALE_RUN_DIR`, leaving all other
inputs unchanged. Only the UNet is compiled with `fullgraph=True`; failures
propagate instead of silently falling back. Scheduler, resolution, guidance,
precision and sampling budget remain the same. Keep native SDPA enabled in the
direct Diffusers baseline too; do not intentionally weaken the baseline.

Inspect model/input hashes and image differences before measuring speed. Report
`images_per_second`, peak allocated memory, setup/warmup time and job wall time.
Prediction time includes denoising, decoding, safety checking and CPU image
return; PNG encoding/writing is outside prediction time but inside job wall time.
Initial compilation occurs during warmup. A differently shaped last batch may
trigger further compilation during measurement; batch1 avoids this issue.

Use at least three paired repeats with fixed prompts/seeds and no other GPU
jobs. Reduced diffusion steps or a new scheduler are different algorithm/quality
experiments, not pure execution speedups. Production acceptance additionally
requires pinned pretrained weights and independent image-quality evaluation.
