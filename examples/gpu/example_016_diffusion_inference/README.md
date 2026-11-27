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

### Batched Offline Generation

`batched.json` changes only `batch_size` from 1 to 8 and keeps eager execution.
Use a fresh output directory with the same model and prompts:

```bash
export PARASCALE_RUN_DIR=/data/runs/diffusion-batch8-01
CUDA_VISIBLE_DEVICES=0 python -m parascale.cli infer \
  --config examples/gpu/example_016_diffusion_inference/batched.json \
  --output "$PARASCALE_RUN_DIR/summary.json"
```

This is an offline throughput configuration, not a lower per-request latency
guarantee. Tune batch size against available VRAM and your model/resolution;
the runtime does not silently retry an OOM with a different batch budget.
Compare against direct Diffusers using the same batch size, precision and
sampling settings. Batching gains are also available to the native framework.
Do not expect bitwise-identical images across different batch sizes, even with
the same request seeds. Use within-batch native parity and separate quality
evaluation instead of interpreting cross-batch pixel changes as framework bugs.

### UNet Compilation

Compilation is an experimental, opt-in candidate. Passing small-component tests
does not establish pretrained-model fidelity. Floating-point changes can grow
over denoising steps; qualify images and safety decisions on your workload
before enabling it in production. Keep eager execution for strict reproduction.

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

### Four-Way Benchmark

From a source checkout with the environment above, run isolated native Diffusers
and ParaScale workers in both eager and compiled modes:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/benchmarks/scripts/benchmark_diffusion_inference.py \
  --model "$PARASCALE_MODEL" --prompts "$PARASCALE_PROMPTS" \
  --output /data/runs/diffusion-comparison-01 \
  --repeats 3 --requests 8 --height 512 --width 512 --steps 30 --batch-size 1
```

The output directory must not exist. Each repeat rotates execution order and
records model/input hashes, seeds, settings, dependency versions, GPU model,
compile-cache presence, timing and image manifests. Cache directories are
separate per mode. Preserve the complete output outside the source checkout.

Every four-mode repeat is checked before another repeat starts. Native/product
eager images must be identical; compiled images must meet PSNR >= 35 dB and
mean absolute error <= 1.5 in 8-bit pixel units, with unchanged safety flags.
A fidelity failure stops further repeats and retains the evidence. These
thresholds check execution fidelity, not semantic image quality.

`comparison.json` reports paired median speed ratios, including the optimized
native control. Initial performance gates require compiled/eager >= 1.10 and
ParaScale/native >= 0.95 for each execution mode. Negative results remain in
the report. Smaller budgets are labelled `smoke_only` and cannot establish a
production speedup; image fidelity still must pass. Compilation warmup and
subprocess wall time remain separate from steady prediction throughput.
