# Fused INT8 and INT4 inference

## Experimental INT8 path

**Not production-qualified; do not merge or deploy the current artifact-fusion
increment until its GPU numerical failures are resolved.** On A6000 with
Torch 2.9.1 / Triton 3.5.1 / GemLite 0.5.1, the stored PTQ/QAT INT8 matrix
tests fail for `K=576, N=1536, M=32`, with and without bias. Relative L2 error
is approximately 0.329-0.331 against the reference result (limit: 0.01).
The current dimension admission check does not protect against this kernel
tail-handling issue. Other passing shapes and successful dependency resolution
do not establish general correctness. No end-to-end artifact performance
qualification is available for this tuple.

The experimental `prepare_quantized_inference_model` API consumes converted or
reloaded canonical PTQ/QAT INT8 layers, preserving their stored integer codes
and FP32 scales instead of quantizing floating-point weights again. Its
benchmark entry point accepts `--reference-artifact`; the BF16 control
dequantizes those same weights once before timing. Keep this path experimental
until the numerical regression matrix and matched model quality/performance
gates pass. The fresh-floating-point INT8 path uses the same kernel backend
and must also be requalified.

The opt-in `InferenceQuantizationConfig(backend="gemlite_int8")` uses channelwise
INT8 weights and BF16 activations, with GemLite 0.5.1 / Triton 3.5.1 /
PyTorch 2.9.1 on Linux CUDA. Install `parascale[inference-int8]` in an isolated
environment. `group_size` is an INT4-only option; leave its default for INT8.
Set the model device and BF16 dtype **before** conversion. Do not cast the
converted model's FP32 scales to BF16. Shared weights, output heads, hooks, and
native transformer modules that read weights functionally are excluded.
Custom parents must call `Linear.forward`; direct floating-point `.weight`
access is unsupported for INT8. No float reconstruction fallback is supplied.

Use a separate Linux environment; do not update a shared training runtime:

```bash
python3 -m venv .venv-int8
source .venv-int8/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[inference-int8]' transformers==4.51.3
python -m pip check
```

GemLite 0.6.0, 0.6.0.post1 and 0.6.0.post2 require Triton >=3.6.0,
which conflicts with Torch 2.9.1's Linux x86-64 Triton 3.5.1 requirement.
Do not bypass dependency resolution with `--no-deps`. The 0.5.1 helper requires
explicit `fp32_scale=True` to retain FP32 scales. Packed kernel tensors are
registered, but standalone GemLite checkpoint loading is not a supported
deployment artifact contract. Keep canonical source artifacts and reconstruct
the runtime representation instead.

CI resolves both inference extras with `--dry-run --ignore-installed`.
Resolver success and `pip check` do not establish CUDA numerical correctness
or speed. Measurements from a different GemLite release do not qualify this
tuple; repeat matched quality, dispatch and performance validation.

Dependency validation on Linux Python 3.10.12 resolved both extras with installed
packages ignored. A separate GemLite 0.5.1 target, installed with normal dependency
resolution and the existing runtime's NumPy/tqdm versions, imported successfully;
full `pip check` passed with that target on `PYTHONPATH`. This was an overlay on
Torch 2.9.1+cu126 / Triton 3.5.1, not a fresh-venv installation qualification or
a GemLite 0.5.1 GPU correctness/performance result.

The Qwen2.5-7B-Instruct measurements and 1.7B trials described below used the
previous runtime, not the GemLite 0.5.1 tuple above. They are historical evidence
and do not establish performance or quality qualification for GemLite 0.5.1.

To reproduce the historical Qwen2.5-7B-Instruct setup, configure generation with
`cache_implementation="static"` and
`CompileConfig(mode="max-autotune", fullgraph=True)`. Both compared variants
must use the same settings. Compile costs are startup costs, not warm latency.
The 1.7B trials did not establish a practical speedup; do not generalize the
7B result to small models, every batch size, vision models, or embodied tasks.

```bash
python tests/benchmarks/scripts/benchmark_quantized_inference.py \
  --model /path/to/Qwen2.5-7B-Instruct --data /path/to/heldout.txt \
  --data-description 'Dataset, split and revision' \
  --backend gemlite_int8 --static-cache --compile \
  --eval-offset 4096 --eval-tokens 32768 --warmup 10 --iterations 20 --repeats 5 \
  --output runs/int8/acceptance.json
```

The first 4096 WikiText tokens were used for exploration; the subsequent fixed
32768-token window is reserved for validation. PPL is a language-model quality
metric, not task accuracy. See the dated benchmark report for actual results.

## Experimental INT4 path

The previous `QuantizedLinear` stores packed weights but dequantizes the entire
matrix on every forward before calling floating-point `linear`. It remains the
reference PTQ/QAT implementation and keeps its checkpoint contract.

The new opt-in inference adapter uses TorchAO's tile-packed INT4 weight format
and `aten._weight_int4pack_mm`. Dequantization is fused with multiplication;
activations remain BF16. This is weight-only INT4 inference, not INT4 activation
training or an all-integer model. Conversion alone does **not** prove a speedup.

## Scope and dependencies

- One CUDA GPU, compute capability 8.0 or later; BF16 model in eval mode.
- Exact `torch.nn.Linear` layers with input dimension divisible by 1024 and
  output dimension divisible by 8. This avoids TorchAO's activation padding
  overhead. Small or unaligned layers remain floating point and are reported.
- Shared parameters/modules and `lm_head` are excluded. Custom Linear subclasses,
  Conv2d, existing reference quantized layers and distributed/sharded models are
  not converted by this adapter.
- PyTorch **2.9.1**, torchvision **0.24.1**, TorchAO **0.15.0**. The optional
  `inference-int4` extra pins this combination. Use a separate environment;
  do not upgrade the node's shared training environment.
- CUDA dispatch and numerical regression tests have passed on node 145 using
  port 360 and an isolated RTX A6000. End-to-end performance and quality must
  still pass together for the specific workload; conversion is not acceptance.

The packing format and version choice follow the
[TorchAO implementation](https://github.com/pytorch/ao/blob/v0.15.0/torchao/quantization/quantize_/workflows/int4/int4_tile_packed_to_4d_tensor.py)
and [official compatibility matrix](https://github.com/pytorch/ao/issues/2919).

Example isolated Linux environment, after verifying driver compatibility:

```bash
python3 -m venv .venv-int4
source .venv-int4/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install torchao==0.15.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[inference-int4]' transformers==4.51.3 pytest
```

## Runtime use

```python
import torch
from parascale import InferenceEngine
from parascale.quantization import InferenceQuantizationConfig, quantize_inference_model

# model is a loaded floating-point model, not a reference INT4 checkpoint.
model = model.eval().to(device="cuda", dtype=torch.bfloat16)
model, report = quantize_inference_model(
    model,
    InferenceQuantizationConfig(group_size=128, exclude_modules=("lm_head",)),
    inplace=True,
)
engine = InferenceEngine().load_model(model)
print(report)
```

`inplace=False` (default) copies the model and preserves the source. This needs
extra memory during conversion. `inplace=True` avoids that full copy; a backend
error can leave partial conversion, so reload the float checkpoint after failure.
No silent float fallback is allowed for selected layers. A model with zero
eligible layers fails explicitly. The report includes converted/skipped layers
and storage bytes for converted weights only; these are not total GPU memory.

The adapter packs weights once, then stores packed buffers in `PackedInt4Linear`
and calls the fused ATen operation directly. This avoids repeated TorchAO tensor
dispatch and activation padding. Layers with execution hooks remain unchanged.
The read-only `weight` property exposes a packed TorchAO view for `F.linear`;
this compatibility path retains TorchAO dispatch overhead. Training and arbitrary
floating-point weight mutation after conversion are unsupported.

Run direct model calls under `torch.inference_mode()`. ParaScale's InferenceEngine
already supplies the inference execution context. Optional `torch.compile` is
left to the caller and must also be applied to the BF16 comparison. Reapply the
adapter when loading the original pretrained checkpoint; the reference PTQ/QAT
exporter and loader do not serialize this new backend.

## Real-model performance and quality test

Use a local **pretrained** causal model and a documented held-out UTF-8 corpus.
Do not substitute random model perplexity for task accuracy. The script uses
`local_files_only=True` and does not download weights or run remote model code.

```bash
python tests/benchmarks/scripts/benchmark_quantized_inference.py \
  --model /path/to/pretrained-model \
  --data /path/to/heldout.txt \
  --data-description 'Dataset name, version, source, held-out split' \
  --batch-size 1 --sequence-length 128 --new-tokens 32 \
  --eval-tokens 32768 --warmup 10 --iterations 20 --repeats 5 \
  --output runs/int4/model-b1-eager.json
```

Repeat with `--batch-size 4` and separately with `--compile` (different output
paths). Both BF16 and INT4 get identical settings. `compare` runs them in separate
processes, preventing a second model's allocation from contaminating memory
measurements. Run on an idle GPU and repeat comparisons with reversed order when
investigating thermal/clock drift. Current order is BF16 then INT4.

Use `--group-size 32 --exclude-modules lm_head self_attn` to evaluate selective
quantization. It is a measured candidate, not a universal accuracy guarantee.
Use `--eval-offset 4096` to reserve a disjoint token window after exploratory
evaluation of the first 4096 tokens. Both variants use the same recorded offset.
Compilation uses `max-autotune-no-cudagraphs`: dynamic generation caches must
remain valid across forward calls. Cold compilation is recorded separately.

Each result retains checkpoint file SHA256s, corpus SHA256, source hashes, package
versions, GPU/driver, seed, shapes, raw wall/CUDA timing samples, per-repeat medians,
P95, peak allocated/reserved GPU bytes, load/conversion/warmup times and token NLL.
The generation measurement includes prefill and fixed-length greedy generation;
it is **not** isolated decode latency. Tokenization and host-to-device input copy
are outside warm timings. Perplexity uses identical non-overlapping context
blocks and omits each block's first token. Generated token IDs are also retained.

Provisional engineering gate: median fixed-length generation speedup **>= 1.2x**,
relative perplexity increase **<= 5%**, nonzero conversion coverage and observed
fused kernel dispatch. At least 20 timing samples and matching model/data/runtime
metadata are required. This gate is not a contract criterion and does not prove
VLM/embodied task accuracy, all batch-size speedups, or online service latency.
Keep slower results as evidence; do not present only favorable shapes.

## Validation commands

```bash
python -m pytest tests/test_quantized_inference.py tests/test_quantized_inference_config_no_torch.py tests/test_quantized_inference_report_no_torch.py -q
python tests/run_tests.py
```

The CUDA test profiles dispatch, checks numerical error and storage reduction,
and verifies that the original model is preserved. It is skipped when CUDA is
unavailable. The real-model benchmark is a separate acceptance step.
