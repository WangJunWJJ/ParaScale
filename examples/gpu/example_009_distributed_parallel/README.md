# Explicit Distributed Parallel Training

Run from the repository root with an isolated `parascale[parallel]` environment:

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 bash examples/gpu/example_009_distributed_parallel/run.sh dp2tp2
MODEL=llama_decoder CUDA_VISIBLE_DEVICES=0,1,3,4 bash examples/gpu/example_009_distributed_parallel/run.sh tp2pp2
```

The Llama fixture additionally requires Transformers (remote reference: 4.51.3).
The largest case uses four GPUs. Full three-dimensional DP2 x TP2 x PP2 is not
available under this resource limit. Models and inputs are synthetic diagnostics,
not pretrained quality benchmarks. Read [the API guide](../../../docs/distributed_parallel.md)
for supported combinations, checkpoint limitations and numerical validation.

## Integer Tokens and Ignored Labels

Run the token-loss contract validator in the same Torch 2.9.1 environment:

```bash
OUTPUT=$(mktemp -d /tmp/parascale-causal-loss.XXXXXX)
CUDA_VISIBLE_DEVICES=0,1,3,4 OMP_NUM_THREADS=1 \
python -m torch.distributed.run \
  --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=4 \
  tests/benchmarks/scripts/validate_parallel_causal_loss.py \
  --dp 1 --tp 2 --pp 2 --output "$OUTPUT/result.json"
```

Use `(dp,tp,pp)` values `(1,1,1)`, `(2,1,1)`, `(4,1,1)`, `(1,2,1)`,
`(1,1,2)`, `(2,2,1)`, `(2,1,2)` or `(1,2,2)`, with matching GPU count and
`--nproc_per_node`. The fixture has an embedding, two residual projection blocks
and an untied output head. It compares loss, logits, gradients, updated weights
and AdamW state against a full-batch reference across four optimizer updates.
It includes locally empty targets, uneven valid-token counts, one-rank invalid
arguments and inconsistent TP counts, checking that all ranks reject errors.

This is a synthetic correctness test, not a pretrained LM quality test, complete
Transformer TP/PP adapter or throughput benchmark. Keep its generated JSON and
logs outside commits. See the [token-loss contract](../../../docs/distributed_parallel.md#token-weighted-loss)
before adapting it to a real model.

## A6000 Batch Throughput

The example sweep uses four RTX A6000 GPUs, four blocks, width 512, sequence
length 128, FP32 with TF32 disabled, and one microbatch per optimizer step.
These are workload-specific diagnostic settings, not global defaults:

| Model fixture | Global batch | Batch per GPU | Microbatches |
| --- | ---: | ---: | ---: |
| `residual_mlp` | 1024 | 256 | 1 |
| `llama_decoder` | 512 | 128 | 1 |

Measure throughput and peak memory before selecting a batch size for your
environment. The validation peak includes a full reference model and is not
the standalone trainer's memory requirement.

From the repository root, in the isolated Torch 2.9.1 environment:

```bash
MODEL=llama_decoder
GLOBAL_BATCH=512
OUTPUT=$(mktemp -d /tmp/parascale-a6000-batch.XXXXXX)
CUDA_VISIBLE_DEVICES=0,1,3,4 OMP_NUM_THREADS=1 \
python -m torch.distributed.run \
  --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=4 \
  tests/benchmarks/scripts/validate_distributed_parallel.py \
  --model "$MODEL" --dp 4 --microbatches 1 \
  --width 512 --sequence-length 128 --global-batch "$GLOBAL_BATCH" \
  --steps 30 --warmup 10 --output "$OUTPUT"
```

For residual MLP, set `MODEL=residual_mlp` and `GLOBAL_BATCH=1024`. Use a new
output directory for each repetition. For a matched single-GPU comparison,
keep the global batch unchanged and use `CUDA_VISIBLE_DEVICES=0`,
`--nproc_per_node=1` and `--dp 1` together. Inside Docker, also limit the
container's `--gpus` device list; an environment variable is not a substitute
for container-level resource isolation.

The sweep tests global batches 32/64/128/256/512/1024 for both fixtures and
2048 for residual MLP. Candidate settings, batch-32 baselines and matched
single-GPU settings are repeated three times. The recorded rate covers only
forward/backward and loss reduction, not optimizer steps or real data loading.
Changing effective batch size can change convergence; these synthetic tests
do not establish training quality or time-to-accuracy.

Keep generated measurements, repeat dispersion and source/runtime provenance
in the chosen output directories, outside source-code commits.
