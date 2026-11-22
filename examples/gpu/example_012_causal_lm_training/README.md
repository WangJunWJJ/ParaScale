# Causal-LM Training Through the Product CLI

This example uses the ordinary ParaScale workload, optimizer, native/DDP runtime
and checkpoint paths. It does not execute a separate benchmark training engine.
Use a local TinyLlama or SmolLM2 model directory including its matching fast
tokenizer, and a separate UTF-8 training file. Install the existing `vlm` extra
for Transformers/tokenizers. No remote model code or implicit download is used.

## Prepare Once

```bash
export PARASCALE_MODEL=/data/models/tinyllama-1.1b
export PARASCALE_TOKEN_CORPUS=/data/tokens/wiki-train
export PARASCALE_RUN_DIR=/data/runs/tinyllama-native
python -m parascale.cli prepare-text \
  --input /data/wikitext/train.txt --tokenizer "$PARASCALE_MODEL" \
  --output "$PARASCALE_TOKEN_CORPUS"
```

The output directory must not exist. Each nonempty stripped UTF-8 line is one
document, tokenized without automatic special tokens and terminated by one EOS.
Lines over 1 MiB are rejected, not truncated. Payload and source hashes, tokenizer
identity and packing policy are recorded in the manifest. This tokenization is
not identical to tokenizing the complete raw file as one string; fair baselines
must use the same prepared tokens. Do not prepare validation/test text as training
data. Keep model, corpus and run outputs outside Git.

## Train and Resume

```bash
CUDA_VISIBLE_DEVICES=0 NPROC=1 bash examples/gpu/example_012_causal_lm_training/run.sh
CUDA_VISIBLE_DEVICES=0 NPROC=1 bash examples/gpu/example_012_causal_lm_training/run.sh --resume-latest
```

For DDP, use `NPROC=2` or `NPROC=4`, matching visible GPU count, and a fresh
`PARASCALE_RUN_DIR` for each experiment. The script selects `native_ddp` for these
sizes. Resume with the same topology, corpus, data seed, batch and model config.
`max_steps=100` means 100 additional optimizer updates per invocation; batch size
4 is per rank and gradient accumulation is 4. These defaults are a starting
configuration, not an A6000 throughput recommendation established by measurement.

Data reads use a read-only memory map, without DataLoader workers. Blocks are
disjoint across DP ranks; incomplete global batches are dropped equally. Fixed
windows of at most 65,536 blocks are shuffled internally with PyTorch's sampler.
Window order stays fixed: this is **not global corpus shuffling**. Reduce or
increase `data.shuffle_window_blocks` to trade mixing against index memory; it
must cover a global microbatch. Cursor restore rebuilds only its current window,
not all preceding tokenization, samples or epoch indices.

Checkpoint preflight binds effective HF model configuration, attention/loss
policy and each rank's corpus/tokenizer/data topology. An incompatible rank
causes all ranks to reject restoration before model, optimizer or RNG state is
applied. Completed data consumption, not prefetch position, is checkpointed.
The prepared files must remain immutable throughout training.

## Compression and Qualification

The `causal_lm` workload also works with the existing `quantize` PTQ command and
`train` QAT path. Their existing constraints remain: native single process;
QAT uses FP32 and full checkpoints. Configure `parascale.quantization` explicitly,
and use the original PTQ/QAT examples for compression settings and independent
quality evaluation. Reference quantization does not itself promise fast inference.

Local regression uses an offline small Llama with real Transformers execution,
including PTQ/QAT export, continuous-versus-resumed native and two-process Gloo DDP
training, and coordinated rejection when one rank changes its data identity.
This is functional evidence, not pretrained quality or GPU performance acceptance.
Real A6000 1/2/4-card throughput, held-out quality, FSDP/DeepSpeed qualification,
TP/PP product-CLI integration, configured scheduler support and long-run tests
remain separate acceptance work. CUDA prefetch defaults off in this example
until the outstanding real-stream stress tests are completed.
