# Quantized inference validation — 2026-11-21

Base main: ed2bb8235eaa41b54b136de83164bd436a353699; additional unmerged workspace changes.
Node 145 was reached on SSH port 360. The earlier port-22 blocker is resolved.

## Real Qwen2.5-7B-Instruct result

Single RTX A6000, torch 2.9.1+cu126, transformers 4.51.3, GemLite 0.6.0.post2,
Triton 3.5.1. BF16 activations, per-output-channel INT8 weights, static cache,
max-autotune compiled decode, batch 1, 128 prompt tokens + 32 greedy new tokens.
Separate processes; 10 warmups, 5 × 20 timed runs per variant.

| Metric | BF16 | INT8 |
|---|---:|---:|
| Median generation ms | 1614.869630 | 977.047881 |
| Peak allocated bytes | 15313387008 | 8754773504 |
| WikiText PPL | 17.298295897 | 17.318935603 |

Speedup 1.652805x; relative PPL increase 0.119316%. The engineering gate passes.
This is not contract acceptance, task accuracy, or a universal speed claim.
Quality: test-corpus tokens [4096,36864), 32512 predicted targets in nonoverlapping
128-token blocks. First 4096 tokens were exploratory and excluded from final quality.
Generation timing includes its own prefill, excludes loading/tokenization/compilation.
The separately reported prefill-forward metric is not static-cache time to first token.

Official weight revision: a09a35458c702b33eeacc393d103063234e8bc28;
all four weight SHA256s match the official Qwen/Qwen2.5-7B-Instruct repository.
Alternative optimized dynamic-cache BF16 measured 1616.740314 ms in exploration.
1.7B INT4 and INT8 trials did not establish a practical speedup; retain failures.

Raw acceptance samples, environment and hashes: `acceptance-int8-b1.json`.
Full raw experiment archive, source snapshots, failures and historical evidence:
Desktop `ParaScale量化优化原始证据_20261121.zip` with checked SHA256 manifest.
Existing local CPU logs remain historical; Linux rerun logs are in that archive.

Validation: Linux full suite 892 passed, 3 skipped. After the final precision-label validation fix, focused CUDA suite 36 passed. Full and focused logs are retained in the evidence ZIP.
