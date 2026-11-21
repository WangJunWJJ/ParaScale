# Llama PTQ：校准、导出与重载

本案例运行真实 Llama 网络和 ParaScale PTQ API，不是 dry-run。计算使用
FP32 reference 路径，支持 4/8-bit 权重存储；压缩文件不代表整数算子加速。
启动脚本复用仓库验证程序，不注册新的 Llama CLI workload。

## 准备

从仓库根目录运行。需要已安装匹配 CPU/CUDA 的 PyTorch，以及：

```bash
python -m pip install -e . transformers==4.51.3
```

## 离线功能验证

```bash
OUTPUT_DIR=runs/examples/llama_ptq_cpu bash examples/gpu/example_005_llama_ptq/run.sh \
  --tiny --bits 8
```

`--tiny` 显式使用随机初始化的两层 Llama 和随机 token；只验证计算、校准和序列化，
不验证预训练模型精度。GPU 运行将 `DEVICE=cuda` 加在命令前；4-bit 使用 `--bits 4`。
每次使用新的输出目录，避免复用或覆盖已有结果。

## 预训练模型与真实文本

准备本地 Hugging Face 格式的 Llama 模型、tokenizer、两份独立 UTF-8 文本。
下面的路径是需替换的输入，不会由案例自动生成或下载：

```bash
DEVICE=cuda OUTPUT_DIR=runs/examples/llama_ptq_pretrained \
bash examples/gpu/example_005_llama_ptq/run.sh \
  --model /models/llama --local-files-only --bits 4 \
  --calibration-text /datasets/text/calibration.txt \
  --evaluation-text /datasets/text/heldout.txt \
  --seqlen 128 --batch-size 1 --batches 16 --warmup 3 --repeats 10
```

不要同时传入 `--tiny`。两份文本各至少提供 `batches * batch_size * seqlen`
个 token；程序拒绝相同路径、相同文件内容或重复的训练/评测 token batch。
正式质量评估还需自行保证语料无语义重叠，固定模型版本及数据划分。
默认排除 `lm_head`，仅转换精确类型的 Linear/Conv2d。

## 固定版本的轻量预训练案例

可使用 [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2)
验证完整预训练模型，而非随机网络。它采用 Llama 架构；共享的 embedding / `lm_head`
保持浮点表示。以下准备步骤需要网络，资产目录应位于源码仓库之外：

```bash
python -m pip install pyarrow
export ASSET_DIR=/tmp/parascale-compression-assets
python - <<'PY'
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

root = Path(os.environ["ASSET_DIR"])
snapshot_download(
    "HuggingFaceTB/SmolLM2-135M",
    revision="93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    local_dir=root / "smollm2-135m",
    allow_patterns=["*.json", "*.txt", "*.safetensors", "README.md"],
)
source = Path(snapshot_download(
    "Salesforce/wikitext", repo_type="dataset",
    revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    allow_patterns=["wikitext-2-raw-v1/*.parquet"],
)) / "wikitext-2-raw-v1"
target = root / "wikitext"
target.mkdir(parents=True, exist_ok=True)
for split in ("train", "validation", "test"):
    rows = pq.read_table(source / f"{split}-00000-of-00001.parquet")["text"]
    (target / f"{split}.txt").write_text(
        "\n".join(rows.to_pylist()), encoding="utf-8"
    )
PY

DEVICE=cuda OUTPUT_DIR=runs/examples/smollm2_ptq8 \
bash examples/gpu/example_005_llama_ptq/run.sh \
  --model "$ASSET_DIR/smollm2-135m" --local-files-only \
  --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 --bits 8 \
  --calibration-text "$ASSET_DIR/wikitext/train.txt" \
  --evaluation-text "$ASSET_DIR/wikitext/validation.txt" \
  --seqlen 128 --batch-size 1 --batches 32 --warmup 5 --repeats 20
```

保留原始行序，不混合数据划分。`validation` 用于调参；确定配置后再使用 `test.txt`
做独立评估。此命令仅评估前 32 个固定长度 batch，不是完整 WikiText 排行榜成绩。
同资产的 QAT 命令见 [QAT 案例](../example_006_llama_qat/README.md)。

## 结果与检查

- `report.json`：应为 `status: passed`；检查 `model_source.kind` 区分随机/预训练来源。
- `baseline` / `ptq`：评测 CE、困惑度、相对基线输出误差、文件体积、延迟和显存。
- `ptq.roundtrip_max_abs_error`：导出重载一致性，而非量化前后精度差。
- `ptq-4bit.pt` 或 `ptq-8bit.pt`：量化产物，重载需要匹配的浮点架构模板。

出现缺依赖、无 CUDA、文本不足或不支持的共享层时应修正输入，不要绕过检查。
这不是训练恢复检查点，也不能直接送入 ParaScale `serve`。
`status: passed` 表示执行、有限值和重载检查通过，不表示满足业务质量或性能门槛。
reference 路径每次前向都解量化权重，可能比浮点模型更慢；部署加速需单独验证融合内核路径。

Windows 可在 Git Bash 中执行同一命令；PowerShell 可直接执行
`python tests/benchmarks/scripts/validate_llama_quantization.py --tiny --mode ptq --bits 8 --device cpu --output runs/examples/llama_ptq_windows`。

更多接口与限制见[量化指南](../../../docs/quantization.md)，实测记录见
[统一测试报告](../../../tests/benchmarks/reports/BENCHMARK_REPORT.md)。
