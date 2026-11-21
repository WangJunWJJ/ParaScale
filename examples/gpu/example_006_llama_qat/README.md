# Llama QAT：训练、转换与重载

本案例调用 ParaScale QAT API，执行 fake quantization、真实反向传播和优化器更新，
冻结 observer 后转换导出，并验证重载。不是只插入 observer 的演示。
需要 PyTorch 和 Transformers 4.51.3，安装方式同 [PTQ 案例](../example_005_llama_ptq/README.md)。

## 离线功能验证

```bash
OUTPUT_DIR=runs/examples/llama_qat_cpu bash examples/gpu/example_006_llama_qat/run.sh \
  --tiny --bits 8 --qat-steps 16 --learning-rate 0.0001
```

`--tiny` 是随机初始化 Llama 与随机 token，仅证明参数更新和转换链路。
使用 `DEVICE=cuda` 切换到单 GPU，`--bits 4` 切换位宽。每次使用新输出目录。

## 预训练模型与真实文本

校准文本在本案例中也是 QAT 训练语料。评测文本必须独立，准备方法与容量要求
见 [PTQ 案例](../example_005_llama_ptq/README.md)。本地路径需替换为实际资产：

```bash
DEVICE=cuda OUTPUT_DIR=runs/examples/llama_qat_pretrained \
bash examples/gpu/example_006_llama_qat/run.sh \
  --model /models/llama --local-files-only --bits 4 \
  --calibration-text /datasets/text/train.txt \
  --evaluation-text /datasets/text/heldout.txt \
  --seqlen 128 --batch-size 1 --batches 16 \
  --qat-steps 100 --learning-rate 0.00001 --warmup 3 --repeats 10
```

学习率和训练步数只是起始配置，不能保证质量保持。报告会对照同一浮点基线，
但不会自动运行独立的 PTQ 实验；比较 PTQ/QAT 时应使用相同模型、语料和位宽。

## 结果与检查

- `report.json` 应为 `status: passed`，`qat.training` 包含真实梯度与权重更新证据。
- `qat` 和 `baseline` 包含评测 CE/困惑度、输出误差、执行时间与存储体积。
- `qat.roundtrip_max_abs_error` 验证转换模型重载一致性。
- `qat-4bit.pt` / `qat-8bit.pt` 是导出的量化权重，不是可续训的 optimizer checkpoint。

单设备 FP32 reference 路径，不支持 AMP、FSDP/DeepSpeed、分布式或 QAT+KD；
没有融合整数推理内核，不能仅凭文件变小宣称显存或速度收益。
该脚本是 API 验证入口，常规 workload 的 QAT 训练/恢复见[量化指南](../../../docs/quantization.md)。

PowerShell 对应入口：
`python tests/benchmarks/scripts/validate_llama_quantization.py --tiny --mode qat --bits 8 --qat-steps 16 --device cpu --output runs/examples/llama_qat_windows`。

## Observer 训练性能回归

比较两个代码版本的 observer 实现时，在同一空闲 GPU 上运行：

```bash
CUDA_VISIBLE_DEVICES=0 python tests/benchmarks/scripts/benchmark_qat_observer.py \
  --baseline-source /path/to/before-change-checkout \
  --widths 128 512 1024 2048 --steps 30 --warmup 5 --repeats 3 \
  --output /tmp/qat-observer-comparison.json
```

程序交替运行基线与当前实现，检查输入、训练后模型状态及梯度哈希一致，
记录包含 forward、backward、SGD 更新的步耗时。输出文件必须尚不存在。
这是固定随机 Llama 输入的热点诊断，不包含数据加载，也不代表预训练质量或
完整生产训练吞吐；不能替代上面的真实文本评测。生成结果无需提交到源码仓库。

实测记录见[统一测试报告](../../../tests/benchmarks/reports/BENCHMARK_REPORT.md)。
