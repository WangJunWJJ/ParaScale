# PTQ与QAT验收

对应INF-02/03；当前`quantization.quality`返回BLOCKED，不自动执行未统一的质量协议。

复用[005 PTQ](../../gpu/example_005_llama_ptq/README.md)、[006 QAT](../../gpu/example_006_llama_qat/README.md)、[011量化产物推理](../../gpu/example_011_quantized_artifact_inference/README.md)。功能路径已存在，不意味着该验收编排已经接通。

正式流程需要独立校准/训练/评估划分、实际更新、完整导出及新进程重载、真实INT8内核证据、同预算FP16基线及质量门禁。参考反量化浮点执行不计为整数内核加速。
