# 生成推理

案例：`generative_inference.transformer`、`generative_inference.diffusion`，对应INF-06/07。

两个案例均通过`parascale-infer run`执行本地预训练模型，CUDA FP16、batch1、一次预热、eager执行。前者采用动态KV缓存和math attention；后者固定种子42、guidance7.5，由环境文件指定分辨率和步数。

需填写相应模型目录与提示词JSONL。案例限制请求数量和生成预算，检查真实执行标志、有限正吞吐、模型/输入身份及完整输出流，不允许synthetic替代。Transformer输出生成JSONL，diffusion输出PNG和manifest。

当前只取得功能与原始计时证据，尚未接入原生对照、至少三次配对重复和独立生成质量评估，因此acceptance模式为BLOCKED。不能用两步低分辨率diffusion功能验证宣称质量或速度达标。

已有配置及基线说明见[013因果生成](../../gpu/example_013_causal_lm_inference/README.md)、[016 Diffusion](../../gpu/example_016_diffusion_inference/README.md)。未更改既有benchmark或实验脚本。
