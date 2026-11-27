# 并行训练与恢复

案例：`parallel_training.single/dp2/dp4/tp2/pp2/dp2_tp2/dp2_pp2/tp2_pp2`。每个名称是单独的`--case`值，不包含斜杠。使用现有`native_parallel`执行器、真实本地Llama和token corpus，不构造随机模型。

流程：生成配置，`torchrun --no-python parascale-train run`完成指定更新数并保存；退出进程后重新启动相同布局，以`--resume-latest`继续同样更新数。检查非有限数值、最终检查点、起止步数及有效token吞吐。保持全局batch固定，累积步数为`global_batch / (DP * microbatch)`，不可整除则配置失败。

TP/PP仅覆盖现有执行器支持的Llama结构。`--plan`和实际执行共享轻量预检：只读取本地`config.json`，不导入torch/transformers、不加载权重；模型类型、正整数维度、TP可整除性、PP层数与绑定权重、attention dropout及pretraining_tp不满足约束时返回明确的`BLOCKED`。配置缺失或损坏也会阻断，且不会计算大型资产哈希或启动训练进程。

元数据预检不是权重完整性或运行正确性认证；实际权重、算子、精度和检查点仍由产品运行时校验。不自动修改模型结构或替换模型。这里的数据并行是native_parallel的DP模式，不等于已对native_ddp/FSDP/DeepSpeed逐一验收。

选择模型时，TP2要求hidden_size、intermediate_size、注意力头数和KV头数均能被2整除；PP2要求至少两层且不跨阶段绑定embedding/head。SmolLM2-135M的9个注意力头、3个KV头以及绑定权重不满足这两种布局，不能因为DP2通过就复用到TP2/PP2。更换模型后必须用新模型自己的tokenizer重新运行`parascale-train prepare-text`，不能复用不匹配的token corpus。

`parallel_training.dp2_tp2_pp2`需要8卡，在本轮最多4卡配置下始终BLOCKED。不能以二维布局代替。

同布局恢复步数通过不等于参数/优化器/RNG逐位相等。正式质量、原生对照、三组配对重复和严格恢复等价仍未接入本入口。已有更细的验证器和原生基线见[012并行训练](../../gpu/example_012_native_parallel_training/README.md)、[010 Llama并行](../../gpu/example_010_llama_parallel/README.md)；后续复用这些实现，不在examples另写训练循环。

正式恢复验收须使用持久化输出目录，并记录存储介质。临时内存盘只能验证同一运行环境内的新进程保存/恢复，不能证明容器销毁或机器断电后的持久恢复；其检查点吞吐也不能与磁盘结果直接比较。

启动器为单机静态rendezvous，端口29507；套件串行执行，不支持同一网络命名空间同时启动多份套件。
