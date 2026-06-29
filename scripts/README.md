# ParaScale 脚本目录

ParaScale 的用户入口应优先收敛到：

```bash
python -m parascale.cli <command>
```

当前根目录脚本只保留通用辅助工具：

- `package_remote.ps1`：本地打包并辅助远程同步。

测试和 benchmark 相关资产统一放入 `tests/`：

- `tests/benchmarks/scripts/`：阶段性验证 shell 脚本。
- `tests/benchmarks/tools/`：历史 benchmark 聚合、报告生成和数据缓存工具。
- `tests/benchmarks/reports/`：后续新增 benchmark 输出报告。

这些文件保留用于追溯和复现实验，但不作为新用户入口。后续新能力应优先接入 `parascale.cli`，避免继续增加一次性脚本。
