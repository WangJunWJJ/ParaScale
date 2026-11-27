# 产品入口检查

案例：`installation.entries`，对应REL-01。

在源码目录外依次调用两个已安装工具的`--version`和`--help`，验证工具名称及版本一致。无需GPU和模型。使用根目录README中的`--mode functional`运行。

这不是全新环境安装验证：当前环境中的命令存在不能证明依赖包可离线安装。正式REL-02仍需要可复现介质和新环境测试。已有安装闭环验证见`tests/packaging/verify_clean_install.py`和`verify_product_dependencies.py`及CI的product-install矩阵。
