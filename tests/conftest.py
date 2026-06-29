# -*- coding: utf-8 -*-
# @Time : 2026/5/3 上午10:01
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import importlib.util


def pytest_ignore_collect(collection_path, config):
    path = str(collection_path)
    if (
        path.endswith("test_config_no_torch.py")
        or path.endswith("test_strategy_no_torch.py")
        or path.endswith("test_strategy_feedback_no_torch.py")
        or path.endswith("test_data_no_torch.py")
        or path.endswith("test_cli_no_torch.py")
        or path.endswith("test_core_architecture_no_torch.py")
    ):
        return False
    if importlib.util.find_spec("torch") is None and path.endswith(".py"):
        return True
    return False
