# -*- coding: utf-8 -*-
# @Time : 2026/5/3 上午10:01
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

import importlib.util


def pytest_ignore_collect(collection_path, config):
    path = str(collection_path)
    if path.endswith("_no_torch.py"):
        return False
    if importlib.util.find_spec("torch") is None and path.endswith(".py"):
        return True
    return False
