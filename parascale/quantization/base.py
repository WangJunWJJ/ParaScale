# -*- coding: utf-8 -*-
# @Time    : 2026/3/8 下午14:30
# @Author  : Jun Wang
# @File    : base.py
# @Software: ParaScale - A PyTorch Distributed Training Framework

"""
ParaScale 量化配置基类模块

本模块重新导出集中配置管理中的 QuantizationConfig，
确保所有量化配置使用统一的来源。
"""

from parascale.config import QuantizationConfig

__all__ = ["QuantizationConfig"]