# -*- coding: utf-8 -*-
# @Time : 2026/6/22 下午1:55
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Communication planning and lightweight native-DDP optimization helpers."""

from .hooks import DdpHookPlan, recommend_ddp_hook
from .plan import CommunicationPlan, build_communication_plan
from .profiler import CommunicationProfile

__all__ = [
    "CommunicationPlan",
    "CommunicationProfile",
    "DdpHookPlan",
    "build_communication_plan",
    "recommend_ddp_hook",
]
