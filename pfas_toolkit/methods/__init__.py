# -*- coding: utf-8 -*-
"""
方法註冊表。新增方法只要：寫一個含 SPEC 的模組，再加進下面的 import 與 _MODULES。
"""
from . import (pca, kmeans, hca, som, nonparam,
               xgboost_reg, bdl, coda, ml_drivers,
               trajectory, pmf, partitioning, lrtp)

# 顯示順序（後 4 個為大氣長程傳輸專用：受體模型 / 來源解析 / 程序模型）
_MODULES = [pca, kmeans, hca, som, nonparam, xgboost_reg, bdl, coda, ml_drivers,
            trajectory, pmf, partitioning, lrtp]

REGISTRY = {m.SPEC.key: m.SPEC for m in _MODULES}
ORDER = [m.SPEC.key for m in _MODULES]


def all_specs():
    return [REGISTRY[k] for k in ORDER]


def get(key):
    return REGISTRY[key]
