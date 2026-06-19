# -*- coding: utf-8 -*-
"""
方法註冊表。新增方法只要：寫一個含 SPEC 的模組，再加進下面的 import 與 _MODULES。
"""
# 註：pfas_hca 已併入 hca（雙模式），不再註冊為獨立方法；hca.py 內部仍 import 其組成引擎。
from . import (pca, kmeans, hca, som, nonparam,
               xgboost_reg, bdl, coda, ml_drivers,
               trajectory, pmf, partitioning, lrtp)
from ._manuals import MANUALS, GLOSSARY, PARAM_MANUALS

# 顯示順序（後 4 個為大氣長程傳輸專用：受體模型 / 來源解析 / 程序模型）
_MODULES = [pca, kmeans, hca, som, nonparam, xgboost_reg, bdl, coda, ml_drivers,
            trajectory, pmf, partitioning, lrtp]

REGISTRY = {m.SPEC.key: m.SPEC for m in _MODULES}
ORDER = [m.SPEC.key for m in _MODULES]

# 掛上使用說明書（內容集中放在 _manuals.py，方法檔保持乾淨）
for _key, _man in MANUALS.items():
    if _key in REGISTRY:
        REGISTRY[_key].manual = _man

# 再掛上「參數詳解」分頁內容（manual["params"]）
for _key, _pman in PARAM_MANUALS.items():
    if _key in REGISTRY:
        if not isinstance(REGISTRY[_key].manual, dict):
            REGISTRY[_key].manual = {}
        REGISTRY[_key].manual["params"] = _pman


def all_specs():
    return [REGISTRY[k] for k in ORDER]


def get(key):
    return REGISTRY[key]
