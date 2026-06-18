# -*- coding: utf-8 -*-
"""
config.py — 讀寫 config.json（取代各腳本的「設定區」）
========================================================
所有預設值集中在專案根目錄的 config.json：
  theme   字型 + 顏色角色
  output  輸出資料夾 / 圖檔格式 / dpi
  methods 各方法被使用者「儲存為預設」的參數覆蓋值

設計重點：找不到檔或欄位時一律回退到內建預設，永遠不會因 config 壞掉而崩潰。
"""
import os
import json
import copy

from .theme import DEFAULT_THEME

# 專案根目錄 = .../資料分析（本檔位於 .../資料分析/pfas_toolkit/core/config.py）
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")


def default_config() -> dict:
    return {
        "theme": dict(DEFAULT_THEME),
        "output": {
            "output_dir": DEFAULT_OUTPUT_DIR,
            "image_format": "png",
            "dpi": 150,
        },
        "methods": {},   # {method_key: {param_key: value}}
    }


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    cfg = default_config()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user = json.load(f)
            cfg = _deep_merge(cfg, user)
        except Exception as e:
            print(f"[config] 讀取 {CONFIG_PATH} 失敗，改用預設值：{e}")
    # 輸出資料夾若留空 → 回退預設
    if not cfg["output"].get("output_dir"):
        cfg["output"]["output_dir"] = DEFAULT_OUTPUT_DIR
    return cfg


def save_config(cfg: dict) -> str:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return CONFIG_PATH
