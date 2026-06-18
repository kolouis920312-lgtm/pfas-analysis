# -*- coding: utf-8 -*-
"""
spec.py — 描述「一個分析方法」的資料結構
=================================================
這些 dataclass 讓 GUI 能用「同一套程式碼」處理所有方法：
  ‧ ParamSpec   一個可調參數（GUI 依 kind 自動產生對應的輸入元件）
  ‧ InputSchema 上傳資料的驗證規則
  ‧ MethodSpec  一個方法的完整描述（名稱、參數、規則、執行函式）
  ‧ OutputSettings / RunResult / ValidationReport 為執行/驗證的輸入輸出
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ParamSpec.kind 支援的型別：
#   "int"     整數      → 數字輸入框
#   "float"   浮點數    → 數字輸入框
#   "bool"    布林      → 勾選框
#   "choice"  下拉選單  → choices 提供選項
#   "column"  欄位選擇  → 下拉，選項來自上傳資料的欄名（optional=True 時可選「(無)」）
#   "columns" 多選欄位  → 核取清單，選項＝上傳資料的欄名；值為 list[str]；空 list＝全部
#   "values"  多選值    → 核取清單，選項＝某欄的相異值（由 source_col 指定是哪一欄）；
#                         值為 list[str]；空 list＝全部（常用來「挑要納入哪些站別/季節」）
#   "text"    文字      → 文字輸入框
#
# columns / values 的「空 list ＝ 全部」設計：使用者不勾就是沿用舊行為（全用），
# 完全向後相容；勾了才會縮成子集。
@dataclass
class ParamSpec:
    key: str
    label: str
    kind: str
    default: Any = None
    choices: list = field(default_factory=list)
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    help: str = ""
    optional: bool = False          # 僅對 "column"：允許留空 / 選「(無)」
    source_col: Optional[str] = None  # 僅對 "values"：相異值取自「哪個欄位型參數」選到的欄


@dataclass
class InputSchema:
    """上傳資料驗證規則。"""
    min_rows: int = 3                       # 最少樣本（列）數
    min_numeric_cols: int = 1               # 最少可用數值特徵欄數
    id_col_param: Optional[str] = None      # 哪個參數指定「ID 欄」
    required_param_cols: list = field(default_factory=list)  # 必填的欄位型參數 key
    check_bdl: bool = False                 # 是否檢查 BDL/零比例並對 >70% 警告
    note: str = ""                          # 額外備註（顯示在驗證面板）


@dataclass
class MethodSpec:
    key: str
    name: str
    summary: str
    params: list                            # list[ParamSpec]
    schema: InputSchema
    template_columns: list                  # 範本 CSV 的欄位（顯示用；實際範本由 make_demo 產生）
    uses_colors: list = field(default_factory=list)   # 用到哪些顏色角色（GUI 提示用）
    run: Optional[Callable] = None          # run(df, params, ctx) -> RunResult
    make_demo: Optional[Callable] = None    # make_demo() -> DataFrame（示範資料）
    manual: dict = field(default_factory=dict)   # 使用說明書 {"beginner": 白話, "pro": 原理算法}

    def default_params(self) -> dict:
        return {p.key: p.default for p in self.params}

    def get_param(self, key) -> Optional[ParamSpec]:
        for p in self.params:
            if p.key == key:
                return p
        return None


@dataclass
class OutputSettings:
    output_dir: str
    image_format: str = "png"               # png / svg / pdf / jpg
    dpi: int = 150
    theme: dict = field(default_factory=dict)   # font_family + 各顏色角色


@dataclass
class RunResult:
    figures: list = field(default_factory=list)   # 交付用圖檔（使用者選的格式）
    previews: list = field(default_factory=list)  # 給 GUI 預覽的 PNG
    tables: list = field(default_factory=list)     # 輸出的 CSV
    extras: list = field(default_factory=list)     # 其他檔案（如模型 json）
    log: str = ""
    summary: str = ""


@dataclass
class ValidationReport:
    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: list = field(default_factory=list)

    def as_text(self) -> str:
        lines = []
        for s in self.info:
            lines.append(f"ℹ {s}")
        for s in self.warnings:
            lines.append(f"⚠ {s}")
        for s in self.errors:
            lines.append(f"✘ {s}")
        if self.ok and not self.errors:
            lines.append("✔ 格式檢查通過，可以執行。")
        return "\n".join(lines) if lines else "（無資料）"
