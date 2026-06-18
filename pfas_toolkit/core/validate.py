# -*- coding: utf-8 -*-
"""
validate.py — 上傳資料驗證器（執行前先檢查格式，避免白跑）
==============================================================
依各方法的 InputSchema + 參數，檢查上傳的 DataFrame：
  ‧ 樣本數 / 數值特徵數是否足夠
  ‧ 指定的 ID 欄 / 必填欄（如 target）是否存在
  ‧ 非數值特徵欄（常見是把 BDL 寫成 'ND'、'<MDL'）→ 明確指出哪欄、怎麼改
  ‧ 缺值數量、BDL/零比例（>70% 對 CLR/SOM 敏感）
回傳 ValidationReport：ok / errors / warnings / info，GUI 直接顯示。
"""
import numpy as np
import pandas as pd

from .spec import ValidationReport

# 常見「應該是數字卻被寫成文字」的 BDL 記號
_BDL_TOKENS = ("nd", "n.d.", "bdl", "<mdl", "<lod", "<dl", "未檢出", "na", "n/a")


def _looks_like_bdl_text(series: pd.Series) -> bool:
    vals = series.dropna().astype(str).str.strip().str.lower().unique()
    return any(any(tok in v for tok in _BDL_TOKENS) for v in vals[:50])


def validate(df: pd.DataFrame, spec, params: dict) -> ValidationReport:
    errors, warnings, info = [], [], []

    if df is None or df.shape[1] == 0:
        return ValidationReport(False, ["讀不到任何欄位，請確認 CSV 格式（第一列需為欄名）。"])

    n = len(df)
    info.append(f"樣本數(列) {n}；總欄位 {df.shape[1]}")
    if n < spec.schema.min_rows:
        errors.append(f"樣本數 {n} 少於此方法最低需求 {spec.schema.min_rows}。")

    # 找出「特殊欄」（ID、必填欄、所有欄位型參數選到的欄）→ 不計入數值特徵
    special = set()

    id_col = None
    if spec.schema.id_col_param:
        id_col = params.get(spec.schema.id_col_param)
        if id_col and id_col not in ("(無)", "", None):
            if id_col not in df.columns:
                errors.append(f"找不到指定的 ID 欄『{id_col}』。")
            else:
                special.add(id_col)

    for pk in spec.schema.required_param_cols:
        col = params.get(pk)
        label = pk
        sp = spec.get_param(pk)
        if sp:
            label = sp.label
        if not col or col in ("(無)", "", None):
            errors.append(f"必須指定欄位：{label}。")
        elif col not in df.columns:
            errors.append(f"找不到欄位『{col}』（{label}）。")
        else:
            special.add(col)

    for p in spec.params:
        if p.kind == "column":
            col = params.get(p.key)
            if col and col not in ("(無)", "", None) and col in df.columns:
                special.add(col)

    # 候選特徵欄 = 全部欄位 - 特殊欄
    feat_cols = [c for c in df.columns if c not in special]
    numeric_cols = df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in feat_cols if c not in numeric_cols]

    # 多選參數（columns＝挑特徵欄、values＝挑某欄要保留哪些值）的提早檢查
    from .prep import as_list
    for p in spec.params:
        if p.kind == "columns":
            sel = as_list(params.get(p.key))
            if not sel:
                continue
            miss = [c for c in sel if c not in df.columns]
            if miss:
                warnings.append(f"「{p.label}」選到的欄在資料中不存在，將被略過：{miss}。")
            sel_numeric = [c for c in sel if c in numeric_cols]
            if sel_numeric:
                # 選了特徵子集 → 用子集數量檢查是否達最低需求
                numeric_cols = sel_numeric
                info.append(f"已選定 {len(sel_numeric)} 個特徵欄分析。")
        elif p.kind == "values":
            src = params.get(p.source_col) if p.source_col else None
            sel = as_list(params.get(p.key))
            if sel and src and src in df.columns:
                have = set(df[src].astype(str).str.strip().unique())
                miss = [v for v in sel if v not in have]
                if miss:
                    warnings.append(f"「{p.label}」選到的值在「{src}」欄中找不到：{miss}。")
                info.append(f"「{p.label}」將只保留：{[v for v in sel if v in have]}。")

    if non_numeric:
        bdl_cols = [c for c in non_numeric if _looks_like_bdl_text(df[c])]
        if bdl_cols:
            errors.append(
                f"這些欄位含非數字（疑似把 BDL 寫成 ND/<MDL 等文字）：{bdl_cols}。"
                "請改用 0 或留空白表示未檢出。")
        other = [c for c in non_numeric if c not in bdl_cols]
        if other:
            warnings.append(f"非數值欄將被忽略（不納入分析）：{other}。")

    if len(numeric_cols) < spec.schema.min_numeric_cols:
        errors.append(
            f"可用數值特徵只有 {len(numeric_cols)} 欄，少於最低需求 "
            f"{spec.schema.min_numeric_cols} 欄。")
    else:
        info.append(f"可用數值特徵 {len(numeric_cols)} 欄")

    if numeric_cols:
        nan = int(df[numeric_cols].isna().sum().sum())
        if nan:
            warnings.append(f"偵測到 {nan} 個缺值 → 執行時將以各欄中位數補值。")
        if spec.schema.check_bdl:
            arr = df[numeric_cols].to_numpy(dtype=float, na_value=np.nan)
            if arr.size:
                zr = float(np.nanmean(arr <= 0))
                info.append(f"BDL/零比例 {zr * 100:.0f}%")
                if zr > 0.70:
                    warnings.append("BDL/零比例 > 70%：CLR / SOM 等結果對零替換敏感，請謹慎詮釋。")

    if spec.schema.note:
        info.append(spec.schema.note)

    return ValidationReport(ok=(len(errors) == 0), errors=errors,
                            warnings=warnings, info=info)
