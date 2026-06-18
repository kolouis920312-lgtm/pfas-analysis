# -*- coding: utf-8 -*-
"""
prep.py — 共用前處理（取代各腳本重複的 load_data 清洗段）
==========================================================
numeric_frame: 設索引、剔除指定欄、只留數值、缺值以中位數補值。
"""
import numpy as np
import pandas as pd


def as_list(val):
    """把參數值正規化成乾淨的字串清單（接受 list 或逗號分隔字串；去空白、去 (無)）。

    columns / values 兩種多選參數的值，前端送來可能是 list，CLI/設定檔可能是字串，
    這裡統一成 list[str]，空 list 代表「全部」。
    """
    if val is None:
        return []
    if isinstance(val, str):
        parts = val.split(",")
    elif isinstance(val, (list, tuple, set)):
        parts = list(val)
    else:
        parts = [val]
    out = []
    for p in parts:
        s = str(p).strip()
        if s and s not in ("(無)", "全部"):
            out.append(s)
    return out


def numeric_frame(df: pd.DataFrame, ctx, id_col=None, drop_cols=(),
                  keep_cols=None) -> pd.DataFrame:
    """設索引、剔除指定欄、（選擇性）只留 keep_cols、只保留數值、缺值以中位數補值。

    keep_cols：多選特徵子集。給了非空清單就只分析這些欄（與資料實際存在的數值欄取交集）；
    留空/None 則沿用舊行為（全部數值欄）。
    """
    df = df.copy()
    if id_col and id_col in df.columns:
        df = df.set_index(id_col)
    for c in drop_cols:
        if c and c in df.columns:
            df = df.drop(columns=[c])

    keep = as_list(keep_cols)
    if keep:
        present = [c for c in keep if c in df.columns]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            ctx.log(f"⚠ 指定的特徵欄不存在，已略過：{missing}")
        if present:
            df = df[present]
            ctx.log(f"只分析選定的 {len(present)} 個特徵欄：{present}")

    X = df.select_dtypes(include=[np.number])
    nonnum = [c for c in df.columns if c not in X.columns]
    if nonnum:
        ctx.log(f"⚠ 忽略非數值欄：{nonnum}")
    nan = int(X.isna().sum().sum())
    if nan:
        ctx.log(f"⚠ {nan} 個缺值 → 以各欄中位數補值")
        X = X.fillna(X.median(numeric_only=True))
    return X


def apply_value_filter(df: pd.DataFrame, ctx, col, keep_values, what="列") -> pd.DataFrame:
    """依某欄的值篩選列（例：只留某幾個站別／季節）。

    keep_values 空 → 不篩（回傳原 df）。比對時兩邊都轉成字串去空白，避免型別不一致。
    """
    keep = as_list(keep_values)
    if not keep or not col or col not in df.columns:
        return df
    colvals = df[col].astype(str).str.strip()
    mask = colvals.isin(set(keep))
    kept = df[mask]
    ctx.log(f"依「{col}」篩選{what}：保留 {keep} → {len(kept)}/{len(df)} 列")
    if len(kept) == 0:
        raise ValueError(f"依「{col}」篩選後沒有任何資料（選了 {keep}，但欄內找不到這些值）。")
    return kept


def cluster_members_table(index, labels) -> pd.DataFrame:
    """把分群結果整理成「每群有哪些樣本」的清單表（給使用者看『哪幾筆同一組』）。

    回傳欄位：Cluster（群編號）、n（該群樣本數）、members（樣本 ID，逗號分隔）。
    """
    s = pd.Series(list(labels), index=[str(i) for i in index])
    rows = []
    for c in sorted(pd.unique(s.values)):
        ids = [i for i, v in zip(s.index, s.values) if v == c]
        rows.append({"Cluster": int(c), "n": len(ids), "members": ", ".join(ids)})
    return pd.DataFrame(rows)
