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
                  keep_cols=None, missing="median", min_coverage=0.0) -> pd.DataFrame:
    """設索引、剔除指定欄、（選擇性）只留 keep_cols、只保留數值欄。

    keep_cols：多選特徵子集。給了非空清單就只分析這些欄（與資料實際存在的數值欄取交集）；
    留空/None 則沿用舊行為（全部數值欄）。

    missing：缺值（NaN）處理策略
      "median" 各欄中位數補值（預設，向後相容；PCA/K-means 等需要完整矩陣）
      "keep"   保留 NaN 不補（給能容忍缺值的方法自行處理）
      "drop"   丟掉仍含 NaN 的『列』（取核心盤後做完整觀測 complete-case）
    min_coverage：>0 時先剔除「非空值比例 < 此值」的欄，建立核心盤。
      ── 對跨研究 PFAS 很重要：沒測(NaN) 不可當 0 或中位數，應先把測得太少的欄剔除，
         而非把『沒測』補成一個數值（會假造組成差異）。
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
        missing_cols = [c for c in keep if c not in df.columns]
        if missing_cols:
            ctx.log(f"⚠ 指定的特徵欄不存在，已略過：{missing_cols}")
        if present:
            df = df[present]
            ctx.log(f"只分析選定的 {len(present)} 個特徵欄：{present}")

    X = df.select_dtypes(include=[np.number])
    nonnum = [c for c in df.columns if c not in X.columns]
    if nonnum:
        ctx.log(f"⚠ 忽略非數值欄：{nonnum}")

    if min_coverage and min_coverage > 0:
        cov = X.notna().mean()
        keep_c = cov[cov >= min_coverage].index.tolist()
        drop_c = [c for c in X.columns if c not in keep_c]
        if drop_c:
            ctx.log(f"覆蓋率 < {min_coverage:.0%} 剔除 {len(drop_c)} 欄；"
                    f"核心盤保留 {len(keep_c)} 欄。")
        X = X[keep_c]

    nan = int(X.isna().sum().sum())
    if missing == "drop":
        before = len(X)
        X = X.dropna(axis=0)
        if before - len(X):
            ctx.log(f"丟棄仍含缺值的 {before - len(X)} 列（核心盤 complete-case）→ 剩 {len(X)} 列")
    elif missing == "keep":
        if nan:
            ctx.log(f"保留 {nan} 個缺值（沒測），不補值，交由方法處理")
    else:  # "median"（預設、向後相容）
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
