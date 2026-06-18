# -*- coding: utf-8 -*-
"""
prep.py — 共用前處理（取代各腳本重複的 load_data 清洗段）
==========================================================
numeric_frame: 設索引、剔除指定欄、只留數值、缺值以中位數補值。
"""
import numpy as np
import pandas as pd


def numeric_frame(df: pd.DataFrame, ctx, id_col=None, drop_cols=()) -> pd.DataFrame:
    df = df.copy()
    if id_col and id_col in df.columns:
        df = df.set_index(id_col)
    for c in drop_cols:
        if c and c in df.columns:
            df = df.drop(columns=[c])
    X = df.select_dtypes(include=[np.number])
    nonnum = [c for c in df.columns if c not in X.columns]
    if nonnum:
        ctx.log(f"⚠ 忽略非數值欄：{nonnum}")
    nan = int(X.isna().sum().sum())
    if nan:
        ctx.log(f"⚠ {nan} 個缺值 → 以各欄中位數補值")
        X = X.fillna(X.median(numeric_only=True))
    return X
