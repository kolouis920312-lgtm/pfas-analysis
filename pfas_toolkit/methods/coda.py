# -*- coding: utf-8 -*-
"""
coda.py — 組成資料分析 CoDA 轉換（由 coda_transform.py 整合而來）
高 BDL 穩健零替換 → CLR（必）/ ILR（選）。輸出可餵 PCA / SOM。
無圖，只有 CSV。
"""
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.prep import numeric_frame


def make_demo(seed=11):
    rng = np.random.default_rng(seed)
    sp = ["PFPeA", "PFHxA", "PFOA", "PFNA", "PFOS", "PFHxS", "6:2 FTS", "FOSA"]
    X = rng.lognormal(0.0, 1.0, size=(24, len(sp)))
    X[rng.random(X.shape) < 0.35] = 0.0
    df = pd.DataFrame(X, columns=sp)
    df.insert(0, "sample_id", [f"S{i:03d}" for i in range(24)])
    return df


def mult_replacement(X, dl, factor):
    """乘法零替換，逐列維持閉合。"""
    X = X.astype(float).copy()
    delta = factor * dl
    out = np.zeros_like(X)
    for r in range(X.shape[0]):
        row = X[r].copy()
        tot = row.sum()
        if tot <= 0:
            out[r] = factor * dl
            continue
        comp = row / tot
        d = delta / tot
        zero = comp <= 0
        comp[zero] = d[zero]
        comp[~zero] = comp[~zero] * (1.0 - d[zero].sum())
        out[r] = comp * tot
    return out


def clr(Xpos):
    L = np.log(Xpos)
    return L - L.mean(axis=1, keepdims=True)


def helmert_basis(D):
    V = np.zeros((D, D - 1))
    for i in range(1, D):
        h = np.zeros(D); h[:i] = 1.0; h[i] = -i
        V[:, i - 1] = h / np.sqrt(i * (i + 1))
    return V


def run(df, params, ctx):
    X = numeric_frame(df, ctx, id_col=params.get("id_col"))
    if X.shape[1] < 2:
        raise ValueError("CoDA 需至少 2 個成分欄。")
    factor = float(params.get("dl_factor", 0.65))
    do_ilr = bool(params.get("do_ilr", True))
    cols = list(X.columns); index = X.index
    Xv = X.values.astype(float)

    dl = np.array([X[c][X[c] > 0].min() if (X[c] > 0).any() else 1.0 for c in cols])
    zero_pct = float((Xv <= 0).mean() * 100)
    ctx.log(f"零/BDL 比例 {zero_pct:.1f}%  (>70% 時 CLR 對替換敏感，請謹慎)")

    Xpos = mult_replacement(Xv, dl, factor)
    ctx.save_table(pd.DataFrame(Xpos, index=index, columns=cols), "zero_replaced")

    C = clr(Xpos)
    ctx.save_table(pd.DataFrame(C, index=index, columns=[f"clr_{c}" for c in cols]),
                   "clr_transformed")
    ctx.log(f"CLR 完成（每列總和≈0：{np.allclose(C.sum(axis=1), 0)}）→ clr_transformed.csv")

    if do_ilr:
        V = helmert_basis(len(cols))
        I = C @ V
        ctx.save_table(pd.DataFrame(I, index=index, columns=[f"ilr{i+1}" for i in range(I.shape[1])]),
                       "ilr_transformed")
        ctx.log(f"ILR 完成，維度 {I.shape[1]}（Helmert 基底）→ ilr_transformed.csv")

    return ctx.result(summary="CoDA 完成：零替換 + CLR" + ("（含 ILR）" if do_ilr else "") +
                              "。clr_transformed.csv 可餵 PCA / SOM。")


SPEC = MethodSpec(
    key="coda",
    name="CoDA 組成轉換",
    summary="PFAS 占比資料的零替換 + CLR/ILR 對數比轉換，移除閉合偏誤後再做 PCA/SOM。",
    params=[
        ParamSpec("id_col", "ID 欄（可空）", "column", default="sample_id", optional=True),
        ParamSpec("dl_factor", "零替換係數 ×偵測極限", "float", default=0.65,
                  minimum=0.1, maximum=1.0, help="替換值 = 係數 × 各物種最小正值。"),
        ParamSpec("do_ilr", "同時輸出 ILR", "bool", default=True),
    ],
    schema=InputSchema(min_rows=3, min_numeric_cols=2, id_col_param="id_col", check_bdl=True),
    template_columns=["sample_id", "PFPeA", "PFHxA", "PFOA", "…"],
    uses_colors=[],
)
SPEC.run = run
SPEC.make_demo = make_demo
