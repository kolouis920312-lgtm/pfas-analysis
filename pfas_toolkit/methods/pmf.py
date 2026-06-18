# -*- coding: utf-8 -*-
"""
pmf.py — 正矩陣因子分解 PMF 來源解析（不確定度加權 NMF）
=========================================================
PMF 與 PCA 的差別：PMF 對「非負」並用「測量不確定度」加權，分解出的因子可直接
解讀成實體污染來源（來源指紋＋各樣本貢獻），這是大氣來源解析的標準受體模型。

  X ≈ G · F           X：樣本×物種濃度；G：樣本×因子貢獻；F：因子×物種指紋
  目標 Q = Σ ((X − GF) / U)²   以不確定度 U 加權的最小平方
  解法：權重 W=1/U² 的乘法更新（Lee–Seung 加權版，等價 PMF 目標）

輸出：
  ‧ pmf_factor_profiles.csv（來源指紋，每因子的物種組成，行和=1）
  ‧ pmf_contributions.csv（各樣本對各因子的貢獻）
  ‧ pmf_source_share.csv（各來源平均貢獻佔比）
  ‧ 因子指紋圖 + 來源平均佔比圖
"""
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.theme import get_plt
from ..core.prep import numeric_frame


def make_demo(seed=21):
    """三個已知來源指紋合成資料：短鏈 PFCA、長鏈 PFSA、前驅物(FTS/FOSA)。"""
    rng = np.random.default_rng(seed)
    species = ["PFBA", "PFPeA", "PFHxA", "PFHpA", "PFOA", "PFNA", "PFDA",
               "PFBS", "PFHxS", "PFOS", "6:2 FTS", "FOSA"]
    profiles = np.array([
        [0.20, 0.18, 0.16, 0.12, 0.10, 0.07, 0.04, 0.02, 0.02, 0.02, 0.04, 0.03],
        [0.02, 0.02, 0.03, 0.03, 0.05, 0.05, 0.05, 0.10, 0.15, 0.35, 0.05, 0.05],
        [0.05, 0.05, 0.05, 0.05, 0.06, 0.05, 0.04, 0.03, 0.04, 0.05, 0.24, 0.24],
    ])
    n = 60
    G = rng.gamma(2.0, 1.0, size=(n, 3))
    X = (G @ profiles) * rng.lognormal(0, 0.10, size=(n, len(species)))
    df = pd.DataFrame(np.round(X, 4), columns=species)
    df.insert(0, "sample_id", [f"S{i:03d}" for i in range(n)])
    return df


def _weighted_nmf(M, U, p, max_iter, rng, ctx):
    """權重 W=1/U² 的乘法更新；最小化 Q=Σ((M-GF)/U)²。"""
    n, m = M.shape
    Wt = 1.0 / (U ** 2)
    G = rng.random((n, p)) + 0.1
    F = rng.random((p, m)) + 0.1
    eps = 1e-12
    WX = Wt * M
    for it in range(max_iter):
        G *= (WX @ F.T) / ((Wt * (G @ F)) @ F.T + eps)
        F *= (G.T @ WX) / (G.T @ (Wt * (G @ F)) + eps)
        if it % 500 == 0:
            Q = float(np.sum(((M - G @ F) / U) ** 2))
            ctx.log(f"  iter {it:>4}: Q = {Q:.1f}")
    return G, F


def run(df, params, ctx):
    plt = get_plt(ctx.theme)
    X = numeric_frame(df, ctx, id_col=params.get("id_col"))
    if X.shape[1] < 3:
        raise ValueError("PMF 需至少 3 個物種欄。")
    species = list(X.columns); index = X.index
    M = np.clip(X.values.astype(float), 0, None)
    n, m = M.shape

    p = int(params.get("n_factors", 4) or 4)
    p = max(2, min(p, min(n, m) - 1))
    max_iter = int(params.get("max_iter", 3000) or 3000)
    err_frac = float(params.get("error_frac", 0.10))
    mdl = float(params.get("mdl", 0.0))
    seed = int(params.get("seed", 42) or 42)
    rng = np.random.default_rng(seed)

    # 不確定度矩陣（Polissar 類型）：U=sqrt((err·X)²+(0.5·MDL)²)，再設資料尺度下限防權重爆衝
    posmed = float(np.median(M[M > 0])) if (M > 0).any() else 1.0
    U = np.sqrt((err_frac * M) ** 2 + (0.5 * mdl) ** 2)
    U = np.maximum(U, 0.05 * posmed)
    ctx.log(f"樣本 {n}×物種 {m}；因子 p={p}；誤差比例 {err_frac:.0%}")

    G, F = _weighted_nmf(M, U, p, max_iter, rng, ctx)

    # 正規化：每個因子指紋（F 列）縮放到總和=1，尺度推回貢獻 G
    scale = F.sum(axis=1); scale[scale <= 0] = 1.0
    Fn = F / scale[:, None]
    Gn = G * scale[None, :]

    Q_final = float(np.sum(((M - G @ F) / U) ** 2))
    dof = max(n * m - p * (n + m), 1)
    ss_tot = float(np.sum((M - M.mean(0)) ** 2))
    ss_res = float(np.sum((M - G @ F) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    ctx.log(f"Q={Q_final:.1f}；Q/自由度≈{Q_final/dof:.2f}；重建 R²={r2:.3f}")

    fcols = [f"factor{i+1}" for i in range(p)]
    prof = pd.DataFrame(Fn.T, index=species, columns=fcols)
    contrib = pd.DataFrame(Gn, index=index, columns=fcols)
    ctx.save_table(prof.round(4), "pmf_factor_profiles")
    ctx.save_table(contrib.round(4), "pmf_contributions")

    mean_contrib = Gn.mean(0)
    share = mean_contrib / mean_contrib.sum() * 100 if mean_contrib.sum() > 0 else mean_contrib
    share_df = pd.DataFrame({"factor": fcols,
                             "mean_contribution": mean_contrib.round(3),
                             "share_pct": np.round(share, 1)})
    ctx.save_table(share_df, "pmf_source_share", index=False)
    # 每個因子最具代表性的物種（指紋特徵）
    for k in range(p):
        order = np.argsort(Fn[k])[::-1][:4]
        ctx.log(f"factor{k+1}（佔比 {share[k]:.0f}%）主要物種：" +
                ", ".join(f"{species[i]}({Fn[k][i]:.2f})" for i in order))

    cmap_cat = ctx.color("cmap_categorical", "tab10")
    cmobj = plt.get_cmap(cmap_cat)
    ncol = min(2, p); nrow = int(np.ceil(p / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3 * nrow), squeeze=False)
    axflat = axes.ravel()
    xs = np.arange(len(species))
    for k in range(p):
        ax = axflat[k]
        ax.bar(xs, Fn[k], color=cmobj(k % 10))
        ax.set_title(f"factor{k+1}（佔比 {share[k]:.0f}%）", fontsize=9)
        ax.set_xticks(xs); ax.set_xticklabels(species, rotation=90, fontsize=6)
        ax.set_ylabel("組成比例")
    for ax in axflat[p:]:
        ax.axis("off")
    fig.suptitle("PMF 來源指紋（factor profiles）")
    fig.tight_layout(); ctx.save_fig(fig, "pmf_profiles")

    primary = ctx.color("primary", "#4682b4")
    fig, ax = plt.subplots(figsize=(5, 4))
    xf = np.arange(len(fcols))
    ax.bar(xf, share, color=primary)
    ax.set_ylabel("平均貢獻佔比 (%)"); ax.set_title("各來源平均貢獻")
    ax.set_xticks(xf); ax.set_xticklabels(fcols, rotation=30, ha="right", fontsize=8)
    fig.tight_layout(); ctx.save_fig(fig, "pmf_source_share")

    return ctx.result(summary=f"PMF 完成：{p} 個來源因子（不確定度加權 NMF），重建 R²={r2:.2f}。"
                              "pmf_factor_profiles.csv 是各來源指紋、pmf_contributions.csv 是各樣本來源貢獻、"
                              "pmf_source_share.csv 是平均佔比。比對指紋即可命名來源（如長鏈 PFSA／前驅物）。")


SPEC = MethodSpec(
    key="pmf",
    name="PMF 來源解析",
    summary="正矩陣因子分解（非負＋不確定度加權），把 PFAS 濃度矩陣拆成來源指紋與各樣本貢獻，量化各來源佔比。",
    params=[
        ParamSpec("id_col", "ID 欄（可空）", "column", default="sample_id", optional=True),
        ParamSpec("n_factors", "來源因子數 p", "int", default=4, minimum=2,
                  help="先試 3–6，看指紋是否可解讀、Q/自由度是否合理。"),
        ParamSpec("error_frac", "相對測量誤差比例", "float", default=0.10, minimum=0.01, maximum=0.5,
                  help="建構不確定度矩陣用；典型 10–20%。"),
        ParamSpec("mdl", "方法偵測極限 MDL", "float", default=0.0, minimum=0.0,
                  help="低濃度不確定度下限；不確定可留 0。"),
        ParamSpec("max_iter", "最大迭代數", "int", default=3000, minimum=500),
        ParamSpec("seed", "亂數種子", "int", default=42, minimum=0),
    ],
    schema=InputSchema(min_rows=10, min_numeric_cols=3, id_col_param="id_col", check_bdl=True,
                       note="PMF 建議樣本數 ≥ 物種數的數倍；BDL 請以 0 或空白表示。"),
    template_columns=["sample_id", "PFBA", "PFOA", "PFOS", "6:2 FTS", "…"],
    uses_colors=["primary", "cmap_categorical"],
)
SPEC.run = run
SPEC.make_demo = make_demo
