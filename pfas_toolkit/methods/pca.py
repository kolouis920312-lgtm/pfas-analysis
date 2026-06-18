# -*- coding: utf-8 -*-
"""
pca.py — PCA 主成分分析（由 pca_analysis.py 整合而來）
輸出：pca_components.csv / pca_loadings.csv / 解釋變異圖 / biplot
"""
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.theme import get_plt
from ..core.prep import numeric_frame


def make_demo(n_samples=200, n_features=8, n_groups=3, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 5, size=(n_groups, n_features))
    rows, ids = [], []
    for i in range(n_samples):
        g = i % n_groups
        rows.append(centers[g] + rng.normal(0, 1.5, size=n_features))
        ids.append(f"S{i+1:03d}")
    df = pd.DataFrame(rows, columns=[f"feature_{j+1}" for j in range(n_features)])
    df.insert(0, "sample_id", ids)
    return df


def run(df, params, ctx):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    plt = get_plt(ctx.theme)

    X = numeric_frame(df, ctx, id_col=params.get("id_col"),
                      drop_cols=(params.get("target_col"),))
    if X.shape[1] < 2:
        raise ValueError("可用數值特徵少於 2 欄，無法 PCA。")
    feats = list(X.columns)
    ctx.log(f"樣本 {X.shape[0]}；特徵 {X.shape[1]}")

    Xs = StandardScaler().fit_transform(X)
    ncomp = params.get("n_components") or 0
    ncomp = int(ncomp) if int(ncomp or 0) > 0 else None
    maxc = min(Xs.shape)
    if ncomp and ncomp > maxc:
        ctx.log(f"⚠ 主成分數 {ncomp} 超過上限 {maxc}，已下調。")
        ncomp = maxc
    pca = PCA(n_components=ncomp, random_state=42)
    Xp = pca.fit_transform(Xs)
    ratio = pca.explained_variance_ratio_
    ctx.log(f"保留主成分 {pca.n_components_}；總解釋變異 {ratio.sum():.4f}")

    primary = ctx.color("primary", "#4682b4")
    accent = ctx.color("accent", "#ff6347")

    # 解釋變異
    cum = np.cumsum(ratio)
    n = len(ratio)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(range(1, n + 1), ratio, alpha=0.8, color=primary)
    ax[0].set_xlabel("主成分 (PC)"); ax[0].set_ylabel("解釋變異比例")
    ax[0].set_title("各主成分解釋變異")
    ax[1].plot(range(1, n + 1), cum, marker="o", color=accent)
    ax[1].axhline(0.90, ls="--", color="gray", label="90%")
    ax[1].axhline(0.95, ls="--", color="orange", label="95%")
    ax[1].set_xlabel("主成分數量"); ax[1].set_ylabel("累積解釋變異")
    ax[1].set_ylim(0, 1.02); ax[1].set_title("累積解釋變異 (Elbow)"); ax[1].legend()
    fig.tight_layout(); ctx.save_fig(fig, "pca_explained_variance")
    for thr in (0.90, 0.95):
        idx = int(np.searchsorted(cum, thr)) + 1
        if idx <= n:
            ctx.log(f"達到 {thr:.0%} 累積變異需 {idx} 個主成分")

    for pc in range(min(3, pca.n_components_)):
        loading = pca.components_[pc]
        order = np.argsort(np.abs(loading))[::-1][:5]
        ctx.log(f"PC{pc+1}: " + ", ".join(f"{feats[i]}({loading[i]:+.2f})" for i in order))

    # biplot
    if Xp.shape[1] >= 2:
        loadings = pca.components_.T
        imp = np.sqrt(loadings[:, 0] ** 2 + loadings[:, 1] ** 2)
        top = np.argsort(imp)[::-1][:min(10, len(feats))]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(Xp[:, 0], Xp[:, 1], alpha=0.4, s=15, color=primary)
        scale = np.abs(Xp[:, :2]).max()
        for i in top:
            ax.arrow(0, 0, loadings[i, 0] * scale * 0.8, loadings[i, 1] * scale * 0.8,
                     head_width=scale * 0.02, color=accent, alpha=0.85,
                     length_includes_head=True)
            ax.text(loadings[i, 0] * scale * 0.88, loadings[i, 1] * scale * 0.88,
                    feats[i], fontsize=9, color=accent)
        ax.set_xlabel(f"PC1 ({ratio[0]:.1%})"); ax.set_ylabel(f"PC2 ({ratio[1]:.1%})")
        ax.set_title(f"PCA Biplot (前 {len(top)} 重要特徵)")
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        fig.tight_layout(); ctx.save_fig(fig, "pca_biplot")

    cols = [f"PC{i+1}" for i in range(Xp.shape[1])]
    ctx.save_table(pd.DataFrame(Xp, columns=cols, index=X.index), "pca_components")
    ctx.save_table(pd.DataFrame(pca.components_.T, columns=cols, index=feats), "pca_loadings")
    return ctx.result(summary=f"PCA 完成：{pca.n_components_} 個主成分，"
                              f"總解釋變異 {ratio.sum():.1%}。"
                              "pca_components.csv 可餵 K-means / 回歸。")


SPEC = MethodSpec(
    key="pca",
    name="PCA 主成分分析",
    summary="標準化後做 PCA 降維，輸出主成分座標、載荷、解釋變異圖與 biplot。",
    params=[
        ParamSpec("id_col", "ID 欄（可空）", "column", default="", optional=True,
                  help="樣本識別欄，會設為索引；留空則用列號。"),
        ParamSpec("target_col", "排除欄/標籤欄（可空）", "column", default="", optional=True,
                  help="不參與分析的欄（如分類標籤）。"),
        ParamSpec("n_components", "保留主成分數（0=全部）", "int", default=0, minimum=0,
                  help="0＝全部保留，之後看 elbow 圖再決定。"),
    ],
    schema=InputSchema(min_rows=3, min_numeric_cols=2, id_col_param="id_col"),
    template_columns=["sample_id", "feature_1", "feature_2", "feature_3", "…"],
    uses_colors=["primary", "accent"],
)
SPEC.run = run
SPEC.make_demo = make_demo
