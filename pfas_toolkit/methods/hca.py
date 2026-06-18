# -*- coding: utf-8 -*-
"""
hca.py — 階層式分群 HCA（由 hca_analysis.py 整合而來）
輸出：樹狀圖 / 群數評估圖 / 各群輪廓圖 / hca_results.csv
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
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    plt = get_plt(ctx.theme)

    method = params.get("linkage_method", "ward")
    max_clusters = int(params.get("max_clusters", 10))
    n_clusters = int(params.get("n_clusters", 4))
    primary = ctx.color("primary", "#4682b4")
    accent = ctx.color("accent", "#ff6347")
    cmap_cat = ctx.color("cmap_categorical", "tab10")

    X = numeric_frame(df, ctx, id_col=params.get("id_col"),
                      drop_cols=(params.get("target_col"),))
    if X.shape[1] < 2:
        raise ValueError("可用數值特徵少於 2 欄，無法分群。")
    if n_clusters >= X.shape[0]:
        raise ValueError(f"群數 {n_clusters} 不可 >= 樣本數 {X.shape[0]}。")
    ctx.log(f"樣本 {X.shape[0]}；特徵 {X.shape[1]}；linkage={method}")

    Xs = StandardScaler().fit_transform(X)
    Z = linkage(Xs, method=method)

    # 樹狀圖
    fig, ax = plt.subplots(figsize=(14, 6))
    ct = Z[-(n_clusters - 1), 2] if (len(Z) >= n_clusters > 1) else 0
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=9,
               show_contracted=True, color_threshold=ct, ax=ax)
    if n_clusters > 1:
        ax.axhline(ct, color=accent, ls="--", alpha=0.7, label=f"切成 {n_clusters} 群")
        ax.legend()
    ax.set_title(f"HCA 樹狀圖 (method={method})")
    ax.set_xlabel("樣本 / 群"); ax.set_ylabel("距離")
    fig.tight_layout(); ctx.save_fig(fig, "hca_dendrogram")

    # 群數評估
    maxk = min(max_clusters, X.shape[0] - 1)
    ks = list(range(2, maxk + 1))
    sil, ch = [], []
    samp = 1000 if X.shape[0] > 1000 else None
    for k in ks:
        lab = fcluster(Z, t=k, criterion="maxclust")
        if len(np.unique(lab)) < 2:
            sil.append(np.nan); ch.append(np.nan); continue
        sil.append(silhouette_score(Xs, lab, sample_size=samp, random_state=42))
        ch.append(calinski_harabasz_score(Xs, lab))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ks, sil, marker="o", color=primary)
    ax[0].set_title("Silhouette (越高越好)"); ax[0].set_xlabel("群數"); ax[0].set_ylabel("分數")
    ax[1].plot(ks, ch, marker="o", color=accent)
    ax[1].set_title("Calinski-Harabasz (越高越好)"); ax[1].set_xlabel("群數"); ax[1].set_ylabel("分數")
    fig.tight_layout(); ctx.save_fig(fig, "hca_cluster_evaluation")
    if not np.all(np.isnan(sil)):
        ctx.log(f"Silhouette 建議群數 = {ks[int(np.nanargmax(sil))]}")
        ctx.log(f"Calinski-Harabasz 建議群數 = {ks[int(np.nanargmax(ch))]}")

    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    if len(np.unique(labels)) >= 2:
        ctx.log(f"最終 Silhouette={silhouette_score(Xs, labels):.4f}；"
                f"Calinski-Harabasz={calinski_harabasz_score(Xs, labels):.4f}")

    # 各群輪廓
    prof = pd.DataFrame(Xs, columns=X.columns, index=X.index)
    prof["Cluster"] = labels
    g = prof.groupby("Cluster").mean()
    ax = g.T.plot(kind="bar", figsize=(14, 5), colormap=cmap_cat)
    ax.set_title("HCA 各群特徵輪廓 (標準化均值)")
    ax.set_ylabel("均值"); ax.set_xlabel("特徵")
    plt.xticks(rotation=45, ha="right")
    ax.legend(title="群", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig = ax.get_figure(); fig.tight_layout(); ctx.save_fig(fig, "hca_cluster_profile")

    out = X.copy(); out["HCA_Cluster"] = labels
    ctx.save_table(out, "hca_results")
    counts = pd.Series(labels).value_counts().sort_index()
    ctx.log("各群樣本數：" + ", ".join(f"群{k}={v}" for k, v in counts.items()))
    return ctx.result(summary=f"HCA 完成（{method}，{n_clusters} 群）。"
                              "hca_results.csv 的 HCA_Cluster 欄可當回歸類別特徵。")


SPEC = MethodSpec(
    key="hca",
    name="HCA 階層式分群",
    summary="標準化後做階層式分群，輸出樹狀圖、群數評估與各群特徵輪廓。",
    params=[
        ParamSpec("id_col", "ID 欄（可空）", "column", default="", optional=True),
        ParamSpec("target_col", "排除欄/標籤欄（可空）", "column", default="", optional=True),
        ParamSpec("linkage_method", "連結方法", "choice", default="ward",
                  choices=["ward", "complete", "average", "single"]),
        ParamSpec("max_clusters", "評估的最大群數", "int", default=10, minimum=2),
        ParamSpec("n_clusters", "最終切割群數", "int", default=4, minimum=2,
                  help="先看樹狀圖與評估圖再決定。"),
    ],
    schema=InputSchema(min_rows=4, min_numeric_cols=2, id_col_param="id_col"),
    template_columns=["sample_id", "feature_1", "feature_2", "feature_3", "…"],
    uses_colors=["primary", "accent", "cmap_categorical"],
)
SPEC.run = run
SPEC.make_demo = make_demo
