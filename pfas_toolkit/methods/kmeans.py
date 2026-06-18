# -*- coding: utf-8 -*-
"""
kmeans.py — K-means 分群（由 kmeans_pca.py 整合而來）
通常吃 PCA 輸出的 pca_components.csv，也可吃任何數值表。
輸出：4 指標圖 / 散佈圖 / kmeans_labels.csv / kmeans_summary.csv
"""
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.theme import get_plt
from ..core.prep import numeric_frame, cluster_members_table


def make_demo(n_samples=200, n_pcs=5, n_groups=3, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 6, size=(n_groups, n_pcs))
    rows, ids = [], []
    for i in range(n_samples):
        g = i % n_groups
        rows.append(centers[g] + rng.normal(0, 1.5, size=n_pcs))
        ids.append(f"S{i+1:03d}")
    df = pd.DataFrame(rows, columns=[f"PC{j+1}" for j in range(n_pcs)])
    df.insert(0, "sample_id", ids)
    return df


def run(df, params, ctx):
    from sklearn.cluster import KMeans
    from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                                 davies_bouldin_score)
    plt = get_plt(ctx.theme)

    X = numeric_frame(df, ctx, id_col=params.get("id_col"))
    if X.shape[1] < 1:
        raise ValueError("沒有可用的數值欄。")
    use = int(params.get("use_n_pcs") or 0)
    if use > 0:
        use = min(use, X.shape[1]); X = X.iloc[:, :use]
        ctx.log(f"只使用前 {use} 個維度分群。")
    n_clusters = int(params.get("n_clusters", 3))
    max_k = int(params.get("max_k", 10))
    seed = int(params.get("random_seed", 42) or 42)
    if n_clusters >= X.shape[0]:
        raise ValueError(f"群數 {n_clusters} 不可 >= 樣本數 {X.shape[0]}。")
    primary = ctx.color("primary", "#4682b4")
    accent = ctx.color("accent", "#ff6347")
    cmap_cat = ctx.color("cmap_categorical", "tab10")
    ctx.log(f"分群輸入：{X.shape[0]} 樣本 × {X.shape[1]} 維")

    maxk = min(max_k, X.shape[0] - 1)
    ks = list(range(2, maxk + 1))
    inertia, sil, ch, db = [], [], [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        lab = km.fit_predict(X)
        inertia.append(km.inertia_); sil.append(silhouette_score(X, lab))
        ch.append(calinski_harabasz_score(X, lab)); db.append(davies_bouldin_score(X, lab))

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot(ks, inertia, marker="o", color=primary)
    ax[0, 0].set_title("Inertia (Elbow 找手肘)"); ax[0, 0].set_xlabel("k"); ax[0, 0].set_ylabel("群內平方和")
    ax[0, 1].plot(ks, sil, marker="o", color=primary)
    ax[0, 1].axvline(ks[int(np.argmax(sil))], ls="--", color=accent, alpha=0.6)
    ax[0, 1].set_title("Silhouette (越高越好)"); ax[0, 1].set_xlabel("k"); ax[0, 1].set_ylabel("分數")
    ax[1, 0].plot(ks, ch, marker="o", color=primary)
    ax[1, 0].axvline(ks[int(np.argmax(ch))], ls="--", color=accent, alpha=0.6)
    ax[1, 0].set_title("Calinski-Harabasz (越高越好)"); ax[1, 0].set_xlabel("k"); ax[1, 0].set_ylabel("分數")
    ax[1, 1].plot(ks, db, marker="o", color=primary)
    ax[1, 1].axvline(ks[int(np.argmin(db))], ls="--", color=accent, alpha=0.6)
    ax[1, 1].set_title("Davies-Bouldin (越低越好)"); ax[1, 1].set_xlabel("k"); ax[1, 1].set_ylabel("分數")
    fig.suptitle("K-means 分組品質量化"); fig.tight_layout()
    ctx.save_fig(fig, "kmeans_metrics")
    ctx.log(f"建議 k：Silhouette={ks[int(np.argmax(sil))]}, "
            f"CH={ks[int(np.argmax(ch))]}, DB={ks[int(np.argmin(db))]}")
    ctx.save_table(pd.DataFrame({"k": ks, "Inertia": inertia, "Silhouette": sil,
                                 "Calinski_Harabasz": ch, "Davies_Bouldin": db}),
                   "kmeans_metrics_table", index=False)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    if X.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap=cmap_cat, alpha=0.7, s=22)
        ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                   c="black", marker="X", s=200, label="群中心")
        ax.set_xlabel(X.columns[0]); ax.set_ylabel(X.columns[1])
        ax.set_title(f"K-means 分群結果 (k={n_clusters})")
        leg = ax.legend(*sc.legend_elements(), title="群", loc="upper right")
        ax.add_artist(leg); ax.legend(loc="upper left")
        fig.tight_layout(); ctx.save_fig(fig, "kmeans_scatter")
    else:
        ctx.log("⚠ 只有 1 維，省略散佈圖。")

    out = X.copy(); out["KMeans_Cluster"] = labels
    ctx.save_table(out, "kmeans_labels")
    members = cluster_members_table(X.index, labels)
    ctx.save_table(members, "kmeans_cluster_members", index=False)
    sil_f = silhouette_score(X, labels)
    ch_f = calinski_harabasz_score(X, labels)
    db_f = davies_bouldin_score(X, labels)
    ctx.save_table(pd.DataFrame([{"k": n_clusters, "Inertia": round(km.inertia_, 4),
                                  "Silhouette": round(sil_f, 4),
                                  "Calinski_Harabasz": round(ch_f, 4),
                                  "Davies_Bouldin": round(db_f, 4)}]),
                   "kmeans_summary", index=False)
    counts = pd.Series(labels).value_counts().sort_index()
    ctx.log("各群樣本數：" + ", ".join(f"群{k}={v}" for k, v in counts.items()))
    for _, r in members.iterrows():
        mtxt = str(r["members"])
        if len(mtxt) > 160:
            mtxt = mtxt[:160] + " …"
        ctx.log(f"群{int(r['Cluster'])}（{int(r['n'])} 筆）：{mtxt}")
    return ctx.result(summary=f"K-means 完成（k={n_clusters}）。Silhouette={sil_f:.3f}。"
                              "kmeans_labels.csv 標出每筆的群；"
                              "kmeans_cluster_members.csv 列出每群各有哪些樣本。")


SPEC = MethodSpec(
    key="kmeans",
    name="K-means 分群",
    summary="在數值座標（常用 PCA 輸出）上做 K-means，並用 4 指標量化最佳群數。",
    params=[
        ParamSpec("id_col", "ID 欄（可空）", "column", default="", optional=True),
        ParamSpec("use_n_pcs", "只用前幾維（0=全部）", "int", default=0, minimum=0),
        ParamSpec("max_k", "掃描的最大群數", "int", default=10, minimum=2),
        ParamSpec("n_clusters", "最終分群數", "int", default=3, minimum=2,
                  help="先看指標圖再決定。"),
        ParamSpec("random_seed", "亂數種子", "int", default=42, minimum=0,
                  help="換個數字可檢查分群結果穩不穩定。"),
    ],
    schema=InputSchema(min_rows=4, min_numeric_cols=1, id_col_param="id_col"),
    template_columns=["sample_id", "PC1", "PC2", "PC3", "…"],
    uses_colors=["primary", "accent", "cmap_categorical"],
)
SPEC.run = run
SPEC.make_demo = make_demo
