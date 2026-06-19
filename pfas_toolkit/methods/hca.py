# -*- coding: utf-8 -*-
"""
hca.py — 階層式分群 HCA（由 hca_analysis.py 整合而來）
輸出：樹狀圖 / 群數評估圖 / 各群輪廓圖 / hca_results.csv
距離可選：
  euclidean   標準化後歐氏距離（Ward；一般數值資料）
  braycurtis  Bray–Curtis 相異度（先列封閉成組成比例；適合 PFAS 占比指紋）
  aitchison   CLR 轉換後歐氏距離（= Aitchison 距離；組成資料較嚴謹）
另輸出 cophenetic correlation（樹是否忠實保留原始距離；>0.8 可信）。
"""
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.theme import get_plt
from ..core.prep import numeric_frame, cluster_members_table


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


def _representation(X, metric, method, ctx):
    """依距離類型把資料轉成 (linkage Z, 給 silhouette 的 X 與 metric, 群輪廓基底 df, 輪廓標題)。"""
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage

    if metric == "braycurtis":
        rs = X.sum(axis=1).replace(0, np.nan)
        P = X.div(rs, axis=0).fillna(0.0)               # 列封閉成比例（0/0 → 0）
        cond = pdist(P.values, metric="braycurtis")
        lm = method if method in ("average", "complete", "single", "weighted") else "average"
        if lm != method:
            ctx.log(f"Bray–Curtis 為預先計算距離，不支援 {method} 連結 → 改用 {lm}")
        Z = linkage(cond, method=lm)
        prof = (P * 100.0)
        return Z, cond, squareform(cond), "precomputed", prof, "群平均組成 %"

    if metric == "aitchison":
        from .coda import mult_replacement, clr
        Xv = np.clip(X.values.astype(float), 0, None)
        dl = np.array([X[c][X[c] > 0].min() if (X[c] > 0).any() else 1.0 for c in X.columns])
        C = clr(mult_replacement(Xv, dl, 0.65))         # 乘法零替換 → CLR
        Z = linkage(C, method=method)                   # CLR 空間歐氏 = Aitchison（Ward 可用）
        cond = pdist(C)
        prof = pd.DataFrame(C, columns=[f"clr_{c}" for c in X.columns], index=X.index)
        return Z, cond, C, "euclidean", prof, "群 CLR 平均（對數比）"

    # 預設 euclidean：標準化後 Ward
    rep = StandardScaler().fit_transform(X)
    Z = linkage(rep, method=method)
    cond = pdist(rep)
    prof = pd.DataFrame(rep, columns=X.columns, index=X.index)
    return Z, cond, rep, "euclidean", prof, "標準化均值"


def run(df, params, ctx):
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from scipy.cluster.hierarchy import dendrogram, fcluster, cophenet
    plt = get_plt(ctx.theme)

    method = params.get("linkage_method", "ward")
    metric = params.get("distance_metric", "euclidean")
    min_cov = float(params.get("min_coverage", 0) or 0)
    max_clusters = int(params.get("max_clusters", 10))
    n_clusters = int(params.get("n_clusters", 4))
    dist_thr = float(params.get("distance_threshold", 0) or 0)
    primary = ctx.color("primary", "#4682b4")
    accent = ctx.color("accent", "#ff6347")
    cmap_cat = ctx.color("cmap_categorical", "tab10")

    # 組成距離保留「沒測=NaN」語意：取核心盤後做完整觀測；歐氏沿用中位數補值
    miss = "median" if metric == "euclidean" else "drop"
    X = numeric_frame(df, ctx, id_col=params.get("id_col"),
                      drop_cols=(params.get("target_col"),),
                      keep_cols=params.get("feature_cols"),
                      missing=miss, min_coverage=min_cov)
    if X.shape[1] < 2:
        raise ValueError("可用數值特徵少於 2 欄，無法分群。（核心盤覆蓋率門檻可能太高，或特徵欄選太少）")
    if X.shape[0] < 4:
        raise ValueError(f"可用樣本只剩 {X.shape[0]} 列，無法分群（組成距離會丟棄含缺值的列）。")
    if n_clusters >= X.shape[0]:
        raise ValueError(f"群數 {n_clusters} 不可 >= 樣本數 {X.shape[0]}。")
    ctx.log(f"樣本 {X.shape[0]}；特徵 {X.shape[1]}；距離={metric}；linkage={method}")

    Z, cond, sil_X, sil_metric, prof_basis, prof_title = _representation(X, metric, method, ctx)

    # 樹是否忠實保留原始距離（cophenetic correlation）
    try:
        coph, _ = cophenet(Z, cond)
        ctx.log(f"Cophenetic correlation = {coph:.3f}"
                "（>0.8 樹結構可信；越接近 1 越忠實保留原始距離）")
    except Exception as e:
        ctx.log(f"（cophenetic 計算略過：{e}）")

    # 決定切群方式：距離門檻 > 0 → 用距離切；否則用群數切
    use_dist = dist_thr > 0
    if use_dist:
        labels = fcluster(Z, t=dist_thr, criterion="distance")
        ct = dist_thr
        n_eff = int(len(np.unique(labels)))
        ctx.log(f"以距離門檻 {dist_thr:g} 切群 → 得到 {n_eff} 群。")
    else:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        ct = Z[-(n_clusters - 1), 2] if (len(Z) >= n_clusters > 1) else 0
        n_eff = n_clusters

    # 樹狀圖
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=9,
               show_contracted=True, color_threshold=ct, ax=ax)
    if ct > 0:
        cut_label = (f"距離門檻 {dist_thr:g}（{n_eff} 群）" if use_dist
                     else f"切成 {n_clusters} 群")
        ax.axhline(ct, color=accent, ls="--", alpha=0.7, label=cut_label)
        ax.legend()
    ax.set_title(f"HCA 樹狀圖（{metric}，method={method}）")
    ax.set_xlabel("樣本 / 群"); ax.set_ylabel("距離")
    fig.tight_layout(); ctx.save_fig(fig, "hca_dendrogram")

    # 群數評估（silhouette / Calinski-Harabasz）
    maxk = min(max_clusters, X.shape[0] - 1)
    ks = list(range(2, maxk + 1))
    sil, ch = [], []
    samp = 1000 if (X.shape[0] > 1000 and sil_metric != "precomputed") else None
    for k in ks:
        lab = fcluster(Z, t=k, criterion="maxclust")
        if len(np.unique(lab)) < 2:
            sil.append(np.nan); ch.append(np.nan); continue
        sil.append(silhouette_score(sil_X, lab, metric=sil_metric,
                                    sample_size=samp, random_state=42))
        ch.append(np.nan if sil_metric == "precomputed"
                  else calinski_harabasz_score(sil_X, lab))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ks, sil, marker="o", color=primary)
    ax[0].set_title("Silhouette (越高越好)"); ax[0].set_xlabel("群數"); ax[0].set_ylabel("分數")
    if np.all(np.isnan(ch)):
        ax[1].text(0.5, 0.5, "Bray–Curtis 距離\n不適用 Calinski-Harabasz",
                   ha="center", va="center", transform=ax[1].transAxes, color="#888")
    else:
        ax[1].plot(ks, ch, marker="o", color=accent)
    ax[1].set_title("Calinski-Harabasz (越高越好)"); ax[1].set_xlabel("群數"); ax[1].set_ylabel("分數")
    fig.tight_layout(); ctx.save_fig(fig, "hca_cluster_evaluation")
    if not np.all(np.isnan(sil)):
        ctx.log(f"Silhouette 建議群數 = {ks[int(np.nanargmax(sil))]}")
    if not np.all(np.isnan(ch)):
        ctx.log(f"Calinski-Harabasz 建議群數 = {ks[int(np.nanargmax(ch))]}")

    if len(np.unique(labels)) >= 2:
        s = silhouette_score(sil_X, labels, metric=sil_metric)
        ctx.log(f"最終 Silhouette = {s:.4f}")

    # 各群輪廓
    prof = prof_basis.copy()
    prof["Cluster"] = labels
    g = prof.groupby("Cluster").mean()
    ax = g.T.plot(kind="bar", figsize=(14, 5), colormap=cmap_cat)
    ax.set_title(f"HCA 各群特徵輪廓（{prof_title}）")
    ax.set_ylabel(prof_title); ax.set_xlabel("特徵")
    plt.xticks(rotation=45, ha="right")
    ax.legend(title="群", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig = ax.get_figure(); fig.tight_layout(); ctx.save_fig(fig, "hca_cluster_profile")

    out = X.copy(); out["HCA_Cluster"] = labels
    ctx.save_table(out, "hca_results")
    members = cluster_members_table(X.index, labels)
    ctx.save_table(members, "hca_cluster_members", index=False)
    counts = pd.Series(labels).value_counts().sort_index()
    ctx.log("各群樣本數：" + ", ".join(f"群{k}={v}" for k, v in counts.items()))
    for _, r in members.iterrows():
        mtxt = str(r["members"])
        if len(mtxt) > 160:
            mtxt = mtxt[:160] + " …"
        ctx.log(f"群{int(r['Cluster'])}（{int(r['n'])} 筆）：{mtxt}")
    cut_desc = (f"距離門檻 {dist_thr:g} → {n_eff} 群" if use_dist else f"{n_clusters} 群")
    return ctx.result(summary=f"HCA 完成（{metric}，{method}，{cut_desc}）。"
                              "hca_results.csv 的 HCA_Cluster 欄可當回歸類別特徵；"
                              "hca_cluster_members.csv 列出每群各有哪些樣本。"
                              "Cophenetic correlation 已記於執行紀錄，可判斷樹是否可信。")


SPEC = MethodSpec(
    key="hca",
    name="HCA 階層式分群",
    summary="標準化後做階層式分群，輸出樹狀圖、群數評估與各群特徵輪廓；距離可選 Euclidean / Bray–Curtis / CLR-Aitchison，並報 cophenetic correlation。",
    params=[
        ParamSpec("id_col", "ID 欄（可空）", "column", default="", optional=True),
        ParamSpec("target_col", "排除欄/標籤欄（可空）", "column", default="", optional=True),
        ParamSpec("feature_cols", "納入的特徵欄（不選＝全部）", "columns", default=[],
                  help="勾選要納入分群的數值欄；不勾就用全部。"),
        ParamSpec("distance_metric", "距離類型", "choice", default="euclidean",
                  choices=["euclidean", "braycurtis", "aitchison"],
                  help="euclidean＝標準化後歐氏（一般數值）；braycurtis＝先列封閉成組成比例再算"
                       "Bray–Curtis（PFAS 占比指紋）；aitchison＝CLR 轉換後歐氏（組成資料較嚴謹）。"
                       "選 braycurtis/aitchison 時會保留『沒測=空白』語意：取核心盤後丟棄含缺值的列。"),
        ParamSpec("min_coverage", "核心盤覆蓋率門檻（0=不過濾）", "float", default=0.0,
                  minimum=0.0, maximum=1.0,
                  help="只保留『有測比例 ≥ 此值』的化合物欄，避免把測得太少的欄硬補值。"
                       "跨研究 PFAS 建議 0.2~0.5。"),
        ParamSpec("linkage_method", "連結方法", "choice", default="ward",
                  choices=["ward", "complete", "average", "single"],
                  help="ward 僅適用 euclidean/aitchison；braycurtis 會自動改用 average。"),
        ParamSpec("max_clusters", "評估的最大群數", "int", default=10, minimum=2),
        ParamSpec("n_clusters", "最終切割群數", "int", default=4, minimum=2,
                  help="先看樹狀圖與評估圖再決定。距離門檻 > 0 時此項會被忽略。"),
        ParamSpec("distance_threshold", "距離門檻（0=用群數切）", "float", default=0.0,
                  minimum=0.0,
                  help="填 > 0 改用距離切群：樹狀圖在此高度橫切，距離低於此值的樣本算同一組。"),
    ],
    schema=InputSchema(min_rows=4, min_numeric_cols=2, id_col_param="id_col"),
    template_columns=["sample_id", "feature_1", "feature_2", "feature_3", "…"],
    uses_colors=["primary", "accent", "cmap_categorical"],
)
SPEC.run = run
SPEC.make_demo = make_demo
