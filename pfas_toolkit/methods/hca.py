# -*- coding: utf-8 -*-
"""
hca.py — HCA 階層式分群（雙模式合併）
================================================================================
一個方法、兩種模式（由「資料類型」開關 data_kind 切換）：

  ‧ 一般數值（標準化歐氏）：任何數值寬表 → StandardScaler ＋ 距離（歐氏／Bray–Curtis／
    CLR-Aitchison）＋ 連結法（ward…）＋ 群數或距離門檻切群。
    輸出樹狀圖／群數評估（silhouette、Calinski-Harabasz）／各群輪廓 ＋ cophenetic correlation。

  ‧ PFAS 組成（占比指紋）：跨研究 PFAS 占比資料的組成分群（沿用組成方法學引擎 pfas_hca）：
    核心盤覆蓋率 ＋ complete／pairwise 缺值 ＋ Bray–Curtis／CLR 雙距離 ＋ silhouette 自動選 k
    （取近似最佳中最精簡）＋ 合併整群的組成中心指紋 ＋ cluster×論文／國家 批次交叉表；
    沒測（空白）全程不當 0。

由原 hca（通用）與 pfas_hca（PFAS 組成）合併而成。組成模式的引擎仍放在 pfas_hca.py，
但 pfas_hca 不再於 methods/__init__ 註冊為獨立方法 —— 網站只剩這一個 HCA 入口。
"""
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.theme import get_plt
from ..core.prep import numeric_frame, cluster_members_table

# PFAS 組成模式：沿用原 pfas_hca 引擎（同檔保留邏輯，未獨立註冊為方法）。
# make_demo 也用組成版：其資料為正值濃度＋NaN（沒測）＋BDL(0)，一般模式與組成模式皆可跑；
# 反之通用版 demo 含負值，會讓組成模式的列封閉／CLR 失效，故統一用組成版 demo。
from .pfas_hca import run as _run_compositional, make_demo as make_demo  # noqa: F401


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


def _run_general(df, params, ctx):
    """一般數值模式：標準化後階層式分群（原 hca 邏輯）。"""
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
    return ctx.result(summary=f"HCA 完成（一般數值模式，{metric}，{method}，{cut_desc}）。"
                              "hca_results.csv 的 HCA_Cluster 欄可當回歸類別特徵；"
                              "hca_cluster_members.csv 列出每群各有哪些樣本。"
                              "Cophenetic correlation 已記於執行紀錄，可判斷樹是否可信。"
                              "（PFAS 占比/組成資料請把『資料類型』切成 PFAS 組成模式。）")


def run(df, params, ctx):
    """依『資料類型』分派：一般數值 → 標準化歐氏路徑；PFAS 組成 → 占比指紋引擎。"""
    kind = str(params.get("data_kind", "一般數值（標準化歐氏）"))
    is_comp = ("PFAS" in kind) or ("組成" in kind) or kind.lower().startswith("comp")
    if is_comp:
        p2 = dict(params)
        # 組成模式未設核心盤覆蓋率 → 給合理預設 0.5（一般模式維持 0＝不過濾）
        if p2.get("min_coverage") in (None, "", 0, 0.0):
            p2["min_coverage"] = 0.5
            ctx.log("（PFAS 組成模式：未設核心盤覆蓋率 → 採預設 0.5；可在參數調整）")
        return _run_compositional(df, p2, ctx)
    return _run_general(df, params, ctx)


SPEC = MethodSpec(
    key="hca",
    name="HCA 階層式分群（通用 + PFAS 指紋組成）",
    summary="一個方法兩種模式：一般數值（標準化歐氏，輸出樹狀圖／群數評估／各群輪廓＋cophenetic），"
            "或切換『PFAS 組成模式』做占比指紋分群（核心盤覆蓋率＋Bray–Curtis/CLR 雙距離＋silhouette "
            "自動選 k＋合併整群指紋＋cluster×論文/國家 批次交叉表；缺值可完整觀測或逐對共同測項，沒測不當 0）。",
    params=[
        ParamSpec("data_kind", "資料類型（模式切換）", "choice",
                  default="一般數值（標準化歐氏）",
                  choices=["一般數值（標準化歐氏）", "PFAS 組成（占比指紋）"],
                  help="一般數值＝任何數值表，標準化後歐氏/連結法分群（最穩、最不挑資料）。"
                       "PFAS 組成＝PFAS 占比指紋的組成資料分群（核心盤＋BC/CLR＋自動選k＋批次交叉表，"
                       "沒測留空白不當 0）；上傳 PFAS 化合物濃度/占比寬表時請選這個。"),
        # ── 共用 ──
        ParamSpec("id_col", "樣本 ID 欄（可空）", "column", default="sample_id", optional=True,
                  help="每列樣本的識別欄；樹狀圖葉子會顯示它。留空則自動編號。"),
        ParamSpec("feature_cols", "特徵／化合物欄（不選＝自動全部數值欄）", "columns", default=[],
                  help="勾選要分群的數值欄（PFAS 模式即化合物欄）；不勾就用全部數值欄"
                       "（會自動忽略 ID／類別／論文／國家等非數值欄）。"),
        ParamSpec("n_clusters", "切割群數", "int", default=3, minimum=2,
                  help="切成幾群。一般模式：距離門檻=0 時生效。PFAS 模式：關閉自動選群數時生效。"),
        ParamSpec("max_clusters", "群數評估／掃描上限", "int", default=8, minimum=2,
                  help="群數評估圖、或自動選群數掃描 2～此值。"),
        ParamSpec("min_coverage", "核心盤覆蓋率門檻", "float", default=0.0, minimum=0.0, maximum=1.0,
                  help="只保留『有測比例 ≥ 此值』的欄（核心盤），其餘測得太少者剔除（不補值）。"
                       "0＝不過濾（一般模式預設）；PFAS 組成模式未設時自動採 0.5，建議 0.3~0.6。"),
        # ── 一般數值模式 ──
        ParamSpec("target_col", "排除欄／標籤欄（可空）", "column", default="", optional=True,
                  help="【一般模式】要從特徵中排除的欄（如標籤欄）。"),
        ParamSpec("linkage_method", "連結方法", "choice", default="ward",
                  choices=["ward", "complete", "average", "single"],
                  help="【一般模式】兩群距離定義。ward 最穩（搭歐氏/CLR）；選 Bray–Curtis 時自動改 average。"),
        ParamSpec("distance_metric", "距離類型（一般模式）", "choice", default="euclidean",
                  choices=["euclidean", "braycurtis", "aitchison"],
                  help="【一般模式】euclidean＝標準化後歐氏（一般數值）；braycurtis＝列封閉成占比的"
                       "Bray–Curtis；aitchison＝CLR 轉換後歐氏（組成資料）。"),
        ParamSpec("distance_threshold", "距離門檻（0＝用群數切）", "float", default=0.0, minimum=0.0,
                  help="【一般模式】填 > 0 改用距離切群：樹狀圖在此高度橫切，距離低於此值的樣本算同一組，"
                       "群數由資料自動決定。"),
        # ── PFAS 組成模式 ──
        ParamSpec("distance", "距離（組成 complete 模式）", "choice", default="both",
                  choices=["both", "braycurtis", "aitchison"],
                  help="【PFAS 組成模式】both＝同時跑 Bray–Curtis 與 CLR-Aitchison（建議對照）；"
                       "pairwise 缺值模式固定為 BC。"),
        ParamSpec("missing_mode", "缺值處理模式（組成）", "choice", default="complete",
                  choices=["complete", "pairwise"],
                  help="【PFAS 組成模式】complete＝核心盤完整觀測（含缺值列剔除，可雙距離，較嚴謹）；"
                       "pairwise＝保留部分測量樣本，逐對只用共同測項算 Bray–Curtis。"),
        ParamSpec("min_shared", "pairwise：最少共同測項數", "int", default=5, minimum=2,
                  help="【PFAS 組成 pairwise】兩樣本共同測得的化合物若少於此數→設最大距離，"
                       "避免靠少數共同測項假裝相似。"),
        ParamSpec("auto_k", "用 silhouette 自動選群數（組成）", "bool", default=True,
                  help="【PFAS 組成模式】開啟＝自動選群數（在近似最佳 silhouette 中取最精簡 k，避免過度切分）；"
                       "關閉＝用上面『切割群數』。"),
        ParamSpec("parsimony_tol", "群數精簡容差（silhouette）", "float", default=0.02,
                  minimum=0.0, maximum=0.2,
                  help="【PFAS 組成 自動選群數】在最高 silhouette 此容差內取最精簡（最小）群數；"
                       "覺得群切太多可調高。0＝純取最高分（最容易過度切分）。"),
        ParamSpec("dl_factor", "零替換係數 ×偵測極限（組成）", "float", default=0.65,
                  minimum=0.1, maximum=1.0,
                  help="【PFAS 組成模式】BDL(0) 的乘法零替換值＝係數 × 各化合物最小正值（CLR 前處理）。"),
        ParamSpec("filter_col", "類別／相態欄（可空，篩子集）", "column", default="", optional=True,
                  help="【PFAS 組成模式】例如 category 或 phase；配合下面的值選取，只跑某子集"
                       "（如某相態，避免跨相態混分）。"),
        ParamSpec("filter_values", "要保留的值（不選＝全部）", "values", default=[],
                  source_col="filter_col",
                  help="【PFAS 組成模式】選好上面的欄、載入資料後，這裡會列出可勾選的值（如 氣態-gas）。"),
        ParamSpec("paper_col", "論文欄（可空，批次交叉表）", "column", default="", optional=True,
                  help="【PFAS 組成模式】填了才輸出 cluster×論文 交叉表，看某群是否幾乎只來自同一篇（批次效應）。"),
        ParamSpec("country_col", "國家欄（可空，批次交叉表）", "column", default="", optional=True,
                  help="【PFAS 組成模式】填了才輸出 cluster×國家 交叉表。"),
    ],
    schema=InputSchema(min_rows=4, min_numeric_cols=2, id_col_param="id_col", check_bdl=True,
                       note="寬表：每列一個樣本，每欄一個特徵/化合物。"
                            "PFAS 組成模式：沒測請留空白（不要填 0），BDL 才填 0。",
                       missing_policy_note="一般 euclidean 以各欄中位數補值；braycurtis/aitchison 與 PFAS "
                                           "組成模式取核心盤後做完整觀測或逐對共同測項，沒測一律不補值。"),
    template_columns=["sample_id", "feature_1 / PFOA", "feature_2 / PFOS", "feature_3 / …"],
    uses_colors=["primary", "accent", "cmap_categorical"],
)
SPEC.run = run
SPEC.make_demo = make_demo
