# -*- coding: utf-8 -*-
"""
pfas_hca.py — PFAS 指紋 HCA（組成資料專用的階層式分群）
================================================================================
把跨研究 PFAS 占比資料做「來源指紋」分群，嚴格遵守組成資料方法學：

  ‧ 沒測(not measured)=空白(NaN)，未檢出(BDL)=0，真 0 三者不同 → 沒測不補 0、不補中位數。
  ‧ 先用「覆蓋率門檻」建立核心化合物盤（測得太少的化合物剔除），再取完整觀測列。
  ‧ 兩種距離並列：Bray–Curtis（先列封閉成比例）與 CLR-Aitchison（對數比，組成嚴謹）。
  ‧ silhouette 掃描自動選 k；cophenetic correlation 檢查樹是否忠實。
  ‧ 群代表指紋＝該群「全體樣本」的 CLR 組成中心（合併整群，非挑單篇論文），再轉回 %。
  ‧ 另輸出 cluster×論文 / cluster×國家 交叉表 → 抓「這群是不是只是同一篇/同一套測項」的批次效應。

輸入（寬表，每列一個樣本）：
  sample_id（可選）＋ 若干中繼欄（類別/論文/國家，可選）＋ 各 PFAS 化合物欄（占比或濃度皆可）。
可用「類別欄＋要保留的類別值」直接在這裡篩子集（等同把抽取整進來）。
"""
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.theme import get_plt
from ..core.prep import as_list, apply_value_filter
from .coda import mult_replacement, clr


def make_demo(seed=20):
    """150 樣本、3 種來源指紋；不同論文測不同盤（製造『沒測』）、含 BDL 0 與中繼欄。"""
    rng = np.random.default_rng(seed)
    core = ["PFBA", "PFPeA", "PFHxA", "PFHpA", "PFOA", "PFNA", "PFDA",
            "PFOS", "PFHxS", "PFBS", "6:2 FTS", "FOSA"]
    profiles = {
        "短鏈PFCA主導": np.array([40, 20, 15, 8, 7, 3, 1, 2, 2, 1, 0.5, 0.5]),
        "PFOS主導":     np.array([3, 2, 3, 2, 8, 4, 3, 45, 15, 7, 5, 3]),
        "前驅物/氟調":  np.array([5, 4, 5, 4, 6, 3, 2, 3, 2, 2, 40, 19]),
    }
    papers = ["P101", "P102", "P103", "P104", "P105", "P106"]
    panel = {p: set(core) for p in papers}
    panel["P103"] -= {"6:2 FTS", "FOSA", "PFBS"}     # 沒測中性/precursor
    panel["P105"] -= {"PFBA", "PFPeA"}               # 沒測超短鏈
    country = {"P101": "China", "P102": "China", "P103": "USA",
               "P104": "Germany", "P105": "Japan", "P106": "USA"}
    srcs = list(profiles)
    rows = []
    for i in range(150):
        src = srcs[i % 3]
        paper = papers[rng.integers(len(papers))]
        conc = profiles[src] * rng.lognormal(0, 0.25, size=len(core)) * rng.lognormal(3, 0.4)
        rec = {"sample_id": f"S{i:03d}", "category": "URBN_demo",
               "paper": paper, "country": country[paper], "true_source": src}
        for j, c in enumerate(core):
            if c not in panel[paper]:
                rec[c] = np.nan                       # 沒測
            else:
                rec[c] = 0.0 if rng.random() < 0.12 else round(float(conc[j]), 3)  # 12% BDL
        rows.append(rec)
    return pd.DataFrame(rows)


def _clr_center_pct(C_rows):
    """一群樣本的 CLR 組成中心 → 轉回百分比（合併整群的幾何平均組成）。"""
    center = C_rows.mean(axis=0)
    e = np.exp(center - center.max())
    return e / e.sum() * 100.0


def _best_k(Z, silX, silm, ks, fcluster, silhouette_score):
    sils = []
    for k in ks:
        lab = fcluster(Z, t=k, criterion="maxclust")
        sils.append(silhouette_score(silX, lab, metric=silm)
                    if len(np.unique(lab)) > 1 else np.nan)
    if np.all(np.isnan(sils)):
        return ks[0], sils
    return ks[int(np.nanargmax(sils))], sils


def run(df, params, ctx):
    plt = get_plt(ctx.theme)
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
    from sklearn.metrics import silhouette_score

    id_col = params.get("id_col") or ""
    filter_col = params.get("filter_col") or ""
    filter_vals = as_list(params.get("filter_values"))
    paper_col = params.get("paper_col") or ""
    country_col = params.get("country_col") or ""
    comp_sel = as_list(params.get("compound_cols"))
    min_cov = float(params.get("min_coverage", 0.5) or 0)
    dist_opt = params.get("distance", "both")
    auto_k = bool(params.get("auto_k", True))
    n_clusters = int(params.get("n_clusters", 3))
    max_k = int(params.get("max_clusters", 8))
    dl_factor = float(params.get("dl_factor", 0.65))
    cmap_cat = ctx.color("cmap_categorical", "tab10")
    primary = ctx.color("primary", "#4682b4")
    accent = ctx.color("accent", "#ff6347")

    d = df.copy()

    # 0) 依「類別欄」篩子集（把抽取整進方法；不填則用全部）
    if filter_col and filter_col in d.columns and filter_vals:
        d = apply_value_filter(d, ctx, filter_col, filter_vals, what="樣本")

    # 1) 決定化合物欄
    meta = {c for c in (id_col, filter_col, paper_col, country_col) if c}
    if comp_sel:
        comps = [c for c in comp_sel if c in d.columns]
    else:
        comps = [c for c in d.select_dtypes(include=[np.number]).columns if c not in meta]
    if len(comps) < 2:
        raise ValueError("可用化合物欄少於 2，請用『化合物欄』指定，或確認資料是寬表（每隻化合物一欄）。")
    for c in comps:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # 2) 核心盤（覆蓋率門檻）
    cov_all = d[comps].notna().mean()
    panel = [c for c in comps if cov_all[c] >= min_cov]
    if len(panel) < 2:
        top = cov_all.sort_values(ascending=False).head(8)
        raise ValueError(f"覆蓋率≥{min_cov:.0%} 的化合物不足 2 個。此資料覆蓋率最高："
                         + "、".join(f"{k} {v:.0%}" for k, v in top.items())
                         + "。請調低『核心盤覆蓋率門檻』。")
    ctx.log(f"核心盤 {len(panel)} 隻（覆蓋率≥{min_cov:.0%}）：{panel}")

    # 3) 設索引 + 完整觀測（盤上不可有沒測；沒測不補值，整列剔除）
    if id_col and id_col in d.columns:
        d = d.set_index(id_col)
    else:
        d.index = [f"S{i:04d}" for i in range(len(d))]
    before = len(d)
    sub = d.dropna(subset=panel, axis=0)
    ctx.log(f"完整觀測（盤上無『沒測』）：{len(sub)}/{before} 列"
            "（沒測不補 0/中位數，含缺值的列直接排除）")
    if len(sub) < 4:
        raise ValueError(f"完整觀測只剩 {len(sub)} 列，無法分群。請調低覆蓋率門檻或減少化合物欄。")

    P = sub[panel].astype(float)
    Pv = np.clip(P.values, 0.0, None)
    dl = np.array([P[c][P[c] > 0].min() if (P[c] > 0).any() else 1.0 for c in panel])
    C = clr(mult_replacement(Pv, dl, dl_factor))          # 共用的 CLR 座標（零替換後）

    # 盤覆蓋率報告（哪些化合物入盤 / 被剔除）
    cov_tab = pd.DataFrame({"compound": comps,
                            "coverage_pct": (cov_all[comps].values * 100).round(1),
                            "in_core_panel": [c in panel for c in comps]}
                           ).sort_values("coverage_pct", ascending=False)
    ctx.save_table(cov_tab, "pfas_hca_panel_coverage", index=False)

    tracks = []
    if dist_opt in ("both", "braycurtis"):
        tracks.append("braycurtis")
    if dist_opt in ("both", "aitchison"):
        tracks.append("aitchison")
    if not tracks:
        tracks = ["aitchison"]

    ks = list(range(2, min(max_k, len(sub) - 1) + 1))
    label_cols, summary_rows = {}, []

    for tr in tracks:
        if tr == "braycurtis":
            rs = P.sum(axis=1).replace(0, np.nan)
            Prop = P.div(rs, axis=0).fillna(0.0)
            cond = pdist(Prop.values, metric="braycurtis")
            Z = linkage(cond, method="average")
            silX, silm, dname, tag = squareform(cond), "precomputed", "Bray–Curtis", "BC"
        else:
            Z = linkage(C, method="ward")
            cond = pdist(C)
            silX, silm, dname, tag = C, "euclidean", "CLR-Aitchison", "CLR"

        coph, _ = cophenet(Z, cond)
        if auto_k:
            bestk, sils = _best_k(Z, silX, silm, ks, fcluster, silhouette_score)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(ks, sils, marker="o", color=primary)
            ax.axvline(bestk, color=accent, ls="--", label=f"最佳 k={bestk}")
            ax.set_title(f"{dname} Silhouette vs k"); ax.set_xlabel("群數"); ax.set_ylabel("silhouette")
            ax.legend(); fig.tight_layout(); ctx.save_fig(fig, f"pfas_hca_silhouette_{tag}")
        else:
            bestk = max(2, min(n_clusters, len(sub) - 1))
        labels = fcluster(Z, t=bestk, criterion="maxclust")
        sil_final = (silhouette_score(silX, labels, metric=silm)
                     if len(np.unique(labels)) > 1 else float("nan"))
        label_cols[f"{tag}_cluster"] = labels
        ctx.log(f"{dname}：k={bestk}，silhouette={sil_final:.3f}，cophenetic={coph:.3f}"
                "（cophenetic >0.8 樹可信）")
        summary_rows.append({"track": tag, "distance": dname, "best_k": int(bestk),
                             "silhouette": round(float(sil_final), 3),
                             "cophenetic": round(float(coph), 3), "n_rows": int(len(sub))})

        # 樹狀圖
        fig, ax = plt.subplots(figsize=(12, 5))
        ct = Z[-(bestk - 1), 2] if (bestk > 1 and len(Z) >= bestk) else 0
        dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8,
                   color_threshold=ct, ax=ax)
        ax.set_title(f"PFAS 指紋 HCA 樹狀圖（{dname}，k={bestk}）")
        ax.set_xlabel("樣本 / 群"); ax.set_ylabel("距離")
        fig.tight_layout(); ctx.save_fig(fig, f"pfas_hca_dendrogram_{tag}")

        # 群指紋＝該群全體的 CLR 組成中心（合併整群）→ %
        fp = []
        for cl in sorted(np.unique(labels)):
            p = _clr_center_pct(C[labels == cl])
            row = {"cluster": int(cl), "n": int((labels == cl).sum())}
            row.update({panel[j]: round(float(p[j]), 2) for j in range(len(panel))})
            fp.append(row)
        fp_df = pd.DataFrame(fp)
        ctx.save_table(fp_df, f"pfas_hca_fingerprint_{tag}", index=False)

        fig, ax = plt.subplots(figsize=(12, 5))
        fp_df.set_index("cluster")[panel].plot(kind="bar", stacked=True, ax=ax,
                                               colormap=cmap_cat, width=0.85)
        ax.set_title(f"各群來源指紋（{dname}；CLR 組成中心，合併整群）")
        ax.set_ylabel("組成 %"); ax.set_xlabel("群")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
        fig.tight_layout(); ctx.save_fig(fig, f"pfas_hca_fingerprint_{tag}")

        # 批次效應交叉表：cluster × 論文 / 國家
        for mc, mname in [(paper_col, "paper"), (country_col, "country")]:
            if mc and mc in sub.columns:
                ctab = pd.crosstab(pd.Series(labels, name="cluster"), sub[mc].values)
                ctx.save_table(ctab, f"pfas_hca_crosstab_{mname}_{tag}")

    # 分群標籤總表（含中繼欄 + 各距離的群號）
    lab_out = pd.DataFrame(index=sub.index)
    lab_out.index.name = id_col or "sample_id"
    for mc in (filter_col, paper_col, country_col, "true_source"):
        if mc and mc in sub.columns:
            lab_out[mc] = sub[mc].values
    for k, v in label_cols.items():
        lab_out[k] = v
    ctx.save_table(lab_out, "pfas_hca_labels")
    ctx.save_table(pd.DataFrame(summary_rows), "pfas_hca_summary", index=False)

    bc_clr = " + ".join(s["distance"] for s in summary_rows)
    return ctx.result(
        summary=f"PFAS 指紋 HCA 完成：核心盤 {len(panel)} 隻、完整觀測 {len(sub)} 列、距離 {bc_clr}。"
                "群指紋為『合併整群的 CLR 組成中心』(%）。"
                "請對照 pfas_hca_summary（best_k/silhouette/cophenetic）與 "
                "cluster×論文/國家 交叉表：若某群幾乎只來自同一篇論文或同一套測項，"
                "可能是批次/測項效應而非真實來源。沒測值全程未當 0 處理。")


SPEC = MethodSpec(
    key="pfas_hca",
    name="PFAS 指紋 HCA（組成）",
    summary="PFAS 占比指紋的組成資料分群：核心盤覆蓋率＋Bray–Curtis/CLR 雙距離＋silhouette 選 k＋"
            "cophenetic＋合併整群的 CLR 組成中心指紋＋cluster×論文/國家 批次效應交叉表；沒測不當 0。",
    params=[
        ParamSpec("id_col", "樣本 ID 欄（可空）", "column", default="sample_id", optional=True),
        ParamSpec("compound_cols", "化合物欄（不選＝自動取數值欄）", "columns", default=[],
                  help="勾選 PFAS 化合物欄；不勾就自動用所有數值欄（會排除 ID/類別/論文/國家欄）。"),
        ParamSpec("filter_col", "類別欄（可空，用來篩子集）", "column", default="", optional=True,
                  help="例如『category』。配合下面的值選取，就能只跑某一大類。"),
        ParamSpec("filter_values", "要保留的類別值（不選＝全部）", "values", default=[],
                  source_col="filter_col",
                  help="選好上面的類別欄、載入資料後，這裡會列出可勾選的值（如 URBN、WWTP）。"),
        ParamSpec("paper_col", "論文欄（可空，做批次交叉表）", "column", default="", optional=True),
        ParamSpec("country_col", "國家欄（可空，做批次交叉表）", "column", default="", optional=True),
        ParamSpec("min_coverage", "核心盤覆蓋率門檻", "float", default=0.5, minimum=0.0, maximum=1.0,
                  help="只保留『有測比例 ≥ 此值』的化合物；其餘測得太少者剔除（不補值）。"
                       "跨研究 PFAS 建議 0.3~0.6；門檻越高盤越小但完整觀測列越多。"),
        ParamSpec("distance", "距離（雙軌或擇一）", "choice", default="both",
                  choices=["both", "braycurtis", "aitchison"],
                  help="both＝同時跑 Bray–Curtis 與 CLR-Aitchison（建議互相對照）。"),
        ParamSpec("auto_k", "用 silhouette 自動選群數", "bool", default=True),
        ParamSpec("n_clusters", "群數（auto 關閉時使用）", "int", default=3, minimum=2),
        ParamSpec("max_clusters", "silhouette 掃描的最大群數", "int", default=8, minimum=2),
        ParamSpec("dl_factor", "零替換係數 ×偵測極限", "float", default=0.65, minimum=0.1, maximum=1.0,
                  help="BDL(0) 的乘法零替換值＝係數 × 各化合物最小正值（CLR 前處理）。"),
    ],
    schema=InputSchema(min_rows=8, min_numeric_cols=2, id_col_param="id_col", check_bdl=True,
                       note="寬表：每列一個樣本，每隻化合物一欄；沒測請留空白（不要填 0），BDL 才填 0。"),
    template_columns=["sample_id", "category", "paper", "country", "PFBA", "PFOA", "PFOS", "…"],
    uses_colors=["cmap_categorical", "primary", "accent"],
)
SPEC.run = run
SPEC.make_demo = make_demo

SPEC.manual = {
    "beginner": (
        "## 這個方法在做什麼\n"
        "把很多筆 PFAS 占比資料,依『組成像不像』分成幾群,每群給一個代表指紋,"
        "用來推測來源型態。專為『跨研究、各篇測的化合物不一樣』設計。\n\n"
        "## 最重要的觀念\n"
        "- **沒測 ≠ 未檢出 ≠ 0**。沒測請在表格留**空白**,未檢出(BDL)才填 **0**。\n"
        "- 本方法**不會把沒測補成 0 或中位數**;而是先挑出『大家都有測』的核心化合物盤,"
        "再只用盤上資料完整的樣本來分群。\n\n"
        "## 怎麼看結果\n"
        "1. `pfas_hca_summary`:每種距離的最佳群數、silhouette(越高越分得開)、cophenetic(>0.8 樹可信)。\n"
        "2. `各群來源指紋`圖/表:每群的代表組成(合併整群算出來的)。\n"
        "3. `cluster×論文 / 國家`交叉表:若某群幾乎只來自同一篇論文,可能只是該研究的測項習慣,"
        "不是真的來源差異 —— 要小心解讀。"
    ),
    "params": (
        "## 參數\n"
        "- **類別欄 / 要保留的類別值**:資料若有一欄標記大類(如 category),可只跑某幾類。\n"
        "- **核心盤覆蓋率門檻**:只留有測比例≥此值的化合物。調高→盤更乾淨但化合物更少;"
        "調低→化合物多但完整觀測的樣本變少。跨研究 PFAS 建議 0.3~0.6。\n"
        "- **距離**:both 同時跑 Bray–Curtis 與 CLR-Aitchison,建議兩者對照。\n"
        "- **論文欄 / 國家欄**:填了才會輸出批次效應交叉表。\n"
        "- **零替換係數**:BDL(0) 在做 CLR 前用『係數×最小正值』替換,預設 0.65。"
    ),
    "pro": (
        "## 原理\n"
        "核心盤(coverage≥門檻)→ 完整觀測(complete-case)→ 兩種距離:\n"
        "- **Bray–Curtis**:列封閉成比例後算 BC 相異度,average 連結。\n"
        "- **CLR-Aitchison**:乘法零替換 → CLR → 歐氏(=Aitchison),Ward 連結。\n"
        "k 由 silhouette 掃描決定;cophenetic correlation 檢查樹是否忠實保留原始距離。\n"
        "群指紋＝該群全體樣本在 CLR 空間的平均(組成中心)再轉回 %(合併整群,非單篇代表)。\n"
        "cluster×論文/國家 交叉表用於辨識 study-specific / analytical-panel / batch effect。"
    ),
}
