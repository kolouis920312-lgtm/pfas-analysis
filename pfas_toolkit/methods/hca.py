# -*- coding: utf-8 -*-
"""
hca.py — HCA 階層式分群（PFAS 指紋組成）
================================================================================
跨研究 PFAS 占比資料的「來源指紋」組成分群（引擎在 pfas_hca.py）：
  核心盤覆蓋率 ＋ complete/pairwise 缺值 ＋ Bray–Curtis/CLR 雙距離 ＋ silhouette 自動選 k
  ＋ 群指紋（可選聚合法）＋ 自動 cluster×category 交叉表；沒測(空白)全程不當 0。

群指紋聚合法（fingerprint_agg，皆輸出線性 %，可直接餵 CMB/PMF 當來源 profile）：
  中位數 / 等權平均 / CLR 幾何中心 / medoid(最中心樣本) / silhouette 加權。

註：原「通用數值模式」已移除——一般數值分群請改用 K-means 或其他方法；
    本方法專注 PFAS 占比組成，參數更精簡。
"""
from ..core.spec import MethodSpec, ParamSpec, InputSchema
from .pfas_hca import run as run, make_demo as make_demo  # noqa: F401  組成引擎

SPEC = MethodSpec(
    key="hca",
    name="HCA 階層式分群（PFAS 指紋組成）",
    summary="PFAS 占比指紋的組成資料分群：核心盤覆蓋率＋Bray–Curtis/CLR 雙距離＋silhouette 自動選 k"
            "＋cophenetic＋群指紋（可選聚合法：中位數/等權/CLR中心/medoid/silhouette加權，皆輸出線性%）"
            "＋自動 cluster×category 交叉表；缺值可選 complete 完整觀測或 pairwise 逐對共同測項；沒測不當 0。",
    params=[
        ParamSpec("id_col", "樣本 ID 欄（可空）", "column", default="sample_id", optional=True,
                  help="每列樣本的識別欄；樹狀圖葉子會顯示它。留空則自動編號。"),
        ParamSpec("compound_cols", "化合物欄（不選＝自動取數值欄）", "columns", default=[],
                  help="勾選 PFAS 化合物欄；不勾就自動用所有數值欄"
                       "（會自動排除 ID/category/country/paper/n_samples 等中繼欄）。"),
        ParamSpec("filter_col", "類別欄（可空，用來篩子集）", "column", default="", optional=True,
                  help="例如『category』。配合下面的值選取，就能只跑某一大類/相態。"),
        ParamSpec("filter_values", "要保留的值（不選＝全部）", "values", default=[],
                  source_col="filter_col",
                  help="選好上面的欄、載入資料後，這裡會列出可勾選的值（如 URBN、氣態-gas）。"),
        ParamSpec("min_coverage", "核心盤覆蓋率門檻", "float", default=0.5, minimum=0.0, maximum=1.0,
                  help="只保留『有測比例 ≥ 此值』的化合物（核心盤），其餘測得太少者剔除（不補值）。"
                       "跨研究 PFAS 建議 0.3~0.6；門檻越高盤越小但完整觀測列越多。"
                       "（若上傳已是核心盤/來源剖面，設 0 不再過濾。）"),
        ParamSpec("missing_mode", "缺值處理模式", "choice", default="complete",
                  choices=["complete", "pairwise"],
                  help="complete＝核心盤完整觀測（含缺值列剔除，可跑 BC＋CLR 雙距離，較嚴謹）；"
                       "pairwise＝保留部分測量的樣本，逐對只用共同測項算 Bray–Curtis（僅 BC）。"),
        ParamSpec("min_shared", "pairwise：最少共同測項數", "int", default=5, minimum=2,
                  help="僅 pairwise 模式：兩樣本共同測得的化合物若少於此數→設最大距離，"
                       "避免靠少數共同測項假裝相似。"),
        ParamSpec("distance", "距離（complete 模式用）", "choice", default="both",
                  choices=["both", "braycurtis", "aitchison"],
                  help="both＝同時跑 Bray–Curtis 與 CLR-Aitchison（建議互相對照）；pairwise 固定為 BC。"),
        ParamSpec("fingerprint_agg", "群指紋聚合法", "choice", default="中位數",
                  choices=["中位數", "等權平均", "CLR幾何中心", "medoid(最中心樣本)", "silhouette加權"],
                  help="把一群成員合成一條代表指紋的方式（皆輸出線性 %，可餵 CMB/PMF）："
                       "中位數＝抗離群(推薦)；等權平均＝每筆 1 票算術平均；CLR幾何中心＝對數比中心；"
                       "medoid＝直接取最中心的那一筆樣本(抗邊緣)；silhouette加權＝邊緣樣本少算一點。"),
        ParamSpec("auto_k", "用 silhouette 自動選群數", "bool", default=True,
                  help="開啟＝自動選群數（近似最佳 silhouette 中取最精簡 k，避免過度切分）；關閉＝用下方群數。"),
        ParamSpec("n_clusters", "群數（auto 關閉時使用）", "int", default=3, minimum=2,
                  help="關閉自動選群數時，固定切成這麼多群。"),
        ParamSpec("max_clusters", "silhouette 掃描的最大群數", "int", default=8, minimum=2,
                  help="自動選群數時，掃描 2～此值的範圍。"),
        ParamSpec("parsimony_tol", "群數精簡容差（silhouette）", "float", default=0.02,
                  minimum=0.0, maximum=0.2,
                  help="自動選群數時：在最高 silhouette 的此容差內取最精簡（最小）群數；"
                       "覺得群切太多可調高。0＝純取最高分（最容易過度切分）。"),
        ParamSpec("dl_factor", "零替換係數 ×偵測極限", "float", default=0.65, minimum=0.1, maximum=1.0,
                  help="BDL(0) 的乘法零替換值＝係數 × 各化合物最小正值（CLR 前處理）。"),
    ],
    schema=InputSchema(min_rows=8, min_numeric_cols=2, id_col_param="id_col", check_bdl=True,
                       note="寬表：每列一個樣本，每隻化合物一欄；沒測請留空白（不要填 0），BDL 才填 0。",
                       missing_policy_note="沒測值不會補值：complete 模式剔除含缺值的列、"
                                           "pairwise 模式逐對只用共同測項（皆不以 0／中位數填補）"),
    template_columns=["sample_id", "category", "PFBA", "PFOA", "PFOS", "6:2 FTS", "…"],
    uses_colors=["cmap_categorical", "primary", "accent"],
)
SPEC.run = run
SPEC.make_demo = make_demo
