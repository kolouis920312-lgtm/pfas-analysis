# -*- coding: utf-8 -*-
"""
nonparam_stats.py — 無母數統計套件 (群差異 / 趨勢 / 相關 + 多重比較校正)
================================================================
【做什麼】
    1. 群間差異：Mann-Whitney U(2群) / Kruskal-Wallis(>2群) + 效應量 → group_tests.csv
    2. 時間趨勢：Mann-Kendall + Sen's slope → trend_tests.csv
    3. 相關：Spearman ρ 矩陣 + p 值 + BH-FDR q 值 → spearman_rho.csv / spearman_fdr.csv
    + 相關熱圖 spearman_heatmap.png

【為何需要】
    PFAS 濃度右偏、含大量 BDL → 不滿足常態/變異數同質，**參數檢定(t/ANOVA/Pearson)無效**。
    ‧ 「春 vs 冬」「清邁 vs 鹿林」「站別差異」需無母數檢定背書。
    ‧ ~39 物種 × 多氣象變數的相關矩陣，不校正多重比較會充滿偽陽性 → 必用 BH-FDR。
    ‧ 為 Paper1 描述結果、Paper2 的 XGBoost/SHAP 補上「顯著性」這一層(ML 本身沒有)。

【底層邏輯】
    ‧ 秩檢定：只用大小次序，免分布假設；對離群穩健。
    ‧ Mann-Kendall：累計所有時間對的符號 S，標準化為 Z 檢定單調趨勢(含 ties 校正)。
    ‧ Sen's slope：所有配對斜率的中位數，抗離群的趨勢量值。
    ‧ BH-FDR：把 p 由小到大排序，乘 n/rank 並做累積最小，控制偽發現率。

【用法】
    1. 設定 DATA_PATH（CSV：sample_id, [GROUP_COL], [TIME_COL], 各物種/變數...）。
    2. GROUP_COL 做群差異(如 "season"/"site")；TIME_COL 做趨勢(如 "date")。
    3. python nonparam_stats.py  （無檔則用 demo 假資料）
"""
import os, sys, io
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, spearmanr, norm

# ============================ 設定區 ============================
DATA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data_ready", "pfas_2025_wide_gp.csv")  # 真實資料(全2025 G+P)
INDEX_COL   = "sample_id"
GROUP_COL   = "season"     # 群差異分組欄:比較四季(凸顯春季抬升)；None=略過
TIME_COL    = "date"        # 趨勢時間欄；None=略過
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "03_nonparam")
# ===============================================================

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    if os.path.exists(DATA_PATH):
        print(f"[讀取] {DATA_PATH}")
        return pd.read_csv(DATA_PATH)
    print("[警告] 找不到資料檔 → 產生 demo 假資料")
    rng = np.random.default_rng(3)
    n = 36
    season = np.array((["冬"] * 12 + ["春"] * 12 + ["夏"] * 12))
    date = pd.date_range("2025-01-01", periods=n, freq="10D")
    sp = ["PFOA", "PFNA", "PFOS", "PFHxA", "PFHxS"]
    data = {"sample_id": [f"S{i:02d}" for i in range(n)], "season": season, "date": date}
    for j, s in enumerate(sp):
        trend = np.linspace(0, 1.2 * j, n)                     # 人工趨勢
        data[s] = np.clip(rng.lognormal(0.2, 0.7, n) + trend, 0, None)
    return pd.DataFrame(data)


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def mann_kendall(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return dict(n=n, S=np.nan, Z=np.nan, p=np.nan, trend="n/a")
    S = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, counts = np.unique(x, return_counts=True)
    tie = sum(c * (c - 1) * (2 * c + 5) for c in counts)
    var = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    if var <= 0:
        return dict(n=n, S=S, Z=np.nan, p=np.nan, trend="n/a")
    Z = (S - 1) / np.sqrt(var) if S > 0 else (S + 1) / np.sqrt(var) if S < 0 else 0.0
    p = 2 * (1 - norm.cdf(abs(Z)))
    trend = "上升" if (p < 0.05 and S > 0) else "下降" if (p < 0.05 and S < 0) else "無顯著趨勢"
    return dict(n=n, S=int(S), Z=round(float(Z), 3), p=round(float(p), 4), trend=trend)


def sen_slope(x, t):
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    sl = [(x[j] - x[i]) / (t[j] - t[i])
          for i in range(len(x) - 1) for j in range(i + 1, len(x))
          if t[j] != t[i] and not np.isnan(x[i]) and not np.isnan(x[j])]
    return float(np.median(sl)) if sl else np.nan


def main():
    df = load_data()
    drop = {INDEX_COL, GROUP_COL, TIME_COL}
    species = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

    # 1. 群差異
    if GROUP_COL and GROUP_COL in df.columns:
        groups = [g for _, g in df.groupby(GROUP_COL)]
        labels = [k for k, _ in df.groupby(GROUP_COL)]
        rows = []
        for s in species:
            arrs = [g[s].dropna().values for g in groups]
            arrs = [a for a in arrs if len(a) > 0]
            if len(arrs) < 2:
                continue
            # 高 BDL 真實資料常見:某物種跨群全為同值(多為全 0)→ 檢定無意義,跳過
            if np.unique(np.concatenate(arrs)).size < 2:
                continue
            if len(arrs) == 2:
                U, p = mannwhitneyu(arrs[0], arrs[1], alternative="two-sided")
                eff = 1 - 2 * U / (len(arrs[0]) * len(arrs[1]))      # rank-biserial
                rows.append(dict(variable=s, test="Mann-Whitney", stat=round(float(U), 2),
                                 p=p, effect=round(float(eff), 3)))
            else:
                H, p = kruskal(*arrs)
                ntot = sum(len(a) for a in arrs)
                eta2 = (H - len(arrs) + 1) / (ntot - len(arrs)) if ntot > len(arrs) else np.nan
                rows.append(dict(variable=s, test="Kruskal-Wallis", stat=round(float(H), 2),
                                 p=p, effect=round(float(eta2), 3)))
        if rows:
            gt = pd.DataFrame(rows)
            gt["p_fdr"] = bh_fdr(gt["p"].values)
            gt["sig"] = np.where(gt["p_fdr"] < 0.05, "*", "")
            gt["p"] = gt["p"].round(4); gt["p_fdr"] = gt["p_fdr"].round(4)
            gt.to_csv(os.path.join(OUTPUT_DIR, "group_tests.csv"), index=False)
            print(f"\n[群差異 by {GROUP_COL}] (groups={labels})\n", gt.to_string(index=False))

    # 2. 趨勢
    if TIME_COL and TIME_COL in df.columns:
        t = pd.to_datetime(df[TIME_COL], errors="coerce")
        torder = df.assign(_t=t).sort_values("_t")
        tnum = (pd.to_datetime(torder["_t"]) - pd.to_datetime(torder["_t"]).min()).dt.days.values
        rows = []
        for s in species:
            x = torder[s].values
            mk = mann_kendall(x)
            mk.update(variable=s, sen_slope_per_day=round(sen_slope(x, tnum), 5))
            rows.append(mk)
        tt = pd.DataFrame(rows)[["variable", "n", "S", "Z", "p", "sen_slope_per_day", "trend"]]
        if len(tt):
            tt["p_fdr"] = bh_fdr(tt["p"].fillna(1).values).round(4)
            tt.to_csv(os.path.join(OUTPUT_DIR, "trend_tests.csv"), index=False)
            print("\n[Mann-Kendall 趨勢 + Sen slope]\n", tt.to_string(index=False))

    # 3. Spearman + FDR
    sp_df = df[species]
    rho = sp_df.corr(method="spearman")
    cols = list(sp_df.columns)
    pmat = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = sp_df[cols[i]], sp_df[cols[j]]
            m = a.notna() & b.notna()
            if m.sum() >= 4:
                r, p = spearmanr(a[m], b[m])
                pmat.iloc[i, j] = pmat.iloc[j, i] = p
                pairs.append((i, j, p))
    if pairs:
        qs = bh_fdr([p for *_, p in pairs])
        qmat = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
        for (i, j, _), q in zip(pairs, qs):
            qmat.iloc[i, j] = qmat.iloc[j, i] = q
        rho.round(3).to_csv(os.path.join(OUTPUT_DIR, "spearman_rho.csv"))
        qmat.round(4).to_csv(os.path.join(OUTPUT_DIR, "spearman_fdr.csv"))
        print(f"\n[Spearman] ρ 矩陣 + FDR q 已輸出 ({len(pairs)} 對)")

        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]; plt.rcParams["axes.unicode_minus"] = False
            fig, ax = plt.subplots(figsize=(1 + 0.6 * len(cols), 1 + 0.6 * len(cols)))
            im = ax.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=7)
            ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7)
            # 標記 FDR 顯著
            for i in range(len(cols)):
                for j in range(len(cols)):
                    if i != j and qmat.iloc[i, j] < 0.05:
                        ax.text(j, i, "*", ha="center", va="center", color="black", fontsize=8)
            fig.colorbar(im, fraction=0.046); ax.set_title("Spearman ρ  (* = FDR<0.05)")
            fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "spearman_heatmap.png"), dpi=150)
            print("[圖] spearman_heatmap.png 已存。")
        except Exception as e:
            print("[圖略過]", e)

    print(f"\n✓ 完成，輸出於 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
