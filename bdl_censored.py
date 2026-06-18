# -*- coding: utf-8 -*-
"""
bdl_censored.py — 偵測極限(BDL/censored)資料處理
================================================================
【做什麼】
  把含大量「未檢出 <MDL」的 PFAS 濃度資料做「統計上正確」的處理，輸出：
    1. 各物種偵測頻率表 (detection_frequency.csv)
    2. 三種替代法資料 (LOD/2, LOD/√2) 供比較 (subst_*.csv)
    3. ROS(Regression on Order Statistics) 填補後的完整資料 (ros_imputed.csv)
    4. Tobit(對數常態 MLE) 的母體平均/標準差摘要 (censored_summary.csv)
    5. PMF 用的逐格不確定度矩陣 s_ij (uncertainty_matrix.csv)
    6. 偵測/未檢出熱圖 (detection_heatmap.png)

【為何需要】
  你的 2025 大表約 75% 是 BDL、清邁/鹿林春季 ~62–69%。在如此高 censoring 下，
  直接「補 0」或「median 補值」會嚴重低估平均、扭曲比值與相關，且讓 PMF 失真。
  本模組是 Paper1 診斷比值、Paper2 PMF/統計 的「地基」——所有下游都先吃它的輸出。

【底層邏輯】
  ‧ 受限資料不是隨機缺失：它已知「小於某門檻」。
  ‧ ROS：把已檢出值依序排在常態機率圖上，用迴歸外推被censored的下尾，
        保留分布形狀（Helsel 推薦，優於替代法）。
  ‧ Tobit：對數常態假設下，已檢出用 pdf、未檢出用 cdf(<MDL) 寫 likelihood，
        MLE 估真實母體 μ、σ → 還原 lognormal 平均。
  ‧ 不確定度矩陣 (EPA/Polissar)：conc≤MDL → unc=5/6·MDL；
        conc>MDL → unc=√((ef·conc)²+(0.5·MDL)²)。

【用法】
  1. 設定下方「設定區」的 DATA_PATH（CSV：sample_id, 各物種...；BDL 以 0 或空白表示）。
  2. (選) 提供 MDL_PATH（CSV：species, mdl）；不給則以各物種最小正值估計並警告。
  3. 執行：  python bdl_censored.py
  4. 找不到資料檔會自動產生 demo 假資料，先看輸出格式再換真資料。
"""
import os, sys, io
import numpy as np
import pandas as pd
from scipy.stats import norm, linregress
from scipy.optimize import minimize

# ============================ 設定區 ============================
DATA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data_ready", "pfas_2025_wide_gp.csv")  # 真實資料(全2025 G+P)
INDEX_COL   = "sample_id"        # ID 欄名稱；None 表示無
MDL_PATH    = None               # 例如 "mdl.csv" (欄: species, mdl)；None=自動估計
ERROR_FRAC  = 0.10               # 不確定度矩陣的相對分析誤差 (10%)
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "00_bdl")
# ===============================================================

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """讀資料；找不到則產生 demo 假資料 (含人工 BDL)。"""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if INDEX_COL and INDEX_COL in df.columns:
            df = df.set_index(INDEX_COL)
        num = df.select_dtypes(include=[np.number])
        print(f"[讀取] {DATA_PATH}  樣本={len(num)} 物種={num.shape[1]}")
        return num
    print("[警告] 找不到資料檔 → 產生 demo 假資料")
    rng = np.random.default_rng(42)
    n, species = 40, ["PFPeA", "PFHxA", "PFOA", "PFNA", "PFOS", "PFHxS", "6:2 FTS", "FOSA"]
    data = {}
    for j, s in enumerate(species):
        vals = rng.lognormal(mean=0.5 - 0.3 * j, sigma=1.0, size=n)  # 不同檢出率
        thr = np.quantile(vals, 0.3 + 0.07 * j)                      # 人工 MDL
        vals[vals < thr] = 0.0                                       # 設為 BDL(0)
        data[s] = vals
    df = pd.DataFrame(data, index=[f"S{i:03d}" for i in range(n)])
    return df


def get_mdl(df):
    """取得每物種 MDL：優先讀檔，否則以最小正值估計。"""
    if MDL_PATH and os.path.exists(MDL_PATH):
        m = pd.read_csv(MDL_PATH).set_index("species")["mdl"]
        return m.reindex(df.columns).astype(float)
    mdl = {}
    for c in df.columns:
        pos = df[c][df[c] > 0]
        mdl[c] = pos.min() if len(pos) else 1.0
    print("[註] 未提供 MDL，已用各物種最小正值估計 (僅供示範，正式請提供真實 MDL)")
    return pd.Series(mdl)


def detection_frequency(df):
    """偵測頻率 = 大於 0 的比例。"""
    det = (df > 0).mean().rename("detection_freq")
    cnt = (df > 0).sum().rename("n_detected")
    return pd.concat([cnt, det], axis=1)


def ros_impute_col(values, mdl):
    """單一偵測極限的 ROS 填補。values:1D(0=BDL)；回傳填補後陣列。"""
    v = values.astype(float).copy()
    cens = v <= 0
    c, n = int(cens.sum()), len(v)
    if c == 0:
        return v
    det = np.sort(v[~cens])
    if len(det) < 3:                       # 檢出太少 → 退回 LOD/√2
        v[cens] = mdl / np.sqrt(2)
        return v
    pe = c / n                              # 被censored比例
    pos = pe + (1 - pe) * (np.arange(1, len(det) + 1) - 0.5) / len(det)
    z = norm.ppf(pos)
    slope, intercept, *_ = linregress(z, np.log(det))   # ln(濃度) ~ 常態分位
    pos_c = (np.arange(1, c + 1) - 0.5) / n             # 下尾 (0..pe)
    pred = np.sort(np.exp(intercept + slope * norm.ppf(pos_c)))
    pred = np.clip(pred, 1e-12, mdl)        # 受censored值應 < MDL
    v[np.where(cens)[0]] = pred
    return v


def tobit_lognormal(values, mdl):
    """左截尾(對數常態) MLE → 回傳 (母體平均, mu, sigma)。"""
    v = values.astype(float)
    cens = v <= 0
    det = v[~cens]
    if len(det) < 3:
        return np.nan, np.nan, np.nan
    ld = np.log(det)
    lt = np.log(mdl)

    def negll(p):
        mu, logs = p
        s = np.exp(logs)
        ll = norm.logpdf(ld, mu, s).sum() + norm.logcdf((lt - mu) / s).sum() * cens.sum()
        return -ll
    res = minimize(negll, [ld.mean(), np.log(ld.std() + 1e-6)], method="Nelder-Mead")
    mu, s = res.x[0], np.exp(res.x[1])
    return float(np.exp(mu + s ** 2 / 2)), float(mu), float(s)


def uncertainty_matrix(df, mdl):
    """EPA/Polissar 不確定度矩陣 s_ij (供 PMF)。"""
    s = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for c in df.columns:
        m = mdl[c]
        x = df[c].values.astype(float)
        u = np.where(x <= 0, (5.0 / 6.0) * m,
                     np.sqrt((ERROR_FRAC * x) ** 2 + (0.5 * m) ** 2))
        s[c] = u
    return s


def main():
    df = load_data()
    mdl = get_mdl(df)

    det = detection_frequency(df)
    det.to_csv(os.path.join(OUTPUT_DIR, "detection_frequency.csv"))
    print("\n[偵測頻率]\n", det.to_string())

    # 替代法 (供比較)
    for name, factor in [("subst_half", 0.5), ("subst_sqrt2", 1 / np.sqrt(2))]:
        sub = df.copy()
        for c in df.columns:
            sub.loc[sub[c] <= 0, c] = mdl[c] * factor
        sub.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"))

    # ROS 填補
    ros = df.copy().astype(float)
    for c in df.columns:
        ros[c] = ros_impute_col(df[c].values, mdl[c])
    ros.to_csv(os.path.join(OUTPUT_DIR, "ros_imputed.csv"))

    # Tobit 摘要 vs 各法平均
    rows = []
    for c in df.columns:
        tmean, mu, sg = tobit_lognormal(df[c].values, mdl[c])
        rows.append({
            "species": c, "detection_freq": det.loc[c, "detection_freq"],
            "mean_subst_half": df[c].where(df[c] > 0, mdl[c] * 0.5).mean(),
            "mean_ros": ros[c].mean(),
            "mean_tobit_lognorm": tmean,
        })
    summ = pd.DataFrame(rows).set_index("species")
    summ.to_csv(os.path.join(OUTPUT_DIR, "censored_summary.csv"))
    print("\n[censored 摘要：不同方法的平均]\n", summ.round(3).to_string())

    # 不確定度矩陣
    unc = uncertainty_matrix(df, mdl)
    unc.to_csv(os.path.join(OUTPUT_DIR, "uncertainty_matrix.csv"))

    # 偵測熱圖
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(min(14, 1 + 0.5 * df.shape[1]), min(10, 1 + 0.18 * len(df))))
        ax.imshow((df.values > 0).astype(int), aspect="auto", cmap="Greens", vmin=0, vmax=1)
        ax.set_xticks(range(df.shape[1]))
        ax.set_xticklabels(df.columns, rotation=90, fontsize=7)
        ax.set_title("偵測(綠) / 未檢出(白)  Detection map")
        ax.set_ylabel("樣本")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "detection_heatmap.png"), dpi=150)
        print("\n[圖] detection_heatmap.png 已存。")
    except Exception as e:
        print("[圖略過]", e)

    print(f"\n✓ 完成，輸出於 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
