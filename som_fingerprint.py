# -*- coding: utf-8 -*-
"""
som_fingerprint.py — 自組織映射(SOM)：PFAS 指紋拓樸
================================================================
【做什麼】
  對多站多季 PFAS 指紋(建議先經 coda_transform.py 的 CLR)訓練 SOM，輸出：
    1. 每樣本的 BMU 節點 + 站/季標註 (som_bmu.csv)
    2. Component planes：每物種在拓樸上的權重熱圖 (som_components.png)
    3. U-matrix：節點間距離(看分群邊界) (som_umatrix.png)
    4. 站/季在拓樸上的落點圖 (som_hits_site.png)
    5. 節點再分群成「PFAS regime」 (som_node_clusters.csv)

【為何需要】(Paper2 核心、你指定的 SOM 主場)
  ‧ 把高維 PFAS 指紋非線性映到 2D，直觀看不同站/季是否屬不同來源型態。
  ‧ Component planes 揭露哪些物種「共峰」(同源/同過程)，比相關矩陣更能看非線性結構。
  ‧ 節點分群 → 少數 regime，再用 pfas_diagnostics.py 的比值賦予來源意義。

【底層邏輯】
  ‧ SOM：每節點有一組權重(碼書向量)。對每筆樣本找最近節點(BMU)，並把 BMU 及鄰域
    朝樣本方向更新；學習率與鄰域半徑隨時間衰減 → 形成保留拓樸的有序映射。
  ‧ 輸入先標準化(z-score)，避免高濃度物種主導距離。

【用法】
  1. DATA_PATH 建議用 outputs/clr_transformed.csv；或原始濃度(會自動標準化)。
  2. (選) 設 SITE_COL/SEASON_COL 以著色拓樸；這兩欄需在資料中。
  3. python som_fingerprint.py  （無檔則 demo）
"""
import os, sys, io
import numpy as np
import pandas as pd

# ============================ 設定區 ============================
DATA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data_ready", "pfas_2025_wide_gp.csv")  # 真實資料(全2025 G+P,自動標準化)
INDEX_COL   = "sample_id"
SITE_COL    = "site_cn"          # 站名欄(中文,供 FOCUS_PAIR 比對)
SEASON_COL  = "season"
GRID        = (7, 7)              # SOM 節點數 (m × n)
ITERS       = 5000
LR0, SIG0   = 0.5, None           # SIG0=None → 自動取 max(GRID)/2
N_REGIME    = 4                   # 節點再分群數
FOCUS_PAIR  = ("清邁", "鹿林山")   # 焦點站對：量化兩站指紋重疊度(LRT 同源證據)；找不到則跳過
SEED        = 42
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "09_som")
# ===============================================================
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)


class SOM:
    def __init__(self, m, n, dim):
        self.m, self.n, self.dim = m, n, dim
        self.W = rng.normal(0, 1, size=(m * n, dim))
        self.coords = np.array([(i, j) for i in range(m) for j in range(n)], float)

    def _bmu(self, x):
        return int(np.argmin(((self.W - x) ** 2).sum(1)))

    def train(self, X, iters, lr0, sig0):
        for t in range(iters):
            lr = lr0 * np.exp(-t / iters)
            sig = max(sig0 * np.exp(-t / iters), 0.5)
            x = X[rng.integers(len(X))]
            b = self._bmu(x)
            d2 = ((self.coords - self.coords[b]) ** 2).sum(1)
            h = np.exp(-d2 / (2 * sig ** 2))
            self.W += lr * h[:, None] * (x - self.W)

    def bmus(self, X):
        return np.array([self._bmu(x) for x in X])

    def umatrix(self):
        U = np.zeros(self.m * self.n)
        for k in range(self.m * self.n):
            nb = ((self.coords - self.coords[k]) ** 2).sum(1)
            neigh = (nb > 0) & (nb <= 2)
            U[k] = np.mean(np.sqrt(((self.W[neigh] - self.W[k]) ** 2).sum(1))) if neigh.any() else 0
        return U.reshape(self.m, self.n)


def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"[讀取] {DATA_PATH} 樣本={len(df)}")
        return df
    print("[警告] 找不到資料檔 → demo (3 站不同指紋)")
    sites = np.repeat(["清邁", "鹿林", "楠梓"], 12)
    sp = ["PFPeA", "PFHxA", "PFOA", "PFNA", "PFOS", "PFHxS", "6:2 FTS", "FOSA"]
    blocks = {"清邁": [3, 3, 2, 1, 0.5, 0.5, 2, 1],   # 短鏈/前驅物多
              "鹿林": [2, 2, 1, 1, 0.3, 0.3, 1, 0.5],
              "楠梓": [1, 1, 2, 1, 3, 2, 0.5, 0.3]}    # 長鏈/PFSA 多(都市)
    X = []
    for s in sites:
        X.append(np.array(blocks[s]) * rng.lognormal(0, 0.25, len(sp)))
    df = pd.DataFrame(X, columns=sp)
    df.insert(0, "season", np.tile(["春"], len(sites)))
    df.insert(0, "site", sites)
    df.insert(0, "sample_id", [f"S{i:02d}" for i in range(len(sites))])
    return df


def overlap_analysis(out, node_regime, n_regime):
    """步驟三的量化：站×regime 落點 + 站對站指紋重疊度。
    重疊度用『直方圖交集』：兩站在各 regime 的樣本比例分佈，逐 regime 取較小值再加總。
      值域 0~1；1=兩站完全落在相同 regime 組合(指紋型態一致)；0=完全不重疊。
      這是『清邁與鹿林是否同源』的客觀量化(肉眼看落點圖的替代)。"""
    out = out.copy()
    out["regime"] = out["bmu"].map(dict(zip(node_regime["node"], node_regime["regime"])))

    # 站 × regime 計數表(各站樣本落在哪些 regime)
    ct = pd.crosstab(out[SITE_COL], out["regime"])
    ct = ct.reindex(columns=range(n_regime), fill_value=0)
    ct.columns = [f"regime{c}" for c in ct.columns]

    # 各站轉成比例分佈 → 兩兩算直方圖交集相似度
    prop = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    sites = prop.index.tolist()
    sim = pd.DataFrame(np.eye(len(sites)), index=sites, columns=sites)
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            s = float(np.minimum(prop.loc[a], prop.loc[b]).sum())
            sim.loc[a, b] = sim.loc[b, a] = round(s, 3)
    return ct, sim


def main():
    df = load_data()
    meta_cols = [c for c in (INDEX_COL, SITE_COL, SEASON_COL) if c in df.columns]
    feats = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].values.astype(float)
    X = np.nan_to_num(X, nan=np.nanmedian(X))
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)            # z-score

    m, n = GRID
    sig0 = SIG0 if SIG0 else max(m, n) / 2
    som = SOM(m, n, X.shape[1])
    som.train(X, ITERS, LR0, sig0)
    bmu = som.bmus(X)

    out = df[meta_cols].copy()
    out["bmu"] = bmu
    out["bmu_row"], out["bmu_col"] = bmu // n, bmu % n
    out.to_csv(os.path.join(OUTPUT_DIR, "som_bmu.csv"), index=False)

    # 節點再分群
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=min(N_REGIME, m * n), n_init=10, random_state=SEED).fit(som.W)
    nodes = pd.DataFrame({"node": range(m * n), "row": som.coords[:, 0].astype(int),
                          "col": som.coords[:, 1].astype(int), "regime": km.labels_})
    nodes.to_csv(os.path.join(OUTPUT_DIR, "som_node_clusters.csv"), index=False)
    print(f"[SOM] {m}x{n} 訓練完成；節點分 {N_REGIME} regime；BMU 已輸出。")

    # ---- 步驟三：站點落點的量化解讀 ----
    if SITE_COL in df.columns:
        ct, sim = overlap_analysis(out, nodes, N_REGIME)
        ct.to_csv(os.path.join(OUTPUT_DIR, "som_site_regime.csv"))
        sim.to_csv(os.path.join(OUTPUT_DIR, "som_site_overlap.csv"))
        print("\n[站 × regime 落點分佈]\n", ct.to_string())
        print("\n[站對站指紋重疊度 0~1 (1=指紋型態一致)]\n", sim.to_string())
        a, b = FOCUS_PAIR
        if a in sim.index and b in sim.index:
            v = sim.loc[a, b]
            verdict = ("高度重疊 → 指紋型態一致，支持 LRT 同源" if v >= 0.6 else
                       "中度重疊 → 部分共享型態，需配合軌跡/診斷比值佐證" if v >= 0.3 else
                       "低度重疊 → 指紋型態分離，難以單由 SOM 主張同源")
            print(f"\n[焦點站對] {a} ↔ {b} 重疊度 = {v:.3f} → {verdict}")
        else:
            print(f"\n[焦點站對] 資料中找不到 {FOCUS_PAIR} → 跳過(改 FOCUS_PAIR 設定)")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]; plt.rcParams["axes.unicode_minus"] = False
        # Component planes
        ncol = min(4, len(feats)); nrow = int(np.ceil(len(feats) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.6 * nrow))
        for ax, k in zip(np.ravel(axes), range(len(feats))):
            ax.imshow(som.W[:, k].reshape(m, n), cmap="viridis")
            ax.set_title(feats[k], fontsize=8); ax.axis("off")
        for ax in np.ravel(axes)[len(feats):]:
            ax.axis("off")
        fig.suptitle("Component planes (各物種權重)"); fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "som_components.png"), dpi=150)

        # U-matrix
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(som.umatrix(), cmap="bone_r"); fig.colorbar(im, ax=ax)
        ax.set_title("U-matrix (節點距離；亮=分群邊界)")
        fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "som_umatrix.png"), dpi=150)

        # 站落點
        if SITE_COL in df.columns:
            fig, ax = plt.subplots(figsize=(5.5, 5))
            cats = df[SITE_COL].astype("category")
            sc = ax.scatter(out["bmu_col"] + rng.normal(0, 0.12, len(out)),
                            out["bmu_row"] + rng.normal(0, 0.12, len(out)),
                            c=cats.cat.codes, cmap="tab10", s=40)
            ax.set_title("各站在 SOM 拓樸落點"); ax.invert_yaxis()
            handles = [plt.Line2D([], [], marker="o", ls="", color=plt.cm.tab10(i / 10),
                       label=c) for i, c in enumerate(cats.cat.categories)]
            ax.legend(handles=handles, fontsize=8)
            fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "som_hits_site.png"), dpi=150)

        # 季節落點(若有 season 欄)
        if SEASON_COL in df.columns:
            fig, ax = plt.subplots(figsize=(5.5, 5))
            cats = df[SEASON_COL].astype("category")
            ax.scatter(out["bmu_col"] + rng.normal(0, 0.12, len(out)),
                       out["bmu_row"] + rng.normal(0, 0.12, len(out)),
                       c=cats.cat.codes, cmap="Set2", s=40)
            ax.set_title("各季在 SOM 拓樸落點"); ax.invert_yaxis()
            handles = [plt.Line2D([], [], marker="o", ls="", color=plt.cm.Set2(i / 8),
                       label=c) for i, c in enumerate(cats.cat.categories)]
            ax.legend(handles=handles, fontsize=8)
            fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "som_hits_season.png"), dpi=150)
        print("[圖] som_components / som_umatrix / som_hits_site / som_hits_season 已存。")
    except Exception as e:
        print("[圖略過]", repr(e)[:150])

    print(f"\n✓ 完成，輸出於 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
