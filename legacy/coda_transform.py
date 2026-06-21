# -*- coding: utf-8 -*-
"""
coda_transform.py — 組成資料分析(CoDA)轉換：零替換 + CLR / ILR
================================================================
【做什麼】
  將 PFAS 濃度「組成(成分占比)」資料做：
    1. 高 BDL 穩健零替換 (multiplicative replacement) → zero_replaced.csv
    2. CLR(中心對數比) 轉換 → clr_transformed.csv   (餵 PCA / SOM / 分群)
    3. (選) ILR(等距對數比) 轉換 → ilr_transformed.csv
  另輸出零替換比例報告。

【為何需要】
  PFAS 指紋本質是「組成資料」(各物種占 ΣPFAS 的比例)，有「閉合(closure)」限制：
  一個分量上升、其餘被迫下降 → 直接拿原始濃度做 PCA/相關/SOM 會產生**假相關**、
  違反 subcompositional coherence。CLR/ILR 把資料搬到實數空間，移除閉合偏誤。
  這是 Paper2 的 SOM 指紋拓樸、PCA-APCS-MLR 之前的**正確前處理**。

【底層邏輯】
  ‧ 閉合：成分 x 的資訊在「比值」而非絕對值 → 用對數比座標。
  ‧ CLR_i = ln( x_i / 幾何平均(x) )；總和為 0(落在單純形的對數平面)。
  ‧ ILR：在該平面取「正交單位基底」得 D-1 個獨立座標(本檔用 Helmert 基底)，
        避免 CLR 的奇異共變(適合需要滿秩的模型)。
  ‧ 零替換：log(0) 不存在；以 0.65×偵測極限做乘法替換，等比例縮放非零部分以維持閉合
        (Martín-Fernández 法)。**警告**：BDL>~70% 時 CLR/SOM 結果對替換敏感，須謹慎詮釋。

【用法】
  1. 設定 DATA_PATH（CSV：sample_id, 各物種...；BDL 以 0/空白）。
  2. python coda_transform.py  （無檔則用 demo 假資料）
  3. 把 clr_transformed.csv 當 pca_analysis.py / som_fingerprint.py 的輸入。
"""
import os, sys, io
import numpy as np
import pandas as pd

# ============================ 設定區 ============================
DATA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data_ready", "pfas_2025_wide_gp.csv")  # 真實資料(全2025 G+P)
INDEX_COL   = "sample_id"
DL_FACTOR   = 0.65          # 零替換值 = DL_FACTOR × 偵測極限(以各物種最小正值估)
DO_ILR      = True
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "02_coda")
# ===============================================================

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if INDEX_COL and INDEX_COL in df.columns:
            df = df.set_index(INDEX_COL)
        df = df.select_dtypes(include=[np.number])
        print(f"[讀取] {DATA_PATH} 樣本={len(df)} 物種={df.shape[1]}")
        return df
    print("[警告] 找不到資料檔 → 產生 demo 假資料")
    rng = np.random.default_rng(11)
    sp = ["PFPeA", "PFHxA", "PFOA", "PFNA", "PFOS", "PFHxS", "6:2 FTS", "FOSA"]
    X = rng.lognormal(0.0, 1.0, size=(24, len(sp)))
    X[rng.random(X.shape) < 0.35] = 0.0
    return pd.DataFrame(X, columns=sp, index=[f"S{i:03d}" for i in range(24)])


def mult_replacement(X, dl, factor=DL_FACTOR):
    """乘法零替換，逐列(樣本)維持閉合。X:濃度矩陣；dl:每欄偵測極限。"""
    X = X.astype(float).copy()
    delta = factor * dl                       # 替換值(原尺度)
    out = np.zeros_like(X)
    for r in range(X.shape[0]):
        row = X[r].copy()
        tot = row.sum()
        if tot <= 0:
            out[r] = factor * dl              # 整列皆 0 的極端情況
            continue
        comp = row / tot
        d = delta / tot                       # 替換值(閉合尺度)
        zero = comp <= 0
        comp[zero] = d[zero]
        comp[~zero] = comp[~zero] * (1.0 - d[zero].sum())
        out[r] = comp * tot                   # 還原原尺度
    return out


def clr(Xpos):
    L = np.log(Xpos)
    return L - L.mean(axis=1, keepdims=True)


def helmert_basis(D):
    """D 維單純形對數平面的正交單位基底 V (D × (D-1))，用 Helmert 對比。"""
    V = np.zeros((D, D - 1))
    for i in range(1, D):
        h = np.zeros(D)
        h[:i] = 1.0
        h[i] = -i
        V[:, i - 1] = h / np.sqrt(i * (i + 1))
    return V


def main():
    df = load_data()
    dl = np.array([df[c][df[c] > 0].min() if (df[c] > 0).any() else 1.0 for c in df.columns])

    zero_pct = float((df.values <= 0).mean() * 100)
    print(f"[零/BDL 比例] {zero_pct:.1f}%  (>70% 時 CLR 結果對替換敏感，請謹慎)")

    Xpos = mult_replacement(df.values, dl)
    zr = pd.DataFrame(Xpos, index=df.index, columns=df.columns)
    zr.to_csv(os.path.join(OUTPUT_DIR, "zero_replaced.csv"))

    C = clr(Xpos)
    clr_df = pd.DataFrame(C, index=df.index, columns=[f"clr_{c}" for c in df.columns])
    clr_df.to_csv(os.path.join(OUTPUT_DIR, "clr_transformed.csv"))
    print(f"[CLR] 完成 → clr_transformed.csv  (每列總和≈0：{np.allclose(C.sum(axis=1), 0)})")

    if DO_ILR:
        V = helmert_basis(df.shape[1])
        I = C @ V
        ilr_df = pd.DataFrame(I, index=df.index, columns=[f"ilr{i+1}" for i in range(I.shape[1])])
        ilr_df.to_csv(os.path.join(OUTPUT_DIR, "ilr_transformed.csv"))
        print(f"[ILR] 完成 → ilr_transformed.csv  維度={I.shape[1]} (Helmert 基底)")

    print(f"\n✓ 完成，輸出於 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
