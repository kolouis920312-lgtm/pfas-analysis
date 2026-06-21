# -*- coding: utf-8 -*-
"""
PCA 主成分分析
==============================================================
輸入資料格式 (CSV)：
  - 第一列為欄位名稱 (header)
  - 每一列 (row) = 一個樣本 / 觀測值
  - 特徵欄位需為「數值」(非數值欄會自動剔除並警告)
  - 可選的 ID 欄位 → 設定 INDEX_COL (例如 "sample_id")
  - 可選的目標欄位 → 設定 TARGET_COL (會被排除在分析之外)
  - 允許缺值 (NaN) → 自動以該欄中位數補值並警告

範例：
  sample_id, gene_A, gene_B, gene_C, label
  S001,      1.23,   4.56,   7.89,   0
  S002,      2.34,   5.67,   8.90,   1

執行：
  1. 修改下方「設定區」的 DATA_PATH
  2. python pca_analysis.py
  3. 若 DATA_PATH 不存在 → 自動產生 demo 資料讓你先看效果

輸出 (全部存到 outputs/ 資料夾)：
  pca_components.csv  各樣本降維後座標 (給後續分群 / 回歸用)
  pca_loadings.csv    各特徵對每個 PC 的權重
  pca_explained_variance.png / pca_biplot.png
==============================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 確保在 Windows cp950 主控台也能輸出中文與符號 (否則 ⚠ 等字元會崩潰)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 設定區 ────────────────────────────────────────────
DATA_PATH    = "your_data.csv"   # 你的資料路徑；不存在則用 demo 資料
INDEX_COL    = None              # ID/索引欄名稱，無則 None
TARGET_COL   = None              # 目標欄名稱 (排除於分析)，無則 None
N_COMPONENTS = None              # 保留主成分數；None = 全部保留 (看 elbow 再決定)
OUTPUT_DIR   = "outputs"         # 圖表與結果輸出資料夾
RANDOM_STATE = 42

# 中文字型 (Windows)，避免圖中中文變方框
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def make_demo_data(n_samples=200, n_features=8, n_groups=3, seed=RANDOM_STATE):
    """產生有群組結構的示範資料 (找不到 DATA_PATH 時使用)"""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 5, size=(n_groups, n_features))
    rows, ids = [], []
    for i in range(n_samples):
        g = i % n_groups
        rows.append(centers[g] + rng.normal(0, 1.5, size=n_features))
        ids.append(f"S{i+1:03d}")
    cols = [f"feature_{j+1}" for j in range(n_features)]
    df = pd.DataFrame(rows, columns=cols, index=pd.Index(ids, name="sample_id"))
    return df


def load_data(path, index_col, target_col):
    """讀檔 + 清洗：剔除非數值欄、NaN 補值。回傳 (X 數值DataFrame, y 或 None)"""
    if not os.path.exists(path):
        print(f"⚠ 找不到資料檔 '{path}' → 改用自動產生的 demo 資料示範。")
        df = make_demo_data()
    else:
        df = pd.read_csv(path, index_col=index_col)

    y = None
    if target_col and target_col in df.columns:
        y = df[target_col]
        df = df.drop(columns=[target_col])

    # 剔除非數值欄
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"⚠ 偵測到非數值欄並已剔除：{non_numeric}")
        df = df.select_dtypes(include=[np.number])

    if df.shape[1] < 2:
        raise ValueError("可用的數值特徵少於 2 欄，無法進行 PCA。請檢查輸入資料。")

    # NaN 補值 (中位數)
    n_nan = int(df.isna().sum().sum())
    if n_nan > 0:
        print(f"⚠ 偵測到 {n_nan} 個缺值 → 以各欄中位數補值。")
        df = df.fillna(df.median(numeric_only=True))

    # 零變異欄警告 (對標準化無意義)
    zero_var = df.columns[df.std(numeric_only=True) == 0].tolist()
    if zero_var:
        print(f"⚠ 以下欄位變異為 0，建議移除：{zero_var}")

    return df, y


def describe_data(X):
    print("── 資料摘要 ──────────────────────────────")
    print(f"樣本數 (rows)   : {X.shape[0]}")
    print(f"特徵數 (cols)   : {X.shape[1]}")
    print(f"特徵欄位        : {list(X.columns)}")
    print("──────────────────────────────────────────")


def preprocess(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def run_pca(X_scaled, n_components):
    # 防呆：n_components 不可超過 min(樣本數, 特徵數)
    max_comp = min(X_scaled.shape)
    if n_components is not None and n_components > max_comp:
        print(f"⚠ N_COMPONENTS={n_components} 超過上限 {max_comp}，已自動下調。")
        n_components = max_comp
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca


def plot_explained_variance(pca, out_dir):
    ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    n = len(ratio)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(1, n + 1), ratio, alpha=0.75, color="steelblue")
    axes[0].set_xlabel("主成分 (PC)")
    axes[0].set_ylabel("解釋變異比例")
    axes[0].set_title("各主成分解釋變異")

    axes[1].plot(range(1, n + 1), cumulative, marker="o", color="tomato")
    axes[1].axhline(0.90, linestyle="--", color="gray", label="90% 門檻")
    axes[1].axhline(0.95, linestyle="--", color="orange", label="95% 門檻")
    axes[1].set_xlabel("主成分數量")
    axes[1].set_ylabel("累積解釋變異")
    axes[1].set_title("累積解釋變異 (Elbow)")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "pca_explained_variance.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"圖表已儲存：{path}")

    # 達到門檻所需的主成分數
    for thr in (0.90, 0.95):
        idx = int(np.searchsorted(cumulative, thr)) + 1
        if idx <= n:
            print(f"  達到 {thr:.0%} 累積變異需要 {idx} 個主成分")


def plot_biplot(X_pca, pca, feature_names, out_dir, n_arrows=10):
    """PC1 vs PC2 biplot；箭頭挑「對 PC1/PC2 貢獻最大」的前 n 個特徵"""
    loadings = pca.components_.T  # (n_features, n_components)
    importance = np.sqrt(loadings[:, 0] ** 2 + loadings[:, 1] ** 2)
    top_idx = np.argsort(importance)[::-1][:n_arrows]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4, s=15, color="steelblue")

    scale = np.abs(X_pca[:, :2]).max()
    for i in top_idx:
        ax.arrow(0, 0, loadings[i, 0] * scale * 0.8, loadings[i, 1] * scale * 0.8,
                 head_width=scale * 0.02, color="tomato", alpha=0.8, length_includes_head=True)
        ax.text(loadings[i, 0] * scale * 0.88, loadings[i, 1] * scale * 0.88,
                feature_names[i], fontsize=9, color="darkred")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA Biplot (前 %d 個重要特徵)" % len(top_idx))
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(out_dir, "pca_biplot.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"圖表已儲存：{path}")


def report_top_features(pca, feature_names, top_n=5):
    """印出每個 PC 載荷量最大的前幾個特徵，幫助解讀"""
    print("── 各主成分主要貢獻特徵 ──────────────────")
    n_show = min(3, pca.n_components_)  # 只印前 3 個 PC
    for pc in range(n_show):
        loading = pca.components_[pc]
        order = np.argsort(np.abs(loading))[::-1][:top_n]
        items = [f"{feature_names[i]}({loading[i]:+.2f})" for i in order]
        print(f"PC{pc+1}: " + ", ".join(items))
    print("──────────────────────────────────────────")


def save_results(X_pca, pca, feature_names, original_index, out_dir):
    cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    pd.DataFrame(X_pca, columns=cols, index=original_index).to_csv(
        os.path.join(out_dir, "pca_components.csv"))

    # loadings 的 index 用「特徵名稱」(修正原本的 range bug)
    pd.DataFrame(pca.components_.T, columns=cols, index=feature_names).to_csv(
        os.path.join(out_dir, "pca_loadings.csv"))
    print(f"結果已儲存：{out_dir}/pca_components.csv、pca_loadings.csv")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y = load_data(DATA_PATH, INDEX_COL, TARGET_COL)
    describe_data(X)
    feature_names = list(X.columns)

    X_scaled, _ = preprocess(X)
    pca, X_pca = run_pca(X_scaled, N_COMPONENTS)

    print(f"保留主成分數：{pca.n_components_}")
    print(f"總解釋變異  ：{pca.explained_variance_ratio_.sum():.4f}")

    plot_explained_variance(pca, OUTPUT_DIR)
    report_top_features(pca, feature_names)
    if X_pca.shape[1] >= 2:
        plot_biplot(X_pca, pca, feature_names, OUTPUT_DIR)

    save_results(X_pca, pca, feature_names, X.index, OUTPUT_DIR)
    print("\n✓ PCA 完成。pca_components.csv 可直接作為分群 / 回歸的輸入特徵。")


if __name__ == "__main__":
    main()
