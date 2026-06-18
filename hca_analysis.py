# -*- coding: utf-8 -*-
"""
HCA 階層式分群分析 (Hierarchical Clustering)
==============================================================
輸入資料格式 (CSV)：
  - 第一列為欄位名稱 (header)
  - 每一列 (row) = 一個樣本 / 觀測值
  - 特徵欄位需為「數值」(非數值欄會自動剔除並警告)
  - 可選 ID 欄位 → INDEX_COL；可選目標欄 → TARGET_COL (排除於分析)
  - 允許缺值 (NaN) → 自動以該欄中位數補值

執行：
  1. 修改下方 DATA_PATH (不存在則自動用 demo 資料)
  2. 先看 dendrogram 與 silhouette 圖決定群數，再把 N_CLUSTERS 設好重跑
  3. python hca_analysis.py

輸出 (存到 outputs/)：
  hca_dendrogram.png / hca_cluster_evaluation.png / hca_cluster_profile.png
  hca_results.csv     原始資料 + 每個樣本所屬群號 (可當回歸的類別特徵)
==============================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# 確保在 Windows cp950 主控台也能輸出中文與符號 (否則 ⚠ 等字元會崩潰)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 設定區 ────────────────────────────────────────────
DATA_PATH      = "your_data.csv"   # 你的資料路徑；不存在則用 demo 資料
INDEX_COL      = None              # ID/索引欄名稱，無則 None
TARGET_COL     = None              # 目標欄名稱 (排除於分析)，無則 None
LINKAGE_METHOD = "ward"            # ward / complete / average / single
MAX_CLUSTERS   = 10                # 評估 silhouette 的最大群數
N_CLUSTERS     = 4                 # 最終切割群數 (看 dendrogram 後決定)
OUTPUT_DIR     = "outputs"
RANDOM_STATE   = 42

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def make_demo_data(n_samples=200, n_features=8, n_groups=3, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 5, size=(n_groups, n_features))
    rows, ids = [], []
    for i in range(n_samples):
        g = i % n_groups
        rows.append(centers[g] + rng.normal(0, 1.5, size=n_features))
        ids.append(f"S{i+1:03d}")
    cols = [f"feature_{j+1}" for j in range(n_features)]
    return pd.DataFrame(rows, columns=cols, index=pd.Index(ids, name="sample_id"))


def load_data(path, index_col, target_col):
    if not os.path.exists(path):
        print(f"⚠ 找不到資料檔 '{path}' → 改用自動產生的 demo 資料示範。")
        df = make_demo_data()
    else:
        df = pd.read_csv(path, index_col=index_col)

    y = None
    if target_col and target_col in df.columns:
        y = df[target_col]
        df = df.drop(columns=[target_col])

    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"⚠ 偵測到非數值欄並已剔除：{non_numeric}")
        df = df.select_dtypes(include=[np.number])

    if df.shape[1] < 2:
        raise ValueError("可用的數值特徵少於 2 欄，無法分群。請檢查輸入資料。")

    n_nan = int(df.isna().sum().sum())
    if n_nan > 0:
        print(f"⚠ 偵測到 {n_nan} 個缺值 → 以各欄中位數補值。")
        df = df.fillna(df.median(numeric_only=True))

    return df, y


def describe_data(X):
    print("── 資料摘要 ──────────────────────────────")
    print(f"樣本數 (rows)   : {X.shape[0]}")
    print(f"特徵數 (cols)   : {X.shape[1]}")
    print(f"特徵欄位        : {list(X.columns)}")
    if X.shape[0] > 2000:
        print("⚠ 樣本數較大，linkage 記憶體需求約 O(n²)，可能較慢。")
    print("──────────────────────────────────────────")


def preprocess(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def compute_linkage(X_scaled):
    return linkage(X_scaled, method=LINKAGE_METHOD)


def plot_dendrogram(Z, out_dir, n_clusters, truncate_p=30):
    fig, ax = plt.subplots(figsize=(14, 6))
    # color_threshold 設在切成 n_clusters 群的高度，讓配色與最終切割一致
    if len(Z) >= n_clusters >= 1:
        ct = Z[-(n_clusters - 1), 2] if n_clusters > 1 else 0
    else:
        ct = 0
    dendrogram(Z, truncate_mode="lastp", p=truncate_p, leaf_rotation=90,
               leaf_font_size=9, show_contracted=True, color_threshold=ct, ax=ax)
    if n_clusters > 1:
        ax.axhline(ct, color="red", linestyle="--", alpha=0.6,
                   label=f"切成 {n_clusters} 群")
        ax.legend()
    ax.set_title(f"HCA 樹狀圖 (method={LINKAGE_METHOD})")
    ax.set_xlabel("樣本 / 群")
    ax.set_ylabel("距離")
    plt.tight_layout()
    path = os.path.join(out_dir, "hca_dendrogram.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"圖表已儲存：{path}")


def evaluate_clusters(X_scaled, Z, out_dir):
    """用已算好的 Z + fcluster 評估不同群數 (與樹狀圖一致，速度快)"""
    n_samples = X_scaled.shape[0]
    max_k = min(MAX_CLUSTERS, n_samples - 1)
    k_range = list(range(2, max_k + 1))
    sil_scores, ch_scores = [], []

    # silhouette 對大樣本較慢 → 取樣計算
    sample_size = 1000 if n_samples > 1000 else None

    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2:        # 防呆：實際群數不足
            sil_scores.append(np.nan)
            ch_scores.append(np.nan)
            continue
        sil_scores.append(silhouette_score(
            X_scaled, labels, sample_size=sample_size, random_state=RANDOM_STATE))
        ch_scores.append(calinski_harabasz_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(k_range, sil_scores, marker="o", color="steelblue")
    axes[0].set_xlabel("群數")
    axes[0].set_ylabel("Silhouette 分數")
    axes[0].set_title("Silhouette 分數 (越高越好)")

    axes[1].plot(k_range, ch_scores, marker="o", color="tomato")
    axes[1].set_xlabel("群數")
    axes[1].set_ylabel("Calinski-Harabasz 分數")
    axes[1].set_title("Calinski-Harabasz 分數 (越高越好)")

    plt.tight_layout()
    path = os.path.join(out_dir, "hca_cluster_evaluation.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"圖表已儲存：{path}")

    if np.all(np.isnan(sil_scores)):
        return
    best_k_sil = k_range[int(np.nanargmax(sil_scores))]
    best_k_ch = k_range[int(np.nanargmax(ch_scores))]
    print(f"Silhouette 建議群數 = {best_k_sil}")
    print(f"Calinski-Harabasz 建議群數 = {best_k_ch}")


def run_hca(Z, n_clusters):
    """直接用 linkage Z 切割，與樹狀圖完全一致"""
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels


def plot_cluster_profile(X_scaled_df, labels, out_dir):
    df_tmp = X_scaled_df.copy()
    df_tmp["Cluster"] = labels
    profile = df_tmp.groupby("Cluster").mean()

    ax = profile.T.plot(kind="bar", figsize=(14, 5), colormap="tab10")
    ax.set_title("HCA 各群特徵輪廓 (標準化後均值)")
    ax.set_ylabel("均值 (標準化)")
    ax.set_xlabel("特徵")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="群", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(out_dir, "hca_cluster_profile.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"圖表已儲存：{path}")


def save_results(X, labels, out_dir):
    df_out = X.copy()
    df_out["HCA_Cluster"] = labels
    df_out.to_csv(os.path.join(out_dir, "hca_results.csv"))
    print(f"結果已儲存：{out_dir}/hca_results.csv")
    print("各群樣本數：")
    print(pd.Series(labels).value_counts().sort_index().rename("Count").to_string())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y = load_data(DATA_PATH, INDEX_COL, TARGET_COL)
    describe_data(X)

    if N_CLUSTERS >= X.shape[0]:
        raise ValueError(f"N_CLUSTERS={N_CLUSTERS} 不可 >= 樣本數 {X.shape[0]}。")

    X_scaled, _ = preprocess(X)

    print("計算 linkage matrix...")
    Z = compute_linkage(X_scaled)

    plot_dendrogram(Z, OUTPUT_DIR, N_CLUSTERS)
    evaluate_clusters(X_scaled, Z, OUTPUT_DIR)

    print(f"\n以 N_CLUSTERS={N_CLUSTERS} 進行切割...")
    labels = run_hca(Z, N_CLUSTERS)

    if len(np.unique(labels)) >= 2:
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        print(f"Silhouette 分數      : {sil:.4f}")
        print(f"Calinski-Harabasz 分數: {ch:.4f}")

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    plot_cluster_profile(X_scaled_df, labels, OUTPUT_DIR)
    save_results(X, labels, OUTPUT_DIR)
    print("\n✓ HCA 完成。hca_results.csv 的 HCA_Cluster 欄可當回歸的類別特徵。")


if __name__ == "__main__":
    main()
