# -*- coding: utf-8 -*-
"""
K-means 分群 (用於 PCA 降維後的結果) + 分組品質量化
==============================================================
用途：PCA 只負責「降維」不會分組；本程式在 PCA 座標上做 K-means 分組，
      並用 4 個指標「量化」分組好壞、自動建議最佳群數 k。

量化指標：
  Inertia (Elbow)     群內平方和，畫圖找手肘轉折
  Silhouette 輪廓係數  越高越好 (-1~1)，看樣本與自己群 vs 鄰群的緊密度
  Calinski-Harabasz   越高越好，群間/群內變異比
  Davies-Bouldin      越低越好，群間相似度 (越低代表分得越開)

輸入：
  預設讀 PCA 程式的輸出 outputs/pca_components.csv
  (第一欄為樣本 ID，其餘為 PC1, PC2, ... )
  找不到檔案 → 自動產生 demo 資料示範。

執行：
  1. 先跑 pca_analysis.py 產生 pca_components.csv
  2. (可選) 用 USE_N_PCS 只取前幾個主成分分群
  3. 看指標圖決定 k，設好 N_CLUSTERS 重跑
  4. python kmeans_pca.py

輸出 (存到 outputs/)：
  kmeans_metrics.png       4 指標 vs k
  kmeans_pca_scatter.png   PC1-PC2 散佈圖 (依群上色 + 群中心)
  kmeans_pca_labels.csv    每個樣本的群號 (可當回歸的類別特徵)
  kmeans_summary.csv       最終分組的量化指標
==============================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 設定區 ────────────────────────────────────────────
PCA_COMPONENTS_PATH = "outputs/pca_components.csv"  # PCA 輸出檔
USE_N_PCS    = None      # 只取前幾個 PC 分群；None = 全部使用
MAX_K        = 10        # 掃描的最大群數
N_CLUSTERS   = 3         # 最終分群數 (看指標圖後決定)
OUTPUT_DIR   = "outputs"
RANDOM_STATE = 42

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def make_demo_pca(n_samples=200, n_pcs=5, n_groups=3, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 6, size=(n_groups, n_pcs))
    rows, ids = [], []
    for i in range(n_samples):
        g = i % n_groups
        rows.append(centers[g] + rng.normal(0, 1.5, size=n_pcs))
        ids.append(f"S{i+1:03d}")
    cols = [f"PC{j+1}" for j in range(n_pcs)]
    return pd.DataFrame(rows, columns=cols, index=pd.Index(ids, name="sample_id"))


def load_components(path, use_n_pcs):
    if not os.path.exists(path):
        print(f"⚠ 找不到 '{path}' → 改用 demo 資料示範 (請先跑 pca_analysis.py)。")
        df = make_demo_pca()
    else:
        df = pd.read_csv(path, index_col=0)
        # 只留數值欄 (PC 欄)
        df = df.select_dtypes(include=[np.number])

    if use_n_pcs is not None:
        use_n_pcs = min(use_n_pcs, df.shape[1])
        df = df.iloc[:, :use_n_pcs]
        print(f"只使用前 {use_n_pcs} 個主成分進行分群。")
    print(f"分群輸入維度：{df.shape[0]} 樣本 × {df.shape[1]} 主成分")
    return df


def evaluate_k(X, out_dir):
    """掃描 k，計算 4 個量化指標並畫圖、給建議"""
    n_samples = X.shape[0]
    max_k = min(MAX_K, n_samples - 1)
    ks = list(range(2, max_k + 1))
    inertia, sil, ch, db = [], [], [], []

    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, labels))
        ch.append(calinski_harabasz_score(X, labels))
        db.append(davies_bouldin_score(X, labels))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(ks, inertia, marker="o", color="steelblue")
    axes[0, 0].set_title("Inertia (Elbow，找手肘轉折)")
    axes[0, 0].set_xlabel("群數 k"); axes[0, 0].set_ylabel("群內平方和")

    axes[0, 1].plot(ks, sil, marker="o", color="tomato")
    axes[0, 1].axvline(ks[int(np.argmax(sil))], ls="--", color="tomato", alpha=0.5)
    axes[0, 1].set_title("Silhouette 輪廓係數 (越高越好)")
    axes[0, 1].set_xlabel("群數 k"); axes[0, 1].set_ylabel("分數")

    axes[1, 0].plot(ks, ch, marker="o", color="seagreen")
    axes[1, 0].axvline(ks[int(np.argmax(ch))], ls="--", color="seagreen", alpha=0.5)
    axes[1, 0].set_title("Calinski-Harabasz (越高越好)")
    axes[1, 0].set_xlabel("群數 k"); axes[1, 0].set_ylabel("分數")

    axes[1, 1].plot(ks, db, marker="o", color="purple")
    axes[1, 1].axvline(ks[int(np.argmin(db))], ls="--", color="purple", alpha=0.5)
    axes[1, 1].set_title("Davies-Bouldin (越低越好)")
    axes[1, 1].set_xlabel("群數 k"); axes[1, 1].set_ylabel("分數")

    plt.suptitle("K-means 分組品質量化", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "kmeans_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"圖表已儲存：{path}")

    best_sil = ks[int(np.argmax(sil))]
    best_ch = ks[int(np.argmax(ch))]
    best_db = ks[int(np.argmin(db))]
    print("── 各指標建議的最佳群數 ──────────────────")
    print(f"Silhouette       建議 k = {best_sil}  (越高越好)")
    print(f"Calinski-Harabasz建議 k = {best_ch}  (越高越好)")
    print(f"Davies-Bouldin   建議 k = {best_db}  (越低越好)")
    print("──────────────────────────────────────────")

    # 把每個 k 的指標存成表 (量化依據)
    pd.DataFrame({"k": ks, "Inertia": inertia, "Silhouette": sil,
                  "Calinski_Harabasz": ch, "Davies_Bouldin": db}).to_csv(
        os.path.join(out_dir, "kmeans_metrics.csv"), index=False)
    print(f"各 k 指標已存：{out_dir}/kmeans_metrics.csv")
    return best_sil


def run_kmeans(X, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    return km, labels


def plot_scatter(X, labels, km, out_dir):
    """PC1-PC2 散佈圖，依群上色 + 群中心"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="tab10",
                    alpha=0.6, s=20)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               c="black", marker="X", s=200, label="群中心")
    ax.set_xlabel(X.columns[0]); ax.set_ylabel(X.columns[1])
    ax.set_title(f"K-means 分群結果 (k={km.n_clusters})")
    legend1 = ax.legend(*sc.legend_elements(), title="群", loc="upper right")
    ax.add_artist(legend1)
    ax.legend(loc="upper left")
    plt.tight_layout()
    path = os.path.join(out_dir, "kmeans_pca_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"圖表已儲存：{path}")


def save_results(X, labels, km, out_dir):
    out = X.copy()
    out["KMeans_Cluster"] = labels
    out.to_csv(os.path.join(out_dir, "kmeans_pca_labels.csv"))

    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    summary = {"k": km.n_clusters, "Inertia": round(km.inertia_, 4),
               "Silhouette": round(sil, 4),
               "Calinski_Harabasz": round(ch, 4),
               "Davies_Bouldin": round(db, 4)}
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "kmeans_summary.csv"), index=False)

    print(f"結果已儲存：{out_dir}/kmeans_pca_labels.csv、kmeans_summary.csv")
    print("── 最終分組量化指標 ──────────────────────")
    print(pd.DataFrame([summary]).T.rename(columns={0: "Value"}).to_string())
    print("各群樣本數：")
    print(pd.Series(labels).value_counts().sort_index().rename("Count").to_string())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X = load_components(PCA_COMPONENTS_PATH, USE_N_PCS)

    print("\n── 掃描不同群數 k，量化分組品質 ──────────")
    suggested = evaluate_k(X, OUTPUT_DIR)
    print(f"(Silhouette 自動建議 k={suggested}；目前 N_CLUSTERS={N_CLUSTERS})")

    print(f"\n以 N_CLUSTERS={N_CLUSTERS} 做最終分群...")
    km, labels = run_kmeans(X, N_CLUSTERS)
    if X.shape[1] >= 2:
        plot_scatter(X, labels, km, OUTPUT_DIR)
    save_results(X, labels, km, OUTPUT_DIR)
    print("\n✓ 完成。kmeans_pca_labels.csv 的 KMeans_Cluster 欄可當回歸的類別特徵。")


if __name__ == "__main__":
    main()
