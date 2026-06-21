# -*- coding: utf-8 -*-
"""
XGBoost 回歸
==============================================================
用途：用 XGBoost 做回歸預測，輸入特徵可以是
  - PCA 降維特徵     outputs/pca_components.csv
  - NMF 樣本分解特徵  outputs/nmf_W.csv
  - 原始資料特徵
  並可選擇把分群結果 (K-means / HCA 的群號) 當成額外類別特徵加入。

輸入設定 (見下方設定區)：
  FEATURE_PATH   特徵 CSV (第一欄為樣本 ID)
  TARGET_PATH    目標 y 所在的 CSV (可與特徵同檔或不同檔)
  TARGET_COL     目標欄名稱
  CLUSTER_PATH   (可選) 分群結果 CSV，會把群號欄當類別特徵 one-hot 加入
  * 各檔以「樣本 ID (索引)」對齊合併
  找不到檔案 → 自動產生 demo 回歸資料示範。

執行：
  python xgboost_regression.py

輸出 (存到 outputs/)：
  xgb_model.json           訓練好的模型
  xgb_predictions.csv      測試集真實值 vs 預測值
  xgb_feature_importance.png / xgb_shap_summary.png (SHAP 有裝才畫)
  xgb_metrics.csv          RMSE / MAE / R²
==============================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 設定區 ────────────────────────────────────────────
FEATURE_PATH = "outputs/pca_components.csv"  # 特徵來源 (可換 nmf_W.csv 或原始資料)
TARGET_PATH  = "your_data.csv"               # 目標 y 來源
TARGET_COL   = "target"                      # 目標欄名稱
CLUSTER_PATH = None      # 例如 "outputs/kmeans_pca_labels.csv"；None = 不加群特徵
CLUSTER_COL  = "KMeans_Cluster"              # 群號欄名 (HCA 用 "HCA_Cluster")

TEST_SIZE    = 0.2
OUTPUT_DIR   = "outputs"
RANDOM_STATE = 42
XGB_PARAMS   = dict(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.0, reg_lambda=1.0,
    random_state=RANDOM_STATE, n_jobs=-1,
)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def make_demo_data(n_samples=300, n_features=6, seed=RANDOM_STATE):
    """產生有真實關聯的 demo 回歸資料 (X 與 y)"""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    coef = rng.uniform(-3, 3, size=n_features)
    # 非線性 + 交互作用 + 雜訊，讓 XGBoost 有發揮空間
    y = (X @ coef) + 2.0 * X[:, 0] * X[:, 1] + np.sin(X[:, 2]) * 3 \
        + rng.normal(0, 0.5, size=n_samples)
    ids = [f"S{i+1:03d}" for i in range(n_samples)]
    cols = [f"feature_{j+1}" for j in range(n_features)]
    Xdf = pd.DataFrame(X, columns=cols, index=pd.Index(ids, name="sample_id"))
    ydf = pd.Series(y, index=Xdf.index, name=TARGET_COL)
    return Xdf, ydf


def load_features_and_target():
    """讀特徵與目標，依索引對齊；缺檔則用 demo 資料"""
    if not os.path.exists(FEATURE_PATH) or not os.path.exists(TARGET_PATH):
        missing = [p for p in (FEATURE_PATH, TARGET_PATH) if not os.path.exists(p)]
        print(f"⚠ 找不到 {missing} → 改用 demo 回歸資料示範。")
        return make_demo_data()

    X = pd.read_csv(FEATURE_PATH, index_col=0).select_dtypes(include=[np.number])

    tdf = pd.read_csv(TARGET_PATH, index_col=0)
    if TARGET_COL not in tdf.columns:
        raise ValueError(f"目標檔 '{TARGET_PATH}' 沒有欄位 '{TARGET_COL}'。"
                         f" 現有欄位：{list(tdf.columns)}")
    y = tdf[TARGET_COL]

    # 依索引對齊 (只保留兩邊都有的樣本)
    common = X.index.intersection(y.index)
    if len(common) == 0:
        raise ValueError("特徵與目標的樣本索引完全對不上，請確認兩檔第一欄是相同的 ID。")
    X, y = X.loc[common], y.loc[common]
    print(f"特徵與目標對齊後：{len(common)} 個樣本")
    return X, y


def add_cluster_feature(X):
    """(可選) 把分群群號當類別特徵 one-hot 併入"""
    if CLUSTER_PATH is None:
        return X
    if not os.path.exists(CLUSTER_PATH):
        print(f"⚠ 找不到分群檔 '{CLUSTER_PATH}'，略過群特徵。")
        return X
    cdf = pd.read_csv(CLUSTER_PATH, index_col=0)
    if CLUSTER_COL not in cdf.columns:
        print(f"⚠ 分群檔沒有欄位 '{CLUSTER_COL}'，略過群特徵。")
        return X
    cluster = cdf[CLUSTER_COL].reindex(X.index)
    dummies = pd.get_dummies(cluster, prefix="cluster")
    X = X.join(dummies)
    print(f"已加入分群類別特徵：{list(dummies.columns)}")
    return X


def describe(X, y):
    print("── 資料摘要 ──────────────────────────────")
    print(f"樣本數      : {X.shape[0]}")
    print(f"特徵數      : {X.shape[1]}")
    print(f"特徵欄位    : {list(X.columns)}")
    print(f"目標 y 範圍 : [{y.min():.3f}, {y.max():.3f}]  平均={y.mean():.3f}")
    print("──────────────────────────────────────────")


def train_and_evaluate(X, y, out_dir):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print("── 測試集評估 ────────────────────────────")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")
    print("──────────────────────────────────────────")

    pd.DataFrame([{"RMSE": rmse, "MAE": mae, "R2": r2}]).to_csv(
        os.path.join(out_dir, "xgb_metrics.csv"), index=False)
    pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred},
                 index=y_test.index).to_csv(
        os.path.join(out_dir, "xgb_predictions.csv"))

    model.save_model(os.path.join(out_dir, "xgb_model.json"))
    print(f"模型已儲存：{out_dir}/xgb_model.json")
    return model, X_train, X_test, y_test, y_pred


def plot_pred_vs_true(y_test, y_pred, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, color="steelblue")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="red", label="理想 (y=x)")
    ax.set_xlabel("真實值"); ax.set_ylabel("預測值")
    ax.set_title("預測 vs 真實")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "xgb_pred_vs_true.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"圖表已儲存：{path}")


def plot_importance(model, feature_names, out_dir):
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(feature_names))))
    ax.barh([feature_names[i] for i in order][::-1],
            imp[order][::-1], color="steelblue")
    ax.set_xlabel("重要度 (gain)")
    ax.set_title("XGBoost 特徵重要度")
    plt.tight_layout()
    path = os.path.join(out_dir, "xgb_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"圖表已儲存：{path}")

    # 印出排序表
    print("── 特徵重要度排序 ────────────────────────")
    for i in order:
        print(f"  {feature_names[i]:<20} {imp[i]:.4f}")
    print("──────────────────────────────────────────")


def plot_shap(model, X_train, out_dir):
    """SHAP 摘要圖 (有裝 shap 才畫，否則略過)"""
    try:
        import shap
    except ImportError:
        print("ℹ 未安裝 shap，略過 SHAP 分析 (pip install shap 可啟用)。")
        return
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        plt.figure()
        shap.summary_plot(shap_values, X_train, show=False)
        path = os.path.join(out_dir, "xgb_shap_summary.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"圖表已儲存：{path}")
    except Exception as e:
        print(f"ℹ SHAP 計算發生問題，已略過：{e}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, y = load_features_and_target()
    X = add_cluster_feature(X)
    describe(X, y)

    model, X_train, X_test, y_test, y_pred = train_and_evaluate(X, y, OUTPUT_DIR)
    plot_pred_vs_true(y_test, y_pred, OUTPUT_DIR)
    plot_importance(model, list(X.columns), OUTPUT_DIR)
    plot_shap(model, X_train, OUTPUT_DIR)
    print("\n✓ XGBoost 回歸完成。預測結果見 outputs/xgb_predictions.csv。")


if __name__ == "__main__":
    main()
