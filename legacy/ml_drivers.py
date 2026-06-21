# -*- coding: utf-8 -*-
"""
ml_drivers.py — PFAS 驅動因子機器學習 (RF + XGBoost/SHAP/ALE + Hurdle + 時序CV)
================================================================
【做什麼】
  以環境/氣象/傳輸協變數解釋 PFAS 濃度，輸出：
    1. 時序交叉驗證的 RF / XGBoost 表現 (cv_metrics.csv)
    2. 特徵重要度：RF 置換重要度 + XGBoost gain + (可選)SHAP (feature_importance.csv, shap_*.png)
    3. ALE 累積局部效應曲線(前幾個重要特徵) (ale_*.png)
    4. Hurdle 兩段式(高 BDL target 用)：偵測模型 + 正值回歸 (hurdle_report.csv)
    5. (可選) GAM / 混合效應(站隨機效應) 若已安裝 pygam / statsmodels

【為何需要】(Paper2)
  ‧ 回答「哪些環境參數驅動 PFAS」；SHAP/ALE 給可解釋的非線性關係。
  ‧ 樣本少時：時序(blocked)CV 防過度樂觀；Hurdle 處理多零 target；RF 與 XGB 互核重要度。
  ‧ 注意：SHAP/ALE 是「關聯」非「因果」；需與軌跡來源、診斷比值交叉印證。

【底層邏輯】
  ‧ 時序 CV：依時間切折(不隨機)，避免用未來預測過去的洩漏。
  ‧ 置換重要度：打亂某特徵後表現下降幅度 = 該特徵貢獻。
  ‧ ALE：在特徵局部區間內微擾，平均模型預測變化並累積 → 不受共線性污染(優於 PDP)。
  ‧ Hurdle：P(檢出) 用分類器；檢出量值用回歸器；合成期望值。

【用法】
  1. DATA_PATH：CSV(sample_id, [date], [site], target, 各協變數...)。
     協變數示例：本地氣象 + ERA5 PBLH/風 + CAMS AOD + FIRMS 火 + 軌跡群 one-hot + 季節。
  2. 設 TARGET_COL(如 "ΣPFAS")、DATE_COL、(選)SITE_COL。
  3. python ml_drivers.py （無檔則 demo）
"""
import os, sys, io
import numpy as np
import pandas as pd

# ============================ 設定區 ============================
DATA_PATH   = "your_data.csv"
INDEX_COL   = "sample_id"
TARGET_COL  = "target"
DATE_COL    = "date"          # 有則用時序CV；None=KFold
SITE_COL    = "site"          # 有則嘗試混合效應(選)
N_SPLITS    = 5
SEED        = 42
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "10_ml")
# ===============================================================
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)


def load():
    if os.path.exists(DATA_PATH):
        print(f"[讀取] {DATA_PATH}")
        return pd.read_csv(DATA_PATH)
    print("[警告] 找不到資料 → demo (target 由 PBLH/火點/風速驅動，含零)")
    n = 120
    date = pd.date_range("2025-01-01", periods=n, freq="3D")
    pblh = rng.uniform(200, 1500, n)
    fire = rng.poisson(5, n).astype(float)
    ws = rng.uniform(0.5, 8, n)
    rh = rng.uniform(40, 95, n)
    target = np.clip(0.004 * fire * 2 + 0.0008 * (1500 - pblh) + 0.3 * ws
                     + rng.normal(0, 0.5, n), 0, None)
    target[rng.random(n) < 0.25] = 0     # 多零
    return pd.DataFrame({"sample_id": range(n), "date": date, "site": rng.choice(["A", "B"], n),
                         "target": target, "PBLH": pblh, "fire": fire, "WS": ws, "RH": rh})


def main():
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit, KFold
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
    try:
        from xgboost import XGBRegressor
        HAS_XGB = True
    except Exception:
        HAS_XGB = False

    df = load()
    if DATE_COL and DATE_COL in df.columns:
        df = df.sort_values(DATE_COL).reset_index(drop=True)
    drop = {INDEX_COL, TARGET_COL, DATE_COL, SITE_COL}
    feats = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].fillna(df[feats].median()).values
    y = df[TARGET_COL].values.astype(float)
    print(f"[資料] n={len(y)} 特徵={len(feats)} target零比例={np.mean(y==0):.0%}")

    # ---- 時序/KFold CV：RF 與 XGB ----
    splitter = TimeSeriesSplit(n_splits=N_SPLITS) if (DATE_COL in df.columns) else KFold(N_SPLITS, shuffle=True, random_state=SEED)
    models = {"RF": RandomForestRegressor(n_estimators=300, random_state=SEED)}
    if HAS_XGB:
        models["XGB"] = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, random_state=SEED, verbosity=0)
    cv_rows = []
    for name, mdl in models.items():
        r2s, rmses = [], []
        for tr, te in splitter.split(X):
            mdl.fit(X[tr], y[tr])
            pred = mdl.predict(X[te])
            r2s.append(r2_score(y[te], pred))
            rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
        cv_rows.append({"model": name, "cv_R2_mean": np.mean(r2s), "cv_R2_std": np.std(r2s),
                        "cv_RMSE_mean": np.mean(rmses)})
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(os.path.join(OUTPUT_DIR, "cv_metrics.csv"), index=False)
    print("\n[交叉驗證]\n", cv.round(3).to_string(index=False))

    # ---- 全資料配適 + 重要度 ----
    rf = RandomForestRegressor(n_estimators=400, random_state=SEED).fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=20, random_state=SEED)
    imp = pd.DataFrame({"feature": feats, "RF_perm_importance": perm.importances_mean})
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                           subsample=0.8, random_state=SEED, verbosity=0).fit(X, y)
        imp["XGB_gain"] = xgb.feature_importances_
    imp = imp.sort_values("RF_perm_importance", ascending=False)
    imp.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    print("\n[特徵重要度]\n", imp.round(4).to_string(index=False))

    # ---- SHAP (可選) ----
    try:
        import shap
        expl_model = xgb if HAS_XGB else rf
        explainer = shap.TreeExplainer(expl_model)
        sv = explainer.shap_values(X)
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]; plt.rcParams["axes.unicode_minus"] = False
        shap.summary_plot(sv, X, feature_names=feats, show=False)
        plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=150); plt.close()
        print("[SHAP] shap_summary.png 已存。")
    except Exception as e:
        print("[SHAP 略過]", repr(e)[:120])

    # ---- ALE (前3重要特徵) ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = [feats.index(f) for f in imp["feature"].head(3)]
        fig, axes = plt.subplots(1, len(top), figsize=(4.5 * len(top), 3.8))
        for ax, fi in zip(np.atleast_1d(axes), top):
            q = np.unique(np.quantile(X[:, fi], np.linspace(0, 1, 11)))
            eff = np.zeros(len(q) - 1)
            for i in range(len(q) - 1):
                m = (X[:, fi] >= q[i]) & (X[:, fi] <= q[i + 1])
                if m.sum():
                    lo, hi = X[m].copy(), X[m].copy()
                    lo[:, fi], hi[:, fi] = q[i], q[i + 1]
                    eff[i] = np.mean(rf.predict(hi) - rf.predict(lo))
            ale = np.cumsum(eff); ale -= ale.mean()
            ax.plot((q[:-1] + q[1:]) / 2, ale, "-o", ms=3)
            ax.set_title(f"ALE: {feats[fi]}"); ax.axhline(0, ls=":", c="grey")
        fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "ale_top.png"), dpi=150)
        print("[ALE] ale_top.png 已存。")
    except Exception as e:
        print("[ALE 略過]", repr(e)[:120])

    # ---- Hurdle (高零 target) ----
    if np.mean(y == 0) > 0.15:
        detect = (y > 0).astype(int)
        clf = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(X, detect)
        try:
            auc = roc_auc_score(detect, clf.predict_proba(X)[:, 1])
        except Exception:
            auc = np.nan
        pos = y > 0
        reg = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(X[pos], y[pos])
        r2_pos = r2_score(y[pos], reg.predict(X[pos]))
        pd.DataFrame([{"detect_AUC_insample": auc, "positive_part_R2_insample": r2_pos,
                       "zero_fraction": float(np.mean(y == 0))}]).to_csv(
            os.path.join(OUTPUT_DIR, "hurdle_report.csv"), index=False)
        print(f"\n[Hurdle] 偵測AUC={auc:.3f}  正值部R²={r2_pos:.3f} (in-sample；正式請用CV)")

    # ---- 可選進階模型 ----
    try:
        from pygam import LinearGAM
        gam = LinearGAM().fit(X, y)
        print(f"[GAM] pseudo-R²={gam.statistics_['pseudo_r2']['explained_deviance']:.3f}")
    except Exception:
        print("[GAM 略過] 未安裝 pygam (pip install pygam 可啟用 GAM 推論主幹)")
    if SITE_COL in df.columns:
        try:
            import statsmodels.formula.api as smf
            d2 = df[[TARGET_COL, SITE_COL] + feats].dropna()
            md = smf.mixedlm(f"{TARGET_COL} ~ " + " + ".join(feats), d2, groups=d2[SITE_COL]).fit()
            print(f"[混合效應] 站隨機效應已配適 (AIC={md.aic:.1f})")
        except Exception:
            print("[混合效應 略過] 未安裝 statsmodels (pip install statsmodels 可啟用)")

    print(f"\n✓ 完成，輸出於 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
