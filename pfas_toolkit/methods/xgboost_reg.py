# -*- coding: utf-8 -*-
"""
xgboost_reg.py — XGBoost 回歸（由 xgboost_regression.py 整合而來）
輸出：xgb_metrics.csv / xgb_predictions.csv / xgb_model.json /
      預測vs真實圖 / 特徵重要度圖 / (有裝 shap 則) SHAP 摘要圖
※ 已移除原本對新版 XGBoost 較敏感的 fit(eval_set, verbose) 寫法。
"""
import os
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.theme import get_plt


def make_demo(n_samples=300, n_features=6, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    coef = rng.uniform(-3, 3, size=n_features)
    y = (X @ coef) + 2.0 * X[:, 0] * X[:, 1] + np.sin(X[:, 2]) * 3 + rng.normal(0, 0.5, n_samples)
    df = pd.DataFrame(X, columns=[f"feature_{j+1}" for j in range(n_features)])
    df.insert(0, "sample_id", [f"S{i+1:03d}" for i in range(n_samples)])
    df["target"] = y
    return df


def run(df, params, ctx):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from xgboost import XGBRegressor
    plt = get_plt(ctx.theme)

    target = params.get("target_col")
    if not target or target not in df.columns:
        raise ValueError(f"請指定存在的目標欄。現有欄位：{list(df.columns)}")
    work = df.copy()
    id_col = params.get("id_col")
    if id_col and id_col in work.columns:
        work = work.set_index(id_col)

    y = pd.to_numeric(work[target], errors="coerce")
    X = work.drop(columns=[target]).select_dtypes(include=[np.number])
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    if X.shape[1] < 1:
        raise ValueError("沒有可用的數值特徵欄。")
    nan = int(X.isna().sum().sum())
    if nan:
        ctx.log(f"⚠ 特徵有 {nan} 個缺值 → 中位數補值")
        X = X.fillna(X.median(numeric_only=True))
    ctx.log(f"樣本 {X.shape[0]}；特徵 {X.shape[1]}；目標 {target}")

    primary = ctx.color("primary", "#4682b4")
    accent = ctx.color("accent", "#ff6347")
    xgb_params = dict(
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=int(params.get("max_depth", 5)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        reg_lambda=1.0, random_state=42, n_jobs=-1,
    )
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=float(params.get("test_size", 0.2)), random_state=42)
    model = XGBRegressor(**xgb_params)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    mae = float(mean_absolute_error(yte, pred))
    r2 = float(r2_score(yte, pred))
    ctx.log(f"測試集 RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    ctx.save_table(pd.DataFrame([{"RMSE": rmse, "MAE": mae, "R2": r2}]),
                   "xgb_metrics", index=False)
    ctx.save_table(pd.DataFrame({"y_true": yte.values, "y_pred": pred}, index=yte.index),
                   "xgb_predictions")
    mpath = os.path.join(ctx.out.output_dir, "xgb_model.json")
    model.save_model(mpath); ctx.add_extra(mpath); ctx.log(f"模型已存：{mpath}")

    # 預測 vs 真實
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yte, pred, alpha=0.5, color=primary)
    lims = [min(float(yte.min()), float(pred.min())), max(float(yte.max()), float(pred.max()))]
    ax.plot(lims, lims, "--", color=accent, label="理想 y=x")
    ax.set_xlabel("真實值"); ax.set_ylabel("預測值"); ax.set_title("預測 vs 真實"); ax.legend()
    fig.tight_layout(); ctx.save_fig(fig, "xgb_pred_vs_true")

    # 特徵重要度
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    names = list(X.columns)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(names))))
    ax.barh([names[i] for i in order][::-1], imp[order][::-1], color=primary)
    ax.set_xlabel("重要度 (gain)"); ax.set_title("XGBoost 特徵重要度")
    fig.tight_layout(); ctx.save_fig(fig, "xgb_feature_importance")
    ctx.log("特徵重要度 Top：" + ", ".join(f"{names[i]}={imp[i]:.3f}" for i in order[:8]))

    # SHAP（可選）
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xtr)
        plt.figure()
        shap.summary_plot(sv, Xtr, show=False)
        fig = plt.gcf(); fig.tight_layout(); ctx.save_fig(fig, "xgb_shap_summary")
        ctx.log("SHAP 摘要圖已存。")
    except Exception as e:
        ctx.log(f"SHAP 略過：{repr(e)[:100]}")

    return ctx.result(summary=f"XGBoost 回歸完成：測試集 R²={r2:.3f}, RMSE={rmse:.3f}。")


SPEC = MethodSpec(
    key="xgboost_reg",
    name="XGBoost 回歸",
    summary="用 XGBoost 做回歸；輸出評估、預測、特徵重要度與 SHAP（可選）。",
    params=[
        ParamSpec("id_col", "ID 欄（可空）", "column", default="sample_id", optional=True),
        ParamSpec("target_col", "目標欄 y（必填）", "column", default="target"),
        ParamSpec("test_size", "測試集比例", "float", default=0.2, minimum=0.05, maximum=0.5),
        ParamSpec("n_estimators", "樹的數量", "int", default=300, minimum=10),
        ParamSpec("max_depth", "樹深度", "int", default=5, minimum=1),
        ParamSpec("learning_rate", "學習率", "float", default=0.05, minimum=0.001, maximum=1.0),
        ParamSpec("subsample", "樣本抽樣比例", "float", default=0.8, minimum=0.1, maximum=1.0),
        ParamSpec("colsample_bytree", "特徵抽樣比例", "float", default=0.8, minimum=0.1, maximum=1.0),
    ],
    schema=InputSchema(min_rows=10, min_numeric_cols=1, id_col_param="id_col",
                       required_param_cols=["target_col"]),
    template_columns=["sample_id", "feature_1", "feature_2", "…", "target"],
    uses_colors=["primary", "accent"],
)
SPEC.run = run
SPEC.make_demo = make_demo
