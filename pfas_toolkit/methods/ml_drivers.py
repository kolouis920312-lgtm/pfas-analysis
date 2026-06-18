# -*- coding: utf-8 -*-
"""
ml_drivers.py — PFAS 驅動因子機器學習（由 ml_drivers.py 整合而來）
時序/KFold 交叉驗證 (RF + XGBoost) + 置換重要度/gain + SHAP + ALE + Hurdle。
輸出：cv_metrics / feature_importance / hurdle_report（CSV）+ shap_summary / ale_top（圖）
"""
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.theme import get_plt


def make_demo(seed=42):
    rng = np.random.default_rng(seed)
    n = 120
    date = pd.date_range("2025-01-01", periods=n, freq="3D")
    pblh = rng.uniform(200, 1500, n)
    fire = rng.poisson(5, n).astype(float)
    ws = rng.uniform(0.5, 8, n)
    rh = rng.uniform(40, 95, n)
    target = np.clip(0.004 * fire * 2 + 0.0008 * (1500 - pblh) + 0.3 * ws
                     + rng.normal(0, 0.5, n), 0, None)
    target[rng.random(n) < 0.25] = 0
    return pd.DataFrame({"sample_id": range(n), "date": date,
                         "site": rng.choice(["A", "B"], n), "target": target,
                         "PBLH": pblh, "fire": fire, "WS": ws, "RH": rh})


def run(df, params, ctx):
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit, KFold
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
    plt = get_plt(ctx.theme)
    try:
        from xgboost import XGBRegressor
        HAS_XGB = True
    except Exception:
        HAS_XGB = False

    seed = 42
    target = params.get("target_col")
    if not target or target not in df.columns:
        raise ValueError(f"請指定存在的目標欄。現有欄位：{list(df.columns)}")
    id_col = params.get("id_col") or None
    date_col = params.get("date_col") or None
    site_col = params.get("site_col") or None
    if date_col in ("(無)", ""):
        date_col = None
    if site_col in ("(無)", ""):
        site_col = None
    n_splits = int(params.get("n_splits", 5))
    primary = ctx.color("primary", "#4682b4")

    data = df.copy()
    use_ts = bool(date_col and date_col in data.columns)
    if use_ts:
        data = data.sort_values(date_col).reset_index(drop=True)
    drop = {id_col, target, date_col, site_col}
    feats = [c for c in data.columns if c not in drop and pd.api.types.is_numeric_dtype(data[c])]
    if len(feats) < 1:
        raise ValueError("沒有可用的數值協變數。")
    Xdf = data[feats].fillna(data[feats].median())
    y = pd.to_numeric(data[target], errors="coerce").values.astype(float)
    ymask = ~np.isnan(y)
    X = Xdf.values[ymask]
    y = y[ymask]
    ctx.log(f"n={len(y)} 特徵={len(feats)} target零比例={np.mean(y == 0):.0%}；CV={'時序' if use_ts else 'KFold'}")

    splitter = TimeSeriesSplit(n_splits=n_splits) if use_ts else KFold(n_splits, shuffle=True, random_state=seed)
    models = {"RF": RandomForestRegressor(n_estimators=300, random_state=seed)}
    if HAS_XGB:
        models["XGB"] = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, random_state=seed, verbosity=0)
    cv_rows = []
    for name, mdl in models.items():
        r2s, rmses = [], []
        for tr, te in splitter.split(X):
            mdl.fit(X[tr], y[tr])
            pred = mdl.predict(X[te])
            r2s.append(r2_score(y[te], pred))
            rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
        cv_rows.append({"model": name, "cv_R2_mean": np.mean(r2s),
                        "cv_R2_std": np.std(r2s), "cv_RMSE_mean": np.mean(rmses)})
    ctx.save_table(pd.DataFrame(cv_rows), "cv_metrics", index=False)
    ctx.log("交叉驗證：" + "; ".join(f"{r['model']} R²={r['cv_R2_mean']:.3f}±{r['cv_R2_std']:.3f}" for r in cv_rows))

    rf = RandomForestRegressor(n_estimators=400, random_state=seed).fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=20, random_state=seed)
    imp = pd.DataFrame({"feature": feats, "RF_perm_importance": perm.importances_mean})
    xgb = None
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                           subsample=0.8, random_state=seed, verbosity=0).fit(X, y)
        imp["XGB_gain"] = xgb.feature_importances_
    imp = imp.sort_values("RF_perm_importance", ascending=False)
    ctx.save_table(imp, "feature_importance", index=False)
    ctx.log("特徵重要度 Top：" + ", ".join(imp["feature"].head(5).astype(str)))

    # SHAP
    try:
        import shap
        expl_model = xgb if (HAS_XGB and xgb is not None) else rf
        explainer = shap.TreeExplainer(expl_model)
        sv = explainer.shap_values(X)
        shap.summary_plot(sv, X, feature_names=feats, show=False)
        fig = plt.gcf(); fig.tight_layout(); ctx.save_fig(fig, "shap_summary")
        ctx.log("SHAP 摘要圖已存。")
    except Exception as e:
        ctx.log(f"SHAP 略過：{repr(e)[:100]}")

    # ALE（前 3 重要特徵）
    try:
        top = [feats.index(f) for f in imp["feature"].head(3)]
        fig, axes = plt.subplots(1, len(top), figsize=(4.5 * len(top), 3.8))
        for ax, fi in zip(np.atleast_1d(axes).ravel(), top):
            q = np.unique(np.quantile(X[:, fi], np.linspace(0, 1, 11)))
            eff = np.zeros(max(len(q) - 1, 1))
            for i in range(len(q) - 1):
                m = (X[:, fi] >= q[i]) & (X[:, fi] <= q[i + 1])
                if m.sum():
                    lo, hi = X[m].copy(), X[m].copy()
                    lo[:, fi], hi[:, fi] = q[i], q[i + 1]
                    eff[i] = np.mean(rf.predict(hi) - rf.predict(lo))
            ale = np.cumsum(eff); ale -= ale.mean()
            ax.plot((q[:-1] + q[1:]) / 2, ale[:len(q) - 1], "-o", ms=3, color=primary)
            ax.set_title(f"ALE: {feats[fi]}"); ax.axhline(0, ls=":", c="grey")
        fig.tight_layout(); ctx.save_fig(fig, "ale_top")
        ctx.log("ALE 曲線（前 3 特徵）已存。")
    except Exception as e:
        ctx.log(f"ALE 略過：{repr(e)[:100]}")

    # Hurdle（高零 target）
    if np.mean(y == 0) > 0.15:
        detect = (y > 0).astype(int)
        clf = RandomForestClassifier(n_estimators=300, random_state=seed).fit(X, detect)
        try:
            auc = roc_auc_score(detect, clf.predict_proba(X)[:, 1])
        except Exception:
            auc = np.nan
        pos = y > 0
        reg = RandomForestRegressor(n_estimators=300, random_state=seed).fit(X[pos], y[pos])
        r2_pos = r2_score(y[pos], reg.predict(X[pos]))
        ctx.save_table(pd.DataFrame([{"detect_AUC_insample": auc,
                                      "positive_part_R2_insample": r2_pos,
                                      "zero_fraction": float(np.mean(y == 0))}]),
                       "hurdle_report", index=False)
        ctx.log(f"Hurdle：偵測 AUC={auc:.3f} 正值部 R²={r2_pos:.3f}（in-sample，正式請用 CV）")

    return ctx.result(summary="ML 驅動因子完成：CV、置換+gain 重要度、SHAP、ALE"
                              + ("、Hurdle" if np.mean(y == 0) > 0.15 else "") + "。")


SPEC = MethodSpec(
    key="ml_drivers",
    name="ML 驅動因子分析",
    summary="以環境/氣象協變數解釋 PFAS：交叉驗證、可解釋重要度(SHAP/ALE)、多零 target 的 Hurdle。",
    params=[
        ParamSpec("id_col", "ID 欄（可空）", "column", default="sample_id", optional=True),
        ParamSpec("target_col", "目標欄 y（必填）", "column", default="target"),
        ParamSpec("date_col", "日期欄（有則用時序 CV，可空）", "column", default="date", optional=True),
        ParamSpec("site_col", "站別欄（排除於特徵，可空）", "column", default="site", optional=True),
        ParamSpec("n_splits", "交叉驗證折數", "int", default=5, minimum=2),
    ],
    schema=InputSchema(min_rows=10, min_numeric_cols=1, id_col_param="id_col",
                       required_param_cols=["target_col"]),
    template_columns=["sample_id", "date", "site", "target", "PBLH", "fire", "WS", "…"],
    uses_colors=["primary"],
)
SPEC.run = run
SPEC.make_demo = make_demo
