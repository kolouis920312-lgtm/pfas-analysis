# -*- coding: utf-8 -*-
"""
bdl.py — 偵測極限(BDL/censored)處理（由 bdl_censored.py 整合而來）
輸出：detection_frequency / subst_half / subst_sqrt2 / ros_imputed /
      censored_summary / uncertainty_matrix（CSV）+ 偵測熱圖
"""
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.theme import get_plt
from ..core.prep import numeric_frame


def make_demo(seed=42):
    rng = np.random.default_rng(seed)
    n, species = 40, ["PFPeA", "PFHxA", "PFOA", "PFNA", "PFOS", "PFHxS", "6:2 FTS", "FOSA"]
    data = {}
    for j, s in enumerate(species):
        vals = rng.lognormal(mean=0.5 - 0.3 * j, sigma=1.0, size=n)
        thr = np.quantile(vals, 0.3 + 0.07 * j)
        vals[vals < thr] = 0.0
        data[s] = vals
    df = pd.DataFrame(data)
    df.insert(0, "sample_id", [f"S{i:03d}" for i in range(n)])
    return df


def ros_impute_col(values, mdl, norm, linregress):
    v = values.astype(float).copy()
    cens = v <= 0
    c, n = int(cens.sum()), len(v)
    if c == 0:
        return v
    det = np.sort(v[~cens])
    if len(det) < 3:
        v[cens] = mdl / np.sqrt(2)
        return v
    pe = c / n
    pos = pe + (1 - pe) * (np.arange(1, len(det) + 1) - 0.5) / len(det)
    z = norm.ppf(pos)
    slope, intercept, *_ = linregress(z, np.log(det))
    pos_c = (np.arange(1, c + 1) - 0.5) / n
    pred = np.sort(np.exp(intercept + slope * norm.ppf(pos_c)))
    pred = np.clip(pred, 1e-12, mdl)
    v[np.where(cens)[0]] = pred
    return v


def tobit_lognormal(values, mdl, norm, minimize):
    v = values.astype(float)
    cens = v <= 0
    det = v[~cens]
    if len(det) < 3:
        return np.nan, np.nan, np.nan
    ld = np.log(det)
    lt = np.log(mdl)
    ncens = int(cens.sum())

    def negll(p):
        mu, logs = p
        s = np.exp(logs)
        return -(norm.logpdf(ld, mu, s).sum() + float(norm.logcdf((lt - mu) / s)) * ncens)
    res = minimize(negll, [ld.mean(), np.log(ld.std() + 1e-6)], method="Nelder-Mead")
    mu, s = res.x[0], np.exp(res.x[1])
    return float(np.exp(mu + s ** 2 / 2)), float(mu), float(s)


def run(df, params, ctx):
    from scipy.stats import norm, linregress
    from scipy.optimize import minimize
    plt = get_plt(ctx.theme)

    X = numeric_frame(df, ctx, id_col=params.get("id_col"))
    if X.shape[1] < 1:
        raise ValueError("沒有可用的數值欄。")
    error_frac = float(params.get("error_frac", 0.10))
    mdl_override = float(params.get("mdl_override", 0) or 0)
    cmap_seq = ctx.color("cmap_sequential", "Greens")

    mdl = {}
    for c in X.columns:
        pos = X[c][X[c] > 0]
        mdl[c] = pos.min() if len(pos) else 1.0
    mdl = pd.Series(mdl)
    if mdl_override > 0:
        mdl = pd.Series(mdl_override, index=X.columns)
        ctx.log(f"使用固定 MDL = {mdl_override:g}（套用到所有物種）。")
    else:
        ctx.log("MDL 以各物種最小正值估計（正式分析請提供真實 MDL）。")

    det = pd.concat([(X > 0).sum().rename("n_detected"),
                     (X > 0).mean().rename("detection_freq")], axis=1)
    ctx.save_table(det, "detection_frequency")

    for name, factor in [("subst_half", 0.5), ("subst_sqrt2", 1 / np.sqrt(2))]:
        sub = X.copy()
        for c in X.columns:
            sub.loc[sub[c] <= 0, c] = mdl[c] * factor
        ctx.save_table(sub, name)

    ros = X.copy().astype(float)
    for c in X.columns:
        ros[c] = ros_impute_col(X[c].values, mdl[c], norm, linregress)
    ctx.save_table(ros, "ros_imputed")

    rows = []
    for c in X.columns:
        tmean, mu, sg = tobit_lognormal(X[c].values, mdl[c], norm, minimize)
        rows.append({"species": c, "detection_freq": det.loc[c, "detection_freq"],
                     "mean_subst_half": X[c].where(X[c] > 0, mdl[c] * 0.5).mean(),
                     "mean_ros": ros[c].mean(), "mean_tobit_lognorm": tmean})
    ctx.save_table(pd.DataFrame(rows).set_index("species"), "censored_summary")
    ctx.log("censored 摘要（各法平均）→ censored_summary.csv")

    unc = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for c in X.columns:
        m = mdl[c]; x = X[c].values.astype(float)
        unc[c] = np.where(x <= 0, (5.0 / 6.0) * m,
                          np.sqrt((error_frac * x) ** 2 + (0.5 * m) ** 2))
    ctx.save_table(unc, "uncertainty_matrix")

    fig, ax = plt.subplots(figsize=(min(14, 1 + 0.5 * X.shape[1]), min(10, 1 + 0.18 * len(X))))
    ax.imshow((X.values > 0).astype(int), aspect="auto", cmap=cmap_seq, vmin=0, vmax=1)
    ax.set_xticks(range(X.shape[1])); ax.set_xticklabels(X.columns, rotation=90, fontsize=7)
    ax.set_title("偵測 / 未檢出  Detection map"); ax.set_ylabel("樣本")
    fig.tight_layout(); ctx.save_fig(fig, "detection_heatmap")

    return ctx.result(summary="BDL 處理完成：偵測頻率、替代法、ROS、Tobit、不確定度矩陣。"
                              "uncertainty_matrix.csv 供 PMF；ros_imputed.csv 供下游。")


SPEC = MethodSpec(
    key="bdl",
    name="BDL 偵測極限處理",
    summary="高 BDL 資料的統計正確處理：偵測頻率、替代法、ROS 填補、Tobit MLE、PMF 不確定度矩陣。",
    params=[
        ParamSpec("id_col", "ID 欄（可空）", "column", default="sample_id", optional=True),
        ParamSpec("error_frac", "相對分析誤差（不確定度用）", "float", default=0.10,
                  minimum=0.0, maximum=1.0),
        ParamSpec("mdl_override", "固定 MDL（0=自動用各物種最小正值）", "float", default=0.0,
                  minimum=0.0,
                  help="填 > 0 則所有物種改用這個方法偵測極限；正式分析建議填實驗室真實 MDL。"),
    ],
    schema=InputSchema(min_rows=3, min_numeric_cols=1, id_col_param="id_col", check_bdl=True,
                       note="BDL 請以 0 或空白表示；勿用 ND/<MDL 文字。"),
    template_columns=["sample_id", "PFPeA", "PFHxA", "PFOA", "…"],
    uses_colors=["cmap_sequential"],
)
SPEC.run = run
SPEC.make_demo = make_demo
