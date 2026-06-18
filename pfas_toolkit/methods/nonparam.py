# -*- coding: utf-8 -*-
"""
nonparam.py — 無母數統計（由 nonparam_stats.py 整合而來）
群差異(Mann-Whitney/Kruskal-Wallis) + 趨勢(Mann-Kendall/Sen) + Spearman 相關，
全部含 BH-FDR 多重比較校正。輸出對應 CSV 與相關熱圖。
"""
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.theme import get_plt


def make_demo(seed=3):
    rng = np.random.default_rng(seed)
    n = 36
    season = np.array(["冬"] * 12 + ["春"] * 12 + ["夏"] * 12)
    date = pd.date_range("2025-01-01", periods=n, freq="10D")
    sp = ["PFOA", "PFNA", "PFOS", "PFHxA", "PFHxS"]
    data = {"sample_id": [f"S{i:02d}" for i in range(n)], "season": season, "date": date}
    for j, s in enumerate(sp):
        data[s] = np.clip(rng.lognormal(0.2, 0.7, n) + np.linspace(0, 1.2 * j, n), 0, None)
    return pd.DataFrame(data)


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def mann_kendall(x, norm):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return dict(n=n, S=np.nan, Z=np.nan, p=np.nan, trend="n/a")
    S = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, counts = np.unique(x, return_counts=True)
    tie = sum(c * (c - 1) * (2 * c + 5) for c in counts)
    var = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    if var <= 0:
        return dict(n=n, S=int(S), Z=np.nan, p=np.nan, trend="n/a")
    Z = (S - 1) / np.sqrt(var) if S > 0 else (S + 1) / np.sqrt(var) if S < 0 else 0.0
    p = 2 * (1 - norm.cdf(abs(Z)))
    trend = "上升" if (p < 0.05 and S > 0) else "下降" if (p < 0.05 and S < 0) else "無顯著趨勢"
    return dict(n=n, S=int(S), Z=round(float(Z), 3), p=round(float(p), 4), trend=trend)


def sen_slope(x, t):
    x = np.asarray(x, float); t = np.asarray(t, float)
    sl = [(x[j] - x[i]) / (t[j] - t[i])
          for i in range(len(x) - 1) for j in range(i + 1, len(x))
          if t[j] != t[i] and not np.isnan(x[i]) and not np.isnan(x[j])]
    return float(np.median(sl)) if sl else np.nan


def run(df, params, ctx):
    from scipy.stats import mannwhitneyu, kruskal, spearmanr, norm
    plt = get_plt(ctx.theme)

    id_col = params.get("id_col") or None
    group_col = params.get("group_col") or None
    time_col = params.get("time_col") or None
    if group_col in ("(無)", ""):
        group_col = None
    if time_col in ("(無)", ""):
        time_col = None
    drop = {id_col, group_col, time_col}
    species = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    if len(species) < 1:
        raise ValueError("找不到數值變數欄。")
    ctx.log(f"數值變數 {len(species)} 欄；group={group_col}；time={time_col}")
    cmap_div = ctx.color("cmap_diverging", "RdBu_r")
    did = []

    # 1. 群差異
    if group_col and group_col in df.columns:
        groups = [g for _, g in df.groupby(group_col)]
        labels = [k for k, _ in df.groupby(group_col)]
        rows = []
        for s in species:
            arrs = [g[s].dropna().values for g in groups]
            arrs = [a for a in arrs if len(a) > 0]
            if len(arrs) < 2 or np.unique(np.concatenate(arrs)).size < 2:
                continue
            if len(arrs) == 2:
                U, p = mannwhitneyu(arrs[0], arrs[1], alternative="two-sided")
                eff = 1 - 2 * U / (len(arrs[0]) * len(arrs[1]))
                rows.append(dict(variable=s, test="Mann-Whitney", stat=round(float(U), 2),
                                 p=p, effect=round(float(eff), 3)))
            else:
                H, p = kruskal(*arrs)
                ntot = sum(len(a) for a in arrs)
                eta2 = (H - len(arrs) + 1) / (ntot - len(arrs)) if ntot > len(arrs) else np.nan
                rows.append(dict(variable=s, test="Kruskal-Wallis", stat=round(float(H), 2),
                                 p=p, effect=round(float(eta2), 3)))
        if rows:
            gt = pd.DataFrame(rows)
            gt["p_fdr"] = bh_fdr(gt["p"].values)
            gt["sig"] = np.where(gt["p_fdr"] < 0.05, "*", "")
            gt["p"] = gt["p"].round(4); gt["p_fdr"] = gt["p_fdr"].round(4)
            ctx.save_table(gt, "group_tests", index=False)
            ctx.log(f"群差異 by {group_col}（groups={labels}）→ group_tests.csv")
            did.append("群差異")

    # 2. 趨勢
    if time_col and time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
        torder = df.assign(_t=t).sort_values("_t")
        tnum = (pd.to_datetime(torder["_t"]) - pd.to_datetime(torder["_t"]).min()).dt.days.values
        rows = []
        for s in species:
            mk = mann_kendall(torder[s].values, norm)
            mk.update(variable=s, sen_slope_per_day=round(sen_slope(torder[s].values, tnum), 5))
            rows.append(mk)
        tt = pd.DataFrame(rows)[["variable", "n", "S", "Z", "p", "sen_slope_per_day", "trend"]]
        if len(tt):
            tt["p_fdr"] = bh_fdr(tt["p"].fillna(1).values).round(4)
            ctx.save_table(tt, "trend_tests", index=False)
            ctx.log("Mann-Kendall 趨勢 + Sen slope → trend_tests.csv")
            did.append("趨勢")

    # 3. Spearman + FDR + 熱圖
    sp_df = df[species]
    rho = sp_df.corr(method="spearman")
    cols = list(sp_df.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = sp_df[cols[i]], sp_df[cols[j]]
            m = a.notna() & b.notna()
            if m.sum() >= 4:
                r, p = spearmanr(a[m], b[m])
                pairs.append((i, j, p))
    if pairs:
        qs = bh_fdr([p for *_, p in pairs])
        qmat = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
        for (i, j, _), q in zip(pairs, qs):
            qmat.iloc[i, j] = qmat.iloc[j, i] = q
        ctx.save_table(rho.round(3), "spearman_rho")
        ctx.save_table(qmat.round(4), "spearman_fdr")
        ctx.log(f"Spearman {len(pairs)} 對相關 + FDR q → spearman_rho.csv / spearman_fdr.csv")
        did.append("相關")

        fig, ax = plt.subplots(figsize=(1 + 0.6 * len(cols), 1 + 0.6 * len(cols)))
        im = ax.imshow(rho.values, cmap=cmap_div, vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=7)
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7)
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i != j and qmat.iloc[i, j] < 0.05:
                    ax.text(j, i, "*", ha="center", va="center", color="black", fontsize=8)
        fig.colorbar(im, fraction=0.046)
        ax.set_title("Spearman ρ  (* = FDR<0.05)")
        fig.tight_layout(); ctx.save_fig(fig, "spearman_heatmap")

    if not did:
        raise ValueError("沒有產生任何結果：請確認有數值欄，或指定 group/time 欄。")
    return ctx.result(summary="無母數統計完成：" + "、".join(did) + "（皆含 BH-FDR 校正）。")


SPEC = MethodSpec(
    key="nonparam",
    name="無母數統計",
    summary="群差異 / 趨勢 / Spearman 相關，含 BH-FDR 多重比較校正（適合右偏、多 BDL 資料）。",
    params=[
        ParamSpec("id_col", "ID 欄（可空）", "column", default="sample_id", optional=True),
        ParamSpec("group_col", "分組欄（做群差異，可空）", "column", default="season", optional=True,
                  help="如 season / site；2 群用 Mann-Whitney，>2 群用 Kruskal-Wallis。"),
        ParamSpec("time_col", "時間欄（做趨勢，可空）", "column", default="date", optional=True,
                  help="如 date；做 Mann-Kendall 趨勢與 Sen's slope。"),
    ],
    schema=InputSchema(min_rows=4, min_numeric_cols=1, id_col_param="id_col"),
    template_columns=["sample_id", "season", "date", "PFOA", "PFOS", "…"],
    uses_colors=["cmap_diverging"],
)
SPEC.run = run
SPEC.make_demo = make_demo
