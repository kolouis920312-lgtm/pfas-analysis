# -*- coding: utf-8 -*-
"""
som.py — 自組織映射 SOM 指紋拓樸（由 som_fingerprint.py 整合而來）
輸出：som_bmu / som_node_clusters / som_site_regime / som_site_overlap（CSV）
      + component planes / U-matrix / 站落點 / 季落點（圖）
建議輸入先經 CoDA 的 clr_transformed.csv（否則自動 z-score 標準化）。
"""
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.theme import get_plt
from ..core.prep import as_list, apply_value_filter


class SOM:
    def __init__(self, m, n, dim, rng):
        self.m, self.n, self.dim = m, n, dim
        self.rng = rng
        self.W = rng.normal(0, 1, size=(m * n, dim))
        self.coords = np.array([(i, j) for i in range(m) for j in range(n)], float)

    def _bmu(self, x):
        return int(np.argmin(((self.W - x) ** 2).sum(1)))

    def train(self, X, iters, lr0, sig0):
        for t in range(iters):
            lr = lr0 * np.exp(-t / iters)
            sig = max(sig0 * np.exp(-t / iters), 0.5)
            x = X[self.rng.integers(len(X))]
            b = self._bmu(x)
            d2 = ((self.coords - self.coords[b]) ** 2).sum(1)
            h = np.exp(-d2 / (2 * sig ** 2))
            self.W += lr * h[:, None] * (x - self.W)

    def bmus(self, X):
        return np.array([self._bmu(x) for x in X])

    def umatrix(self):
        U = np.zeros(self.m * self.n)
        for k in range(self.m * self.n):
            nb = ((self.coords - self.coords[k]) ** 2).sum(1)
            neigh = (nb > 0) & (nb <= 2)
            U[k] = np.mean(np.sqrt(((self.W[neigh] - self.W[k]) ** 2).sum(1))) if neigh.any() else 0
        return U.reshape(self.m, self.n)


def make_demo(seed=42):
    rng = np.random.default_rng(seed)
    sites = np.repeat(["清邁", "鹿林", "楠梓"], 12)
    sp = ["PFPeA", "PFHxA", "PFOA", "PFNA", "PFOS", "PFHxS", "6:2 FTS", "FOSA"]
    blocks = {"清邁": [3, 3, 2, 1, 0.5, 0.5, 2, 1],
              "鹿林": [2, 2, 1, 1, 0.3, 0.3, 1, 0.5],
              "楠梓": [1, 1, 2, 1, 3, 2, 0.5, 0.3]}
    X = [np.array(blocks[s]) * rng.lognormal(0, 0.25, len(sp)) for s in sites]
    df = pd.DataFrame(X, columns=sp)
    df.insert(0, "season", np.tile(["春"], len(sites)))
    df.insert(0, "site", sites)
    df.insert(0, "sample_id", [f"S{i:02d}" for i in range(len(sites))])
    return df


def overlap_analysis(out, node_regime, n_regime, site_col):
    out = out.copy()
    out["regime"] = out["bmu"].map(dict(zip(node_regime["node"], node_regime["regime"])))
    ct = pd.crosstab(out[site_col], out["regime"]).reindex(columns=range(n_regime), fill_value=0)
    ct.columns = [f"regime{c}" for c in ct.columns]
    prop = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    sites = prop.index.tolist()
    sim = pd.DataFrame(np.eye(len(sites)), index=sites, columns=sites)
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            s = float(np.minimum(prop.loc[a], prop.loc[b]).sum())
            sim.loc[a, b] = sim.loc[b, a] = round(s, 3)
    return ct, sim


def _hits_plot(ctx, plt, out, cats, cmap_cat, rng, title, name):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter(out["bmu_col"] + rng.normal(0, 0.12, len(out)),
               out["bmu_row"] + rng.normal(0, 0.12, len(out)),
               c=cats.cat.codes, cmap=cmap_cat, s=40)
    ax.set_title(title + "\n每點＝一個樣本落在它最像的格子；同色聚一區＝該組指紋一致、散開＝型態多樣",
                 fontsize=10)
    ax.set_xlabel("行 (col)"); ax.set_ylabel("列 (row)"); ax.invert_yaxis()
    cmobj = plt.get_cmap(cmap_cat)
    k = max(len(cats.cat.categories) - 1, 1)
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmobj(i / k), label=c)
               for i, c in enumerate(cats.cat.categories)]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout(); ctx.save_fig(fig, name)


def run(df, params, ctx):
    from sklearn.cluster import KMeans
    plt = get_plt(ctx.theme)

    seed = 42
    rng = np.random.default_rng(seed)
    id_col = params.get("id_col") or None
    site_col = params.get("site_col") or None
    season_col = params.get("season_col") or None
    if site_col in ("(無)", ""):
        site_col = None
    if season_col in ("(無)", ""):
        season_col = None
    grid_m = int(params.get("grid_m", 7)); grid_n = int(params.get("grid_n", 7))
    iters = int(params.get("iters", 5000)); lr0 = float(params.get("lr0", 0.5))
    n_regime = int(params.get("n_regime", 4))
    focus_a = (params.get("focus_a") or "").strip()
    focus_b = (params.get("focus_b") or "").strip()
    cmap_seq = ctx.color("cmap_sequential", "viridis")
    cmap_cat = ctx.color("cmap_categorical", "tab10")

    # ── 先依「站別/季節」篩選要納入的樣本（空＝全部）──
    df = apply_value_filter(df, ctx, site_col, params.get("site_keep"), what="站別樣本")
    df = apply_value_filter(df, ctx, season_col, params.get("season_keep"), what="季節樣本")

    meta_cols = [c for c in (id_col, site_col, season_col) if c and c in df.columns]
    feats = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    # ── 化合物子集（空＝全部）──
    keep_feats = as_list(params.get("feature_cols"))
    if keep_feats:
        miss = [c for c in keep_feats if c not in feats]
        if miss:
            ctx.log(f"⚠ 指定的化合物不在數值欄中，已略過：{miss}")
        feats = [c for c in keep_feats if c in feats]
        ctx.log(f"只納入選定的 {len(feats)} 個化合物：{feats}")
    if len(feats) < 2:
        raise ValueError("SOM 需至少 2 個數值特徵（化合物）。若有用『納入化合物』篩選，請至少選 2 個。")
    X = df[feats].values.astype(float)
    X = np.nan_to_num(X, nan=np.nanmedian(X))
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ctx.log(f"SOM 輸入：{X.shape[0]} 樣本 × {X.shape[1]} 特徵；grid={grid_m}×{grid_n}")

    sig0 = max(grid_m, grid_n) / 2
    som = SOM(grid_m, grid_n, X.shape[1], rng)
    som.train(X, iters, lr0, sig0)
    bmu = som.bmus(X)

    out = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)
    out["bmu"] = bmu
    out["bmu_row"] = bmu // grid_n
    out["bmu_col"] = bmu % grid_n
    ctx.save_table(out, "som_bmu", index=False)

    km = KMeans(n_clusters=min(n_regime, grid_m * grid_n), n_init=10, random_state=seed).fit(som.W)
    nodes = pd.DataFrame({"node": range(grid_m * grid_n),
                          "row": som.coords[:, 0].astype(int),
                          "col": som.coords[:, 1].astype(int),
                          "regime": km.labels_})
    ctx.save_table(nodes, "som_node_clusters", index=False)
    ctx.log(f"SOM {grid_m}×{grid_n} 訓練完成；節點分 {n_regime} regime。")

    if site_col and site_col in df.columns:
        ct, sim = overlap_analysis(out, nodes, n_regime, site_col)
        ctx.save_table(ct, "som_site_regime")
        ctx.save_table(sim, "som_site_overlap")
        ctx.log("站 × regime 落點 + 站對站重疊度 → som_site_regime.csv / som_site_overlap.csv")
        if focus_a and focus_b and focus_a in sim.index and focus_b in sim.index:
            v = sim.loc[focus_a, focus_b]
            verdict = ("高度重疊→指紋型態一致，支持 LRT 同源" if v >= 0.6 else
                       "中度重疊→部分共享，需配合軌跡/診斷比值佐證" if v >= 0.3 else
                       "低度重疊→指紋型態分離")
            ctx.log(f"焦點站對 {focus_a}↔{focus_b} 重疊度={v:.3f} → {verdict}")
        elif focus_a or focus_b:
            ctx.log(f"焦點站對 {focus_a}/{focus_b} 在資料中找不到，已略過。")

    # ── regime 地圖（把節點再分群的結果塗色；過去只有 CSV、沒有圖）──
    regime_grid = km.labels_.reshape(grid_m, grid_n)
    n_used = int(len(np.unique(km.labels_)))
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    disc = plt.get_cmap(cmap_cat, max(n_used, 1))
    im = ax.imshow(regime_grid, cmap=disc, vmin=-0.5, vmax=n_used - 0.5)
    for (r, c), lab in np.ndenumerate(regime_grid):
        ax.text(c, r, str(int(lab)), ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
    ax.set_title("Regime 地圖：節點屬於哪個來源型態區")
    ax.set_xlabel("行 (col)"); ax.set_ylabel("列 (row)")
    cbar = fig.colorbar(im, ax=ax, ticks=range(n_used), fraction=0.046)
    cbar.set_label("regime 編號")
    fig.tight_layout(); ctx.save_fig(fig, "som_regime_map")

    # ── Component planes（每張小圖＝同一張地圖、用一個物種上色）──
    # 用「共用色階」讓各物種可比較；亮＝該格典型指紋裡此物種相對高。
    ncol = min(4, len(feats)); nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.7 * nrow + 0.9),
                             constrained_layout=True)
    axflat = np.atleast_1d(axes).ravel()
    vmin, vmax = float(som.W.min()), float(som.W.max())
    im = None
    for ax, k in zip(axflat, range(len(feats))):
        im = ax.imshow(som.W[:, k].reshape(grid_m, grid_n), cmap=cmap_seq,
                       vmin=vmin, vmax=vmax)
        ax.set_title(feats[k], fontsize=9); ax.axis("off")
    for ax in axflat[len(feats):]:
        ax.axis("off")
    if im is not None:
        cb = fig.colorbar(im, ax=axflat.tolist(), fraction=0.03, pad=0.02)
        cb.set_label("權重（標準化後；亮=相對高、暗=相對低）")
    fig.suptitle("Component planes　每張小圖＝同一張 SOM 地圖、用一個物種上色\n"
                 "亮＝該區典型指紋中此物種相對高；亮區重疊的物種＝常一起出現（可能同源）",
                 fontsize=10)
    ctx.save_fig(fig, "som_components")

    # U-matrix
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(som.umatrix(), cmap=cmap_seq); fig.colorbar(im, ax=ax)
    ax.set_title("U-matrix：相鄰節點的指紋差異\n亮＝差異大（不同群的邊界）；暗成一片＝同一種型態的區域",
                 fontsize=10)
    ax.set_xlabel("行 (col)"); ax.set_ylabel("列 (row)")
    fig.tight_layout(); ctx.save_fig(fig, "som_umatrix")

    if site_col and site_col in df.columns:
        _hits_plot(ctx, plt, out, df[site_col].astype("category"), cmap_cat, rng,
                   "各站在 SOM 拓樸落點", "som_hits_site")
    if season_col and season_col in df.columns:
        _hits_plot(ctx, plt, out, df[season_col].astype("category"), cmap_cat, rng,
                   "各季在 SOM 拓樸落點", "som_hits_season")

    return ctx.result(summary=f"SOM {grid_m}×{grid_n} 完成：BMU、regime 地圖、component planes、U-matrix"
                              + ("、站/季落點與重疊度" if site_col else "") + "。")


SPEC = MethodSpec(
    key="som",
    name="SOM 指紋拓樸",
    summary="自組織映射把高維 PFAS 指紋映到 2D，看站/季來源型態、component planes 與站對站重疊度。",
    params=[
        ParamSpec("id_col", "ID 欄（可空）", "column", default="sample_id", optional=True),
        ParamSpec("site_col", "站別欄（可空）", "column", default="site", optional=True),
        ParamSpec("season_col", "季節欄（可空）", "column", default="season", optional=True),
        ParamSpec("feature_cols", "納入的化合物（不選＝全部）", "columns", default=[],
                  help="勾選要餵進 SOM 的物種；不勾就用全部數值欄。"),
        ParamSpec("site_keep", "只納入這些站別（不選＝全部）", "values", default=[],
                  source_col="site_col",
                  help="只想看某幾站時勾選；對選到的樣本子集跑單一張 SOM。"),
        ParamSpec("season_keep", "只納入這些季節（不選＝全部）", "values", default=[],
                  source_col="season_col",
                  help="只想看某幾季時勾選。"),
        ParamSpec("grid_m", "SOM 列數 m", "int", default=7, minimum=2),
        ParamSpec("grid_n", "SOM 行數 n", "int", default=7, minimum=2),
        ParamSpec("iters", "訓練迭代數", "int", default=5000, minimum=100),
        ParamSpec("lr0", "初始學習率", "float", default=0.5, minimum=0.01, maximum=1.0),
        ParamSpec("n_regime", "節點再分群數 (regime)", "int", default=4, minimum=2),
        ParamSpec("focus_a", "焦點站 A（可空）", "text", default="清邁", optional=True),
        ParamSpec("focus_b", "焦點站 B（可空）", "text", default="鹿林", optional=True),
    ],
    schema=InputSchema(min_rows=4, min_numeric_cols=2, id_col_param="id_col"),
    template_columns=["sample_id", "site", "season", "PFPeA", "PFOA", "…"],
    uses_colors=["cmap_sequential", "cmap_categorical"],
)
SPEC.run = run
SPEC.make_demo = make_demo
