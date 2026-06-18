# -*- coding: utf-8 -*-
"""
trajectory.py — 軌跡受體模型 PSCF / CWT（潛在源區辨識）
=========================================================
針對「大氣長程傳輸」最核心的問題：到站的污染來自哪個地理區域？
輸入「長格式」後向軌跡端點 CSV，每列一個端點：
  軌跡編號 traj_id、緯度 lat、經度 lon、該軌跡對應採樣的到站濃度 conc
（端點通常由 HYSPLIT / TrajStat 產生；同一條軌跡所有端點共用同一個到站濃度。）

輸出：
  ‧ trajectory_grid.csv（各網格中心經緯、端點數 n、PSCF、CWT）
  ‧ PSCF 圖（高濃度條件機率熱點）、CWT 圖（濃度權重來源強度）

原理：
  PSCF_ij = m_ij / n_ij      經過格 ij 且屬高濃度軌跡的端點比例
  CWT_ij  = Σ C_l·τ_ijl / Σ τ_ijl    端點濃度的居留時間加權平均
  兩者皆乘上隨 n_ij 遞增的經驗權重 W_ij，抑制少端點網格的雜訊。
"""
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.theme import get_plt


def make_demo(seed=7):
    """模擬 80 條 72 小時後向軌跡；經過污染熱區者到站濃度較高。"""
    rng = np.random.default_rng(seed)
    r_lat, r_lon = 23.5, 120.9          # 受體站（示意：台灣中部山區）
    n_traj, n_hours = 80, 73
    hot_boxes = [((28, 34), (112, 118)),    # 西北：華中/華東工業區
                 ((20, 24), (118, 122))]    # 近距：海陸交界
    rows = []
    for t in range(n_traj):
        to_nw = rng.random() < 0.55
        dlat = rng.normal(0.10 if to_nw else -0.05, 0.03)
        dlon = rng.normal(-0.14 if to_nw else 0.10, 0.03)
        lat, lon = r_lat, r_lon
        path, passed = [], 0
        for _ in range(n_hours):
            lat += dlat + rng.normal(0, 0.06)
            lon += dlon + rng.normal(0, 0.06)
            path.append((lat, lon))
            for (la_r, lo_r) in hot_boxes:
                if la_r[0] <= lat <= la_r[1] and lo_r[0] <= lon <= lo_r[1]:
                    passed += 1
        conc = max(rng.lognormal(0.0, 0.3) + 0.04 * passed + rng.normal(0, 0.1), 0.01)
        for (la, lo) in path:
            rows.append((f"T{t:03d}", round(la, 3), round(lo, 3), round(conc, 3)))
    return pd.DataFrame(rows, columns=["traj_id", "lat", "lon", "conc"])


def _weight(n, nmean):
    """TrajStat 式經驗權重：端點越少權重越低，抑制偶發軌跡造成的假熱點。"""
    w = np.ones_like(n, dtype=float)
    if nmean <= 0:
        return w
    w = np.where(n <= 3.0 * nmean, 0.70, w)
    w = np.where(n <= 1.5 * nmean, 0.42, w)
    w = np.where(n <= 1.0 * nmean, 0.17, w)
    return w


def run(df, params, ctx):
    plt = get_plt(ctx.theme)
    traj_col = params.get("traj_col") or "traj_id"
    lat_col = params.get("lat_col") or "lat"
    lon_col = params.get("lon_col") or "lon"
    conc_col = params.get("conc_col") or "conc"
    grid = float(params.get("grid_size", 1.0))
    thr_pct = float(params.get("threshold_pct", 75))

    for c in (traj_col, lat_col, lon_col, conc_col):
        if c not in df.columns:
            raise ValueError(f"找不到欄位『{c}』，請在參數面板指定正確欄位。")

    d = df[[traj_col, lat_col, lon_col, conc_col]].copy()
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")
    d[conc_col] = pd.to_numeric(d[conc_col], errors="coerce")
    d = d.dropna()
    if len(d) < 10:
        raise ValueError("有效軌跡端點少於 10 個，無法建立網格統計。")
    n_traj = d[traj_col].nunique()
    ctx.log(f"端點 {len(d)} 個；軌跡 {n_traj} 條；網格邊長 {grid}°")

    # 每條軌跡的代表到站濃度（理論上整條相同，取中位數防雜訊）
    traj_conc = d.groupby(traj_col)[conc_col].median()
    thr = float(np.percentile(traj_conc.values, thr_pct))
    high_trajs = set(traj_conc[traj_conc > thr].index)
    ctx.log(f"高濃度門檻（第 {thr_pct:.0f} 百分位）= {thr:.3f}；高濃度軌跡 {len(high_trajs)} 條")

    lat_edges = np.arange(np.floor(d[lat_col].min()), np.ceil(d[lat_col].max()) + grid, grid)
    lon_edges = np.arange(np.floor(d[lon_col].min()), np.ceil(d[lon_col].max()) + grid, grid)
    ni, nj = len(lat_edges) - 1, len(lon_edges) - 1
    if ni < 1 or nj < 1:
        raise ValueError("經緯度範圍太小或網格太大，無法切出網格。請調小網格邊長。")

    iy = np.clip(np.digitize(d[lat_col].values, lat_edges) - 1, 0, ni - 1)
    ix = np.clip(np.digitize(d[lon_col].values, lon_edges) - 1, 0, nj - 1)
    is_high = d[traj_col].isin(high_trajs).values
    cvals = d[conc_col].values

    n_cell = np.zeros((ni, nj)); m_cell = np.zeros((ni, nj)); csum = np.zeros((ni, nj))
    np.add.at(n_cell, (iy, ix), 1.0)
    np.add.at(csum, (iy, ix), cvals)
    np.add.at(m_cell, (iy[is_high], ix[is_high]), 1.0)

    nmean = n_cell[n_cell > 0].mean() if (n_cell > 0).any() else 0.0
    W = _weight(n_cell, nmean)
    with np.errstate(invalid="ignore", divide="ignore"):
        pscf = np.where(n_cell > 0, m_cell / n_cell, np.nan) * W
        cwt = np.where(n_cell > 0, csum / n_cell, np.nan) * W
    pscf[n_cell == 0] = np.nan
    cwt[n_cell == 0] = np.nan

    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
    recs = []
    for i in range(ni):
        for j in range(nj):
            if n_cell[i, j] > 0:
                recs.append(dict(lat=round(float(lat_c[i]), 3), lon=round(float(lon_c[j]), 3),
                                 n_endpoints=int(n_cell[i, j]),
                                 PSCF=round(float(pscf[i, j]), 3),
                                 CWT=round(float(cwt[i, j]), 3)))
    grid_df = pd.DataFrame(recs).sort_values("CWT", ascending=False)
    ctx.save_table(grid_df, "trajectory_grid", index=False)
    ctx.log(f"網格統計 {len(grid_df)} 個有效格 → trajectory_grid.csv")
    if len(grid_df):
        top = grid_df.iloc[0]
        ctx.log(f"CWT 最高來源格：lat={top['lat']}, lon={top['lon']}（CWT={top['CWT']}）")

    cmap_seq = ctx.color("cmap_sequential", "viridis")
    accent = ctx.color("accent", "#ff6347")
    LON, LAT = np.meshgrid(lon_edges, lat_edges)
    for field, name, title, lab in [
        (pscf, "pscf_map", "PSCF 潛在源區（高濃度條件機率）", "PSCF"),
        (cwt, "cwt_map", "CWT 濃度權重軌跡（來源強度）", "CWT"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        pm = ax.pcolormesh(LON, LAT, np.ma.masked_invalid(field), cmap=cmap_seq, shading="auto")
        fig.colorbar(pm, ax=ax, fraction=0.046, label=lab)
        ax.scatter([d[lon_col].median()], [d[lat_col].median()],
                   marker="*", s=10, color=accent, alpha=0)  # 佔位，維持比例
        ax.set_xlabel("經度 lon"); ax.set_ylabel("緯度 lat"); ax.set_title(title)
        fig.tight_layout(); ctx.save_fig(fig, name)

    return ctx.result(summary=f"PSCF/CWT 完成：{len(grid_df)} 個網格。"
                              "PSCF 圖顯示『高濃度條件機率』最高的潛在源區；"
                              "CWT 圖進一步給出濃度權重的來源強度（可區分中度與強來源）。"
                              "trajectory_grid.csv 可疊到地圖（QGIS/底圖）上呈現。")


SPEC = MethodSpec(
    key="trajectory",
    name="軌跡受體模型 PSCF/CWT",
    summary="用後向軌跡端點＋到站濃度做 PSCF（潛在源區條件機率）與 CWT（濃度權重來源強度），辨識長程傳輸的地理來源。",
    params=[
        ParamSpec("traj_col", "軌跡編號欄", "column", default="traj_id",
                  help="同一條後向軌跡的所有端點共用一個編號。"),
        ParamSpec("lat_col", "緯度欄", "column", default="lat"),
        ParamSpec("lon_col", "經度欄", "column", default="lon"),
        ParamSpec("conc_col", "到站濃度欄", "column", default="conc",
                  help="該軌跡對應採樣的受體濃度（沿整條端點重複）。"),
        ParamSpec("grid_size", "網格邊長（度）", "float", default=1.0, minimum=0.1, maximum=10.0,
                  help="越小解析度越高，但每格端點數變少、雜訊變大。"),
        ParamSpec("threshold_pct", "高濃度百分位門檻", "float", default=75.0, minimum=50.0, maximum=95.0,
                  help="PSCF 用：到站濃度超過此百分位者視為高濃度軌跡。"),
    ],
    schema=InputSchema(min_rows=10, min_numeric_cols=0,
                       required_param_cols=["traj_col", "lat_col", "lon_col", "conc_col"],
                       note="長格式：每列一個軌跡端點（traj_id, lat, lon, conc）。"),
    template_columns=["traj_id", "lat", "lon", "conc"],
    uses_colors=["cmap_sequential", "accent"],
)
SPEC.run = run
SPEC.make_demo = make_demo
