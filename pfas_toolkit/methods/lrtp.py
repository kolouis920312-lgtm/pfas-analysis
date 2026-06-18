# -*- coding: utf-8 -*-
"""
lrtp.py — 長程傳輸潛力：大氣壽命 + 特徵遷移距離 CTD
=========================================================
量化「某化合物能飄多遠」。以與 OH 自由基反應為主要光化學損失：
  kdeg = kOH · [OH]                    一階光化學降解速率 (s⁻¹)
  ktot = kdeg + kwet + kdry            含乾濕沉降的總一階損失（可選）
  大氣壽命 τ = 1/ktot；半衰期 t½ = ln2/ktot
  CTD = u · τ                          特徵遷移距離（van Pul 等）
        ‧ CTD_efold = u·(1/ktot)       濃度衰減到 1/e 的距離
        ‧ CTD_half  = u·(ln2/ktot)     文獻常用的「半距離」(50% 損失)

輸入：每列一個化合物，需有 kOH 欄（cm³ molec⁻¹ s⁻¹）；可選 kwet、kdry（s⁻¹）。
參數：[OH] 平均濃度、風速 u。輸出：lrtp_ctd.csv ＋ CTD 排序長條圖。
"""
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.theme import get_plt


def make_demo(seed=9):
    """kOH 與 OH 之雙分子反應速率常數 (cm³ molec⁻¹ s⁻¹)；短壽命到長壽命都有。"""
    data = [
        ("6:2 FTOH", 1.07e-12), ("8:2 FTOH", 1.08e-12), ("10:2 FTOH", 1.10e-12),
        ("MeFOSE", 2.00e-12), ("EtFOSE", 2.20e-12), ("8:2 FTAc", 5.00e-12),
        ("PFBA", 1.70e-13), ("PFOA", 1.70e-13), ("PFPMIE(惰性參考)", 1.00e-15),
    ]
    return pd.DataFrame(data, columns=["compound", "kOH"])


def run(df, params, ctx):
    plt = get_plt(ctx.theme)
    id_col = params.get("id_col") or None
    if id_col in ("(無)", ""):
        id_col = None
    koh_col = params.get("koh_col") or "kOH"
    oh = float(params.get("oh_conc", 1.0e6))
    u = float(params.get("wind_u", 4.0))
    kwet_col = params.get("kwet_col") or None
    kdry_col = params.get("kdry_col") or None
    if kwet_col in ("(無)", ""):
        kwet_col = None
    if kdry_col in ("(無)", ""):
        kdry_col = None
    if oh <= 0 or u <= 0:
        raise ValueError("[OH] 與風速 u 都需 > 0。")
    if koh_col not in df.columns:
        raise ValueError(f"找不到 kOH 欄『{koh_col}』。")

    raw_names = (df[id_col].astype(str).values if (id_col and id_col in df.columns)
                 else np.array([f"C{i+1}" for i in range(len(df))]))
    koh_all = pd.to_numeric(df[koh_col], errors="coerce").values
    kwet_all = (pd.to_numeric(df[kwet_col], errors="coerce").values
                if (kwet_col and kwet_col in df.columns) else np.zeros(len(df)))
    kdry_all = (pd.to_numeric(df[kdry_col], errors="coerce").values
                if (kdry_col and kdry_col in df.columns) else np.zeros(len(df)))
    kwet_all = np.nan_to_num(kwet_all); kdry_all = np.nan_to_num(kdry_all)

    ok = ~np.isnan(koh_all) & (koh_all > 0)
    if ok.sum() == 0:
        raise ValueError("kOH 欄沒有有效正值。")
    names = raw_names[ok]
    koh = koh_all[ok]; kwet = kwet_all[ok]; kdry = kdry_all[ok]

    kdeg = koh * oh                  # s⁻¹
    ktot = kdeg + kwet + kdry
    day = 86400.0
    tau_e = 1.0 / ktot               # e-folding 壽命 (s)
    thalf = np.log(2) / ktot         # 半衰期 (s)
    ctd_efold = tau_e * u / 1000.0   # km
    ctd_half = thalf * u / 1000.0    # km

    incl_dep = bool((kwet > 0).any() or (kdry > 0).any())
    ctx.log(f"[OH]={oh:.2e} molec/cm³；風速 u={u} m/s；化合物 {len(names)} 個"
            + ("；含乾濕沉降" if incl_dep else "；僅光化學損失"))

    def lrt_class(km):
        return ("高 (>2000 km)" if km > 2000 else
                "中 (500–2000 km)" if km > 500 else "低 (<500 km)")

    out = pd.DataFrame({
        "compound": names,
        "kOH": koh,
        "kdeg_per_s": kdeg,
        "lifetime_days": np.round(tau_e / day, 2),
        "half_life_days": np.round(thalf / day, 2),
        "CTD_efold_km": np.round(ctd_efold, 0),
        "CTD_half_km": np.round(ctd_half, 0),
        "LRT_potential": [lrt_class(k) for k in ctd_efold],
    }).sort_values("CTD_efold_km", ascending=False)
    ctx.save_table(out, "lrtp_ctd", index=False)
    hi = out[out["CTD_efold_km"] > 2000]["compound"].tolist()
    ctx.log(f"高長程傳輸潛力（CTD>2000 km）：{hi if hi else '（無）'}")

    primary = ctx.color("primary", "#4682b4")
    names_o = out["compound"].values
    ctd_o = out["CTD_efold_km"].values
    x = np.arange(len(names_o))
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(names_o)), 5))
    ax.bar(x, np.clip(ctd_o, 1, None), color=primary)
    ax.set_yscale("log")
    ax.axhline(2000, ls="--", color="gray", lw=0.8, label="2000 km")
    ax.set_xticks(x); ax.set_xticklabels(names_o, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("CTD (km, log)"); ax.set_title("特徵遷移距離 CTD（長程傳輸潛力，越大越遠）")
    ax.legend()
    fig.tight_layout(); ctx.save_fig(fig, "lrtp_ctd")

    return ctx.result(summary=f"長程傳輸潛力完成：{len(names_o)} 個化合物。"
                              "大氣壽命 τ=1/(kOH·[OH](+沉降))，CTD=u·τ。"
                              "CTD 越大代表能飄越遠；可比較前驅物與其氧化產物的傳輸潛力差異。")


SPEC = MethodSpec(
    key="lrtp",
    name="長程傳輸潛力 CTD",
    summary="由 kOH 與 [OH] 算大氣壽命，配風速得特徵遷移距離 CTD，量化各化合物能飄多遠（可含乾濕沉降）。",
    params=[
        ParamSpec("id_col", "化合物名稱欄（可空）", "column", default="compound", optional=True),
        ParamSpec("koh_col", "kOH 欄 (cm³ molec⁻¹ s⁻¹)", "column", default="kOH",
                  help="與 OH 自由基的雙分子反應速率常數。"),
        ParamSpec("oh_conc", "平均 [OH] (molec/cm³)", "float", default=1.0e6, minimum=1.0e4,
                  help="大氣 OH 自由基平均濃度，文獻常用 1×10⁶。"),
        ParamSpec("wind_u", "平均風速 u (m/s)", "float", default=4.0, minimum=0.1,
                  help="用於把壽命換算成傳輸距離。"),
        ParamSpec("kwet_col", "濕沉降速率欄 kwet (可空)", "column", default="", optional=True,
                  help="一階濕沉降速率 (s⁻¹)；不填則僅算光化學損失。"),
        ParamSpec("kdry_col", "乾沉降速率欄 kdry (可空)", "column", default="", optional=True,
                  help="一階乾沉降速率 (s⁻¹)。"),
    ],
    schema=InputSchema(min_rows=1, min_numeric_cols=0, id_col_param="id_col",
                       required_param_cols=["koh_col"],
                       note="每列一個化合物；需 kOH 欄，沉降速率為選填。"),
    template_columns=["compound", "kOH"],
    uses_colors=["primary"],
)
SPEC.run = run
SPEC.make_demo = make_demo
