# -*- coding: utf-8 -*-
"""
partitioning.py — 氣粒分配（KOA 吸收模型）
=================================================
決定 PFAS 在大氣中以「氣相」或「粒相」存在，這直接左右長程傳輸方式：
偏氣相者（如 FTOHs）以氣態飄送遠距並沿途氧化生成 PFCA；偏粒相者易隨微粒沉降。

KOA 吸收模型（Pankow 1994；文獻 lgKp = lgKOA + lg(fom) − 11.91）：
  Kp = 10^(lgKOA + lg(fom) − 11.91)        分配係數 (m³/µg)
  φ  = Kp·TSP / (1 + Kp·TSP)               粒相比例
  φ_gas = 1 − φ                            氣相比例
輸入：每列一個化合物，需有 log KOA 欄；參數給 fom（顆粒有機質分率）與 TSP（µg/m³）。

輸出：gas_particle_partitioning.csv（各化合物 Kp、φ_particle、φ_gas、主導相）＋ 堆疊長條圖。
"""
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.theme import get_plt


def make_demo(seed=5):
    """代表性 log KOA（25°C 區間）：中性揮發前驅物偏氣相，半揮發者偏粒相。"""
    data = [
        ("4:2 FTOH", 5.6), ("6:2 FTOH", 6.4), ("8:2 FTOH", 7.2), ("10:2 FTOH", 8.0),
        ("FOSA", 8.3), ("MeFOSE", 8.9), ("EtFOSE", 9.2), ("MeFOSA", 9.8),
        ("長鏈PFCA(粒相代表)", 11.5), ("離子型PFSA(粒相代表)", 12.5),
    ]
    return pd.DataFrame(data, columns=["compound", "log_KOA"])


def run(df, params, ctx):
    plt = get_plt(ctx.theme)
    id_col = params.get("id_col") or None
    if id_col in ("(無)", ""):
        id_col = None
    koa_col = params.get("koa_col") or "log_KOA"
    fom = float(params.get("fom", 0.1))
    tsp = float(params.get("tsp", 30.0))
    if fom <= 0:
        raise ValueError("fom（顆粒有機質分率）需 > 0。")
    if tsp <= 0:
        raise ValueError("TSP（總懸浮微粒）需 > 0 µg/m³。")
    if koa_col not in df.columns:
        raise ValueError(f"找不到 log KOA 欄『{koa_col}』。")

    raw_names = (df[id_col].astype(str).values if (id_col and id_col in df.columns)
                 else np.array([f"C{i+1}" for i in range(len(df))]))
    logkoa_all = pd.to_numeric(df[koa_col], errors="coerce").values
    ok = ~np.isnan(logkoa_all)
    if ok.sum() == 0:
        raise ValueError("log KOA 欄沒有有效數值。")
    names = raw_names[ok]
    logkoa = logkoa_all[ok]

    log_kp = logkoa + np.log10(fom) - 11.91          # m³/µg
    kp = 10.0 ** log_kp
    phi = (kp * tsp) / (1.0 + kp * tsp)              # 粒相比例
    phi_gas = 1.0 - phi
    ctx.log(f"化合物 {len(names)} 個；fom={fom}；TSP={tsp} µg/m³")
    ctx.log(f"以氣相為主（φ_gas>0.5）{int((phi_gas > 0.5).sum())} 個 → 傾向氣態長程傳輸")

    out = pd.DataFrame({
        "compound": names,
        "log_KOA": np.round(logkoa, 2),
        "log_Kp": np.round(log_kp, 3),
        "Kp_m3_per_ug": kp,
        "phi_particle": np.round(phi, 4),
        "phi_gas": np.round(phi_gas, 4),
        "dominant_phase": np.where(phi >= 0.5, "粒相 particle", "氣相 gas"),
    }).sort_values("phi_gas", ascending=False)
    ctx.save_table(out, "gas_particle_partitioning", index=False)

    primary = ctx.color("primary", "#4682b4")
    accent = ctx.color("accent", "#ff6347")
    names_o = out["compound"].values
    phig_o = out["phi_gas"].values
    phip_o = out["phi_particle"].values
    x = np.arange(len(names_o))
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(names_o)), 5))
    ax.bar(x, phig_o, color=primary, label="氣相 gas")
    ax.bar(x, phip_o, bottom=phig_o, color=accent, label="粒相 particle")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names_o, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("比例"); ax.set_ylim(0, 1)
    ax.set_title(f"氣粒分配（KOA 吸收模型；fom={fom}, TSP={tsp} µg/m³）")
    ax.legend()
    fig.tight_layout(); ctx.save_fig(fig, "gas_particle_partitioning")

    return ctx.result(summary=f"氣粒分配完成：{len(names_o)} 個化合物。"
                              "φ_gas 高者（如 FTOHs）以氣態進行長程傳輸並沿途氧化生成 PFCA；"
                              "φ_particle 高者易隨微粒乾濕沉降。可調整 fom／TSP 做情境比較。")


SPEC = MethodSpec(
    key="partitioning",
    name="氣粒分配（KOA 模型）",
    summary="用 log KOA 與 KOA 吸收模型算各化合物的氣相/粒相比例，判斷以氣態或粒態進行長程傳輸。",
    params=[
        ParamSpec("id_col", "化合物名稱欄（可空）", "column", default="compound", optional=True),
        ParamSpec("koa_col", "log KOA 欄", "column", default="log_KOA",
                  help="辛醇-空氣分配係數的 log10 值（同溫度下）。"),
        ParamSpec("fom", "顆粒有機質分率 fom", "float", default=0.1, minimum=0.001, maximum=1.0,
                  help="TSP 中有機質質量分數，文獻常用 0.1。"),
        ParamSpec("tsp", "總懸浮微粒 TSP (µg/m³)", "float", default=30.0, minimum=0.1,
                  help="當地大氣顆粒物濃度，影響粒相比例。"),
    ],
    schema=InputSchema(min_rows=1, min_numeric_cols=0, id_col_param="id_col",
                       required_param_cols=["koa_col"],
                       note="每列一個化合物；需提供 log KOA 欄。"),
    template_columns=["compound", "log_KOA"],
    uses_colors=["primary", "accent"],
)
SPEC.run = run
SPEC.make_demo = make_demo
