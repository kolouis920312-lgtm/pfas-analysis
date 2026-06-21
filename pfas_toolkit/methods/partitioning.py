# -*- coding: utf-8 -*-
"""
partitioning.py — 氣粒分配（多模型：KOA / Junge–Pankow / 雙模型 / 穩態 / 離子鏈長式 / pp-LFER）
================================================================================
決定 PFAS 在大氣中以「氣相」或「粒相」存在，這直接左右長程傳輸方式：偏氣相者（如 FTOHs）
以氣態飄送遠距並沿途氧化生成 PFCA；偏粒相者易隨微粒沉降。

文獻共識：沒有單一公式通用於所有化合物，且**離子型 PFAA（PFCA/PFSA）不適用 KOA 吸收模型**。
本方法因此整合六種模型，並可「自動比較」、用**雙軌建議**（中性走 KOA、離子走鏈長式）給最合理估計：

  (A) KOA 吸收（Harner & Bidleman 1998）  lg Kp = lg KOA + lg fom − 11.91         Kp[m³/µg]
  (B) Junge–Pankow 吸附（Pankow 1987）     φ = cθ /(p_L° + cθ)                      用蒸氣壓 p_L°[Pa]
  (C) 雙模型（Dachs–Eisenreich 2000）       Kp = 吸收項 + fEC·(aEC/aAC)·Ksa·10⁻¹²   Kp[m³/µg]
  (D) 穩態 L–M–Y（Li–Ma–Yang）             lg Kp_S = lg Kp_E + lg α（含沉降損失）    Kp[m³/µg]
  (E) 離子型 PFAA 鏈長式（Yamazaki 2021）   lg Kp = 0.38·Cn−1.49(PFSA) / 0.2·Cn−2.35(PFCA)  Kp[m³/mg]
  (F) pp-LFER（Arp；Okeme 2018）           lg Kp = 1.01S+3.17A+0.30B+0.78L+0.51V−7.42       Kp[m³/g]
  (G) ML 參考（本專案內建）                  階層貝氏＋設限感知，38 支 PFAS 的 φ@25°C＋89% 可信區間（已驗證基準）

三式 Kp 單位不同（µg / mg / g），φ = Kp·TSP/(1+Kp·TSP) 時 TSP 會各自換算。
KOA 可隨溫度校正：lg KOA(T) = A + B/T（提供 A、B 欄時）。fom／TSP／θ 由「地區」預設帶入
（都市/背景/偏遠＝Okeme 2018 / Lohmann–Lammel 2004），亦可自訂。

輸出：gas_particle_partitioning.csv（各模型 Kp、φ＋雙軌建議相態）＋ 建議相態堆疊圖 ＋ 模型比較熱圖。
"""
import os
import numpy as np
import pandas as pd

from ..core.spec import MethodSpec, ParamSpec, InputSchema
from ..core.theme import get_plt


# 地區預設：fom（有機質分率）、TSP（µg/m³）、θ（cm²/cm³, J–P 用）
# 來源：Okeme et al. 2018 / Lohmann & Lammel 2004（θ 1.1e-5/1e-6/1e-7 對應 TSP 55/14/7.7 µg/m³）
REGION_PRESETS = {
    "urban":      dict(fom=0.40, tsp=55.0, theta=1.1e-5),
    "background": dict(fom=0.19, tsp=14.0, theta=1.0e-6),
    "remote":     dict(fom=0.08, tsp=7.7,  theta=1.0e-7),
}

MODEL_AUTO = "自動比較（依欄位盡量全跑）"
MODEL_KOA = "KOA 吸收（Harner–Bidleman）"
MODEL_JP = "Junge–Pankow 吸附"
MODEL_DUAL = "雙模型（KOA＋黑碳吸附 Dachs–Eisenreich）"
MODEL_STEADY = "穩態 L–M–Y（非平衡）"
MODEL_IONIC = "離子型 PFAA 鏈長式（Yamazaki）"
MODEL_LFER = "pp-LFER（多參數）"
MODEL_ML = "ML 參考（內建貝氏，38 支 PFAS）"


_ML_REF_CACHE = None


def _load_ml_reference():
    """載入內建 ML 參考表（本專案階層貝氏、設限感知模型輸出；38 支 PFAS，含 89% 可信區間）。
    來源：PFAS氣粒分配_彙整表.xlsx → pfas_toolkit/data/ml_phi_reference.csv。讀不到則回空表（其餘模型照常）。"""
    global _ML_REF_CACHE
    if _ML_REF_CACHE is None:
        fp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "ml_phi_reference.csv")
        try:
            r = pd.read_csv(fp)
            r["_key"] = r["compound"].astype(str).str.strip().str.upper()
            _ML_REF_CACHE = r
        except Exception:
            _ML_REF_CACHE = pd.DataFrame()
    return _ML_REF_CACHE


def make_demo(seed=5):
    """代表性化合物：中性前驅物（走 KOA/J–P/雙模型/穩態）＋ 離子型 PFAA（走鏈長式）。
    欄位 class＝中性/PFCA/PFSA；Cn＝總碳數（離子用）；log_KOA（中性用）；pL_Pa＝次冷液蒸氣壓（J–P 用）。"""
    data = [
        # name,            class,     Cn,  log_KOA, pL_Pa
        ("4:2 FTOH",       "neutral",  np.nan, 5.6, 8.0),
        ("6:2 FTOH",       "neutral",  np.nan, 6.4, 3.0),
        ("8:2 FTOH",       "neutral",  np.nan, 7.2, 2.5e-1),
        ("10:2 FTOH",      "neutral",  np.nan, 8.0, 1.2e-2),
        ("FOSA",           "neutral",  np.nan, 8.3, 5.0e-3),
        ("MeFOSE",         "neutral",  np.nan, 8.9, 1.5e-3),
        ("EtFOSE",         "neutral",  np.nan, 9.2, 7.0e-4),
        ("PFBA",           "PFCA",     4,      np.nan, np.nan),
        ("PFHxA",          "PFCA",     6,      np.nan, np.nan),
        ("PFOA",           "PFCA",     8,      np.nan, np.nan),
        ("PFNA",           "PFCA",     9,      np.nan, np.nan),
        ("PFBS",           "PFSA",     4,      np.nan, np.nan),
        ("PFHxS",          "PFSA",     6,      np.nan, np.nan),
        ("PFOS",           "PFSA",     8,      np.nan, np.nan),
    ]
    out = pd.DataFrame(data, columns=["compound", "class", "Cn", "log_KOA", "pL_Pa"])
    out["Cn"] = out["Cn"].astype("Int64")   # 可空整數：CSV 顯示 4 而非 4.0，中性留空白
    return out


# ── 小工具 ────────────────────────────────────────────────────────────────
def _opt_col(params, key, df):
    """取『欄位型』參數：選了且該欄存在才回欄名，否則 None（容忍預設欄名在使用者資料不存在）。"""
    c = params.get(key) or ""
    c = str(c).strip()
    if c in ("", "(無)", "None"):
        return None
    return c if c in df.columns else None


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce").values if col else None


def _fnum(params, key, default):
    """數值參數：只有在缺值/空字串時才用 default（保留 0 這個有效值，如 0°C、fEC=0）。"""
    v = params.get(key, default)
    return float(default) if v in (None, "") else float(v)


def _phi(kp, tsp):
    """φ = Kp·TSP /(1 + Kp·TSP)；kp 與 tsp 須為相容單位（乘積無因次）。"""
    with np.errstate(invalid="ignore"):
        x = kp * tsp
        return x / (1.0 + x)


def run(df, params, ctx):
    plt = get_plt(ctx.theme)
    n = len(df)

    # 名稱
    id_col = _opt_col(params, "id_col", df)
    names = (df[id_col].astype(str).values if id_col
             else np.array([f"C{i+1}" for i in range(n)]))

    # 地區預設 + 覆寫
    region_raw = str(params.get("region", "都市 urban"))
    rkey = ("urban" if "urban" in region_raw or "都市" in region_raw else
            "background" if "background" in region_raw or "背景" in region_raw else
            "remote" if "remote" in region_raw or "偏遠" in region_raw else "custom")
    preset = REGION_PRESETS.get(rkey, REGION_PRESETS["urban"])
    fom = float(params.get("fom", 0) or 0) or preset["fom"]
    tsp = float(params.get("tsp", 0) or 0) or preset["tsp"]
    theta = float(params.get("theta", 0) or 0) or preset["theta"]
    if fom <= 0 or fom > 1:
        raise ValueError("fom（顆粒有機質分率）需介於 0~1。")
    if tsp <= 0:
        raise ValueError("TSP（總懸浮微粒）需 > 0 µg/m³。")
    region_label = ({"urban": "都市", "background": "背景", "remote": "偏遠"}.get(rkey, "自訂"))
    ctx.log(f"地區={region_label}：fom={fom}, TSP={tsp} µg/m³, θ={theta:g} cm²/cm³（可在參數覆寫）")

    # KOA（含溫度校正 lg KOA = A + B/T）
    temp_c = _fnum(params, "temp_c", 25)
    T = temp_c + 273.15
    koa_col = _opt_col(params, "koa_col", df)
    a_col = _opt_col(params, "koa_a_col", df)
    b_col = _opt_col(params, "koa_b_col", df)
    if a_col and b_col:
        logkoa = _num(df, a_col) + _num(df, b_col) / T
        ctx.log(f"KOA 溫度校正：lg KOA = A + B/T，T={temp_c}°C（{T:.2f} K）")
    elif koa_col:
        logkoa = _num(df, koa_col)
        if abs(temp_c - 25) > 0.5:
            ctx.log(f"⚠ 未提供 KOA 的 A/B 欄 → 無法做溫度校正，直接採用 log KOA 欄（視為 {temp_c}°C 之值）。")
    else:
        logkoa = np.full(n, np.nan)
    has_koa = np.isfinite(logkoa).any()

    # 其他輸入
    vp = _num(df, _opt_col(params, "vp_col", df))            # 次冷液蒸氣壓 p_L° (Pa)，J–P
    cn = _num(df, _opt_col(params, "cn_col", df))            # 總碳數，離子鏈長式
    cls_col = _opt_col(params, "class_col", df)
    cls = (df[cls_col].astype(str).str.upper().values if cls_col else None)
    logksa = _num(df, _opt_col(params, "logksa_col", df))    # log Ksa（雙模型；無則以 KOA 代理）
    # pp-LFER Abraham 描述符
    S = _num(df, _opt_col(params, "lfer_s_col", df)); A = _num(df, _opt_col(params, "lfer_a_col", df))
    B = _num(df, _opt_col(params, "lfer_b_col", df)); L = _num(df, _opt_col(params, "lfer_l_col", df))
    V = _num(df, _opt_col(params, "lfer_v_col", df))
    has_lfer = all(x is not None for x in (S, A, B, L, V))

    jp_c = _fnum(params, "jp_c", 17.2)
    fec = _fnum(params, "fec", 0.1)
    lmy_c = _fnum(params, "lmy_c", 5.0)

    # 要跑哪些模型
    want = str(params.get("model", MODEL_AUTO))
    auto = (want == MODEL_AUTO)
    ml_ref = _load_ml_reference()
    run_ml = (auto or want == MODEL_ML) and len(ml_ref) > 0
    run_koa = (auto or want == MODEL_KOA) and has_koa
    run_jp = (auto or want == MODEL_JP) and (vp is not None and np.isfinite(vp).any())
    run_dual = (auto or want == MODEL_DUAL) and has_koa
    run_steady = (auto or want == MODEL_STEADY) and has_koa
    run_ionic = (auto or want == MODEL_IONIC) and (cn is not None and cls is not None)
    run_lfer = (auto or want == MODEL_LFER) and has_lfer
    if not any([run_ml, run_koa, run_jp, run_dual, run_steady, run_ionic, run_lfer]):
        raise ValueError("沒有任何模型可執行。請至少提供：log KOA 欄（KOA/雙模型/穩態），"
                         "或蒸氣壓 p_L° 欄（Junge–Pankow），或 class＋Cn 欄（離子鏈長式），"
                         "或 5 個 Abraham 描述符欄（pp-LFER）；或用『ML 參考』比對內建 38 支 PFAS。")

    out = pd.DataFrame({"compound": names})
    if cls_col:
        out["class"] = df[cls_col].astype(str).values
    cols_for_heatmap = {}   # 模型名 -> φ_particle 陣列

    # (G) ML 參考（本專案內建貝氏；已驗證基準，置於最前作為比較基準）
    if run_ml:
        m = ml_ref.set_index("_key")
        key = pd.Series(names).astype(str).str.strip().str.upper()
        phi = key.map(m["phi_25C"]).to_numpy(dtype=float)
        nmatch = int(np.isfinite(phi).sum())
        if nmatch == 0 and want == MODEL_ML:
            raise ValueError("『ML 參考』未對到任何化合物名稱（內建 38 支 PFAS，名稱需相符，"
                             "如 PFOA、PFOS、6:2 FTOH）。請確認名稱欄，或改用其他模型。")
        out["ML_phi_particle"] = np.round(phi, 4)
        out["ML_phi_lo89"] = np.round(key.map(m["phi_lo89"]).to_numpy(dtype=float), 4)
        out["ML_phi_hi89"] = np.round(key.map(m["phi_hi89"]).to_numpy(dtype=float), 4)
        if nmatch > 0:
            cols_for_heatmap["ML 參考"] = phi
        ctx.log(f"ML 參考：對到 {nmatch}/{n} 支（內建 38 支 PFAS；階層貝氏＋設限校驗，含 89% 可信區間）")

    # (A) KOA 吸收
    if run_koa:
        log_kp = logkoa + np.log10(fom) - 11.91          # m³/µg
        phi = _phi(10.0 ** log_kp, tsp)
        out["logKOA_used"] = np.round(logkoa, 2)
        out["KOA_logKp"] = np.round(log_kp, 3)
        out["KOA_phi_particle"] = np.round(phi, 4)
        cols_for_heatmap["KOA"] = phi

    # (B) Junge–Pankow（用蒸氣壓）
    if run_jp:
        phi = (jp_c * theta) / (vp + jp_c * theta)       # 無因次
        out["JP_phi_particle"] = np.round(phi, 4)
        cols_for_heatmap["Junge–Pankow"] = phi

    # (C) 雙模型（吸收 + 黑碳吸附）
    if run_dual:
        absb = 10.0 ** (logkoa + np.log10(fom) - 11.91)  # 吸收項 m³/µg
        lksa = logksa if (logksa is not None and np.isfinite(logksa).any()) else logkoa
        if logksa is None or not np.isfinite(logksa).any():
            ctx.log("⚠ 雙模型未提供 log Ksa 欄 → 以 log Ksa ≈ log KOA 代理（粗略，僅示意黑碳吸附量級）。")
        soot = fec * (10.0 ** lksa) * 1e-12              # 黑碳吸附項 m³/µg（aEC/aAC=1）
        kp = absb + soot
        phi = _phi(kp, tsp)
        out["Dual_logKp"] = np.round(np.log10(kp), 3)
        out["Dual_phi_particle"] = np.round(phi, 4)
        cols_for_heatmap["雙模型"] = phi

    # (D) 穩態 L–M–Y
    if run_steady:
        log_kpe = logkoa + np.log10(fom) - 11.91
        with np.errstate(invalid="ignore"):
            log_alpha = -np.log10(1.0 + (2.09e-6 / fom) * (10.0 ** logkoa) * lmy_c)
        log_kps = log_kpe + log_alpha
        phi = _phi(10.0 ** log_kps, tsp)
        out["Steady_logKp"] = np.round(log_kps, 3)
        out["Steady_phi_particle"] = np.round(phi, 4)
        cols_for_heatmap["穩態L-M-Y"] = phi

    # (E) 離子型 PFAA 鏈長式（Kp m³/mg → TSP 換成 mg/m³）
    if run_ionic:
        log_kp = np.full(n, np.nan)
        is_pfsa = np.array([("PFSA" in c or "SULFON" in c) for c in cls])
        is_pfca = np.array([("PFCA" in c or "CARBOX" in c) for c in cls])
        with np.errstate(invalid="ignore"):
            log_kp = np.where(is_pfsa, 0.38 * cn - 1.49,
                     np.where(is_pfca, 0.20 * cn - 2.35, np.nan))
        phi = _phi(10.0 ** log_kp, tsp / 1e3)            # TSP µg/m³ → mg/m³
        out["Ionic_logKp_m3mg"] = np.round(log_kp, 3)
        out["Ionic_phi_particle"] = np.round(phi, 4)
        cols_for_heatmap["離子鏈長式"] = phi
        if is_pfca.any():
            ctx.log("⚠ Yamazaki 鏈長式對長鏈 PFCA 會嚴重低估粒相（本專案 ML 與設限實測顯示 "
                    "C≥10 之 PFCA φ≈0.7–0.9）→ PFCA 建議以『ML 參考』為準（已列為雙軌建議最優先）。")

    # (F) pp-LFER（Kp m³/g → TSP 換成 g/m³）
    if run_lfer:
        log_kp = 1.01 * S + 3.17 * A + 0.30 * B + 0.78 * L + 0.51 * V - 7.42
        phi = _phi(10.0 ** log_kp, tsp / 1e6)            # TSP µg/m³ → g/m³
        out["LFER_logKp_m3g"] = np.round(log_kp, 3)
        out["LFER_phi_particle"] = np.round(phi, 4)
        cols_for_heatmap["pp-LFER"] = phi

    # ── 雙軌建議：有 ML 參考優先（已驗證）；否則離子走鏈長式、中性走 KOA；再退而求其次 ──
    rec_phi = np.full(n, np.nan)
    rec_model = np.array(["—"] * n, dtype=object)
    pri = [("ML 參考", "ML_phi_particle"),
           ("離子鏈長式", "Ionic_phi_particle"), ("KOA", "KOA_phi_particle"),
           ("雙模型", "Dual_phi_particle"), ("穩態L-M-Y", "Steady_phi_particle"),
           ("Junge–Pankow", "JP_phi_particle"), ("pp-LFER", "LFER_phi_particle")]
    for label, col in pri:
        if col in out.columns:
            v = out[col].values.astype(float)
            take = np.isnan(rec_phi) & np.isfinite(v)
            rec_phi[take] = v[take]
            rec_model[take] = label
    out["phi_particle_best"] = np.round(rec_phi, 4)
    out["phi_gas_best"] = np.round(1.0 - rec_phi, 4)
    out["model_best"] = rec_model
    out["dominant_phase"] = np.where(np.isnan(rec_phi), "—",
                              np.where(rec_phi >= 0.5, "粒相 particle", "氣相 gas"))
    # 建議值若採 ML 參考，附上 89% 可信區間
    if "ML_phi_particle" in out.columns:
        is_ml = (rec_model == "ML 參考")
        out["phi_best_lo89"] = np.where(is_ml, out["ML_phi_lo89"].to_numpy(dtype=float), np.nan).round(4)
        out["phi_best_hi89"] = np.where(is_ml, out["ML_phi_hi89"].to_numpy(dtype=float), np.nan).round(4)

    ran = list(cols_for_heatmap.keys())
    ctx.log(f"執行模型：{', '.join(ran)}")
    valid = np.isfinite(rec_phi)
    ctx.log(f"建議估計：氣相為主 {int((rec_phi[valid] < 0.5).sum())} 個、粒相為主 "
            f"{int((rec_phi[valid] >= 0.5).sum())} 個（共 {int(valid.sum())} 個有結果）。")
    ctx.save_table(out, "gas_particle_partitioning", index=False)

    # ── 圖 1：建議相態堆疊長條（依氣相比例排序）──
    primary = ctx.color("primary", "#4682b4")
    accent = ctx.color("accent", "#ff6347")
    sub = out[valid].copy().sort_values("phi_gas_best", ascending=False)
    if len(sub):
        x = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(sub)), 5))
        ax.bar(x, sub["phi_gas_best"], color=primary, label="氣相 gas")
        ax.bar(x, sub["phi_particle_best"], bottom=sub["phi_gas_best"], color=accent, label="粒相 particle")
        ax.axhline(0.5, ls="--", color="gray", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(sub["compound"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("比例"); ax.set_ylim(0, 1)
        ax.set_title(f"氣粒分配 · 雙軌建議估計（{region_label}：fom={fom}, TSP={tsp} µg/m³）")
        ax.legend()
        fig.tight_layout(); ctx.save_fig(fig, "gas_particle_partitioning")

    # ── 圖 2：模型比較熱圖（φ_particle；看各模型一致/分歧）──
    if len(ran) >= 2:
        M = np.vstack([cols_for_heatmap[k] for k in ran]).T   # rows=化合物, cols=模型
        cmap = ctx.color("cmap_sequential", "viridis")
        fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(ran)), max(4, 0.4 * n)))
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(ran))); ax.set_xticklabels(ran, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
        ax.set_title("各模型 φ_particle 比較（越黃＝越偏粒相）")
        if n <= 30:
            for i in range(n):
                for j in range(len(ran)):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                                fontsize=6, color="white" if M[i, j] < 0.55 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="φ_particle")
        fig.tight_layout(); ctx.save_fig(fig, "partitioning_model_comparison")

    return ctx.result(summary=f"氣粒分配完成（{region_label}地區；模型：{', '.join(ran)}）。"
                              "雙軌建議：有內建 ML 參考（38 支 PFAS，已驗證）者優先採其 φ＋89% 可信區間；"
                              "其餘離子型 PFAA 走 Yamazaki 鏈長式、中性前驅物走 KOA。"
                              "phi_particle_best 為建議值、model_best 標示來源模型。"
                              "注意：Yamazaki 對長鏈 PFCA 會低估粒相，PFCA 請以 ML 參考為準；"
                              "可用模型比較熱圖檢視各模型分歧。")


SPEC = MethodSpec(
    key="partitioning",
    name="氣粒分配（多模型：KOA／J–P／雙模型／穩態／離子鏈長／pp-LFER）",
    summary="整合六種氣粒分配模型並自動比較：KOA 吸收（中性）、Junge–Pankow 吸附（蒸氣壓）、"
            "雙模型（含黑碳吸附）、穩態 L–M–Y（非平衡）、離子型 PFAA 鏈長式（Yamazaki，PFCA/PFSA 用）、"
            "pp-LFER 多參數。fom/TSP/θ 由地區預設帶入、KOA 可隨溫度校正；以『雙軌建議』給最合理相態"
            "（離子走鏈長式、中性走 KOA）。",
    params=[
        ParamSpec("model", "模型（模式切換）", "choice", default=MODEL_AUTO,
                  choices=[MODEL_AUTO, MODEL_ML, MODEL_KOA, MODEL_JP, MODEL_DUAL, MODEL_STEADY, MODEL_IONIC, MODEL_LFER],
                  help="自動比較＝依你提供的欄位盡量全跑＋內建 ML 參考並輸出比較熱圖（建議）。"
                       "ML 參考＝本專案內建 38 支 PFAS 的階層貝氏 φ（已驗證基準，含 89% 可信區間，按化合物名比對）。"
                       "其餘為單一文獻模型；離子型 PFAA 請用 ML 參考/離子鏈長式，勿對離子型硬套 KOA。"),
        ParamSpec("id_col", "化合物名稱欄（可空）", "column", default="compound", optional=True),
        ParamSpec("region", "地區（帶入 fom/TSP/θ 預設）", "choice", default="都市 urban",
                  choices=["都市 urban", "背景 background", "偏遠 remote", "自訂 custom"],
                  help="都市 fom=0.40,TSP=55；背景 0.19,14；偏遠 0.08,7.7（µg/m³；Okeme 2018 / Lohmann–Lammel）。"
                       "下面 fom/TSP/θ 填 >0 即覆寫此預設。"),
        # ── KOA 系列（KOA／雙模型／穩態）──
        ParamSpec("koa_col", "log KOA 欄（中性用）", "column", default="log_KOA", optional=True,
                  help="辛醇–空氣分配係數 log10 值（同溫度）。KOA／雙模型／穩態 的核心輸入。"),
        ParamSpec("fom", "顆粒有機質分率 fom（0＝用地區預設）", "float", default=0.0, minimum=0.0, maximum=1.0,
                  help="顆粒中有機質質量分率。0＝沿用地區預設；填 0.01~1 覆寫。調大→粒相比例上升。"),
        ParamSpec("tsp", "總懸浮微粒 TSP µg/m³（0＝用地區預設）", "float", default=0.0, minimum=0.0,
                  help="當地顆粒物濃度。0＝沿用地區預設；填 >0 覆寫。調大→粒相比例上升。"),
        ParamSpec("temp_c", "溫度 °C（KOA 溫度校正用）", "float", default=25.0,
                  help="配合下面 A、B 欄做 lg KOA = A + B/T 的溫度校正；無 A/B 欄則僅作記錄。"),
        ParamSpec("koa_a_col", "KOA 溫度式 A 欄（可空）", "column", default="", optional=True,
                  help="lg KOA = A + B/T 的截距 A。與 B 欄一起提供時，改用此式（覆寫 log KOA 欄）。"),
        ParamSpec("koa_b_col", "KOA 溫度式 B 欄（可空）", "column", default="", optional=True,
                  help="lg KOA = A + B/T 的斜率 B（K）。需與 A 欄成對提供。"),
        # ── Junge–Pankow ──
        ParamSpec("vp_col", "次冷液蒸氣壓 p_L° 欄（Pa；J–P 用）", "column", default="pL_Pa", optional=True,
                  help="Junge–Pankow 吸附模型的輸入（同溫度的 subcooled liquid vapor pressure, Pa）。"),
        ParamSpec("jp_c", "Junge 常數 c（Pa·cm）", "float", default=17.2, minimum=0.0,
                  help="J–P 經驗常數，文獻常用 17.2 Pa·cm。"),
        ParamSpec("theta", "氣溶膠表面積 θ cm²/cm³（0＝用地區預設）", "float", default=0.0, minimum=0.0,
                  help="J–P 用的單位空氣顆粒表面積。0＝沿用地區預設（都市1.1e-5/背景1e-6/偏遠1e-7）。"),
        # ── 雙模型 ──
        ParamSpec("fec", "元素碳分率 fEC（雙模型）", "float", default=0.1, minimum=0.0, maximum=1.0,
                  help="顆粒中黑碳/元素碳質量分率，文獻常用 0.1。模型對 fEC 比 fom 更敏感。"),
        ParamSpec("logksa_col", "log Ksa 欄（雙模型；可空）", "column", default="", optional=True,
                  help="煤煙–空氣分配係數 log10。留空則以 log Ksa≈log KOA 粗略代理（僅示意量級）。"),
        # ── 離子型 PFAA 鏈長式 ──
        ParamSpec("class_col", "類別欄 class（中性/PFCA/PFSA）", "column", default="class", optional=True,
                  help="標示每個化合物為 neutral/PFCA/PFSA。離子鏈長式據此選 PFSA 或 PFCA 回歸；"
                       "也用於『雙軌建議』判定離子型。"),
        ParamSpec("cn_col", "碳鏈長 Cn 欄（總碳數；離子用）", "column", default="Cn", optional=True,
                  help="Yamazaki 鏈長式輸入：lg Kp=0.38·Cn−1.49(PFSA)／0.2·Cn−2.35(PFCA)，Kp[m³/mg]。"),
        # ── pp-LFER ──
        ParamSpec("lfer_s_col", "pp-LFER：S 欄（可空）", "column", default="", optional=True,
                  help="Abraham 描述符 S（極性/極化率）。S/A/B/L/V 五欄齊全才會跑 pp-LFER。"),
        ParamSpec("lfer_a_col", "pp-LFER：A 欄（可空）", "column", default="", optional=True,
                  help="Abraham A（氫鍵酸度/電子供體）。"),
        ParamSpec("lfer_b_col", "pp-LFER：B 欄（可空）", "column", default="", optional=True,
                  help="Abraham B（氫鍵鹼度/電子受體）。"),
        ParamSpec("lfer_l_col", "pp-LFER：L 欄（可空）", "column", default="", optional=True,
                  help="Abraham L（氣–十六烷分配，凡得瓦力）。"),
        ParamSpec("lfer_v_col", "pp-LFER：V 欄（可空）", "column", default="", optional=True,
                  help="McGowan 分子體積 V。"),
        ParamSpec("lmy_c", "穩態 L–M–Y 調整係數 C", "float", default=5.0, minimum=0.0,
                  help="補償氣–粒質傳的調整係數，多數 SVOC 取 5。"),
    ],
    schema=InputSchema(min_rows=1, min_numeric_cols=0, id_col_param="id_col",
                       note="每列一個化合物。中性物提供 log KOA（或 A/B 溫度式）與/或蒸氣壓 p_L°；"
                            "離子型 PFAA 提供 class（PFCA/PFSA）＋ Cn（總碳數）。至少要能跑一個模型。"),
    template_columns=["compound", "class", "Cn", "log_KOA", "pL_Pa"],
    uses_colors=["primary", "accent", "cmap_sequential"],
)
SPEC.run = run
SPEC.make_demo = make_demo
