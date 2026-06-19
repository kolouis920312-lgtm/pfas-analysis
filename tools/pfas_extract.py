# -*- coding: utf-8 -*-
"""
pfas_extract.py — 從彙整資料庫抽出「一大類 × 指定相態 × 指定/核心化合物」的子集，
                  直接輸出成可丟進 pfas_toolkit（HCA / CoDA / PCA …）的 CSV。
================================================================================
資料庫：PFAS_127-250_彙整_v3.xlsx（指紋清單\抽取）
  ‧ paper_meta.manifest_categories：每篇論文的 24 大類標籤（如 "AFFF|URBN|PREC"）
  ‧ records / records_分析數值 / records_百分比：每列一個樣本，欄含中繼資料 + 各 PFAS 化合物
  ‧ 關鍵語意：空白 = 沒測（not measured）；0 = 有測但未檢出（BDL）。兩者不可混為一談。

本腳本嚴格保留此語意：
  ‧ 沒測 → 留成空白（NaN），不補 0、不補中位數
  ‧ BDL → 維持 0
  ‧ 可用「覆蓋率門檻」建立核心化合物盤（core analyte panel），把測得太少的化合物剔除，
    避免把「沒測」誤當「組成差異」（對應方法學筆記第 2、8–12 節）。

用法範例
--------
  # 先看 24 大類各有多少論文/樣本
  python tools/pfas_extract.py --list-categories

  # 抽 URBN（都市背景）的 particle 相，核心盤＝該子集中覆蓋率≥50% 的化合物
  python tools/pfas_extract.py --category URBN --phase particle --core-coverage 0.5

  # 只要指定幾隻化合物（逗號分隔），相態不限
  python tools/pfas_extract.py --category WWTP --compounds "PFBA,PFOA,PFOS,PFHxS,6:2 FTS"

  # 抽完直接列封閉成 100%（compositional），輸出 xlsx 也一份
  python tools/pfas_extract.py --category POLR --phase gas --core-coverage 0.6 --close --xlsx

輸出（預設寫到 --outdir，檔名前綴＝<類別>_<相態>）
  ‧ <prefix>.csv          → sample_id + 化合物欄（可直接上傳網站版 / 餵 toolkit）
  ‧ <prefix>_meta.csv     → sample_id, 論文, 國家, 相態, 年（給 cluster×論文 批次效應診斷用）
  ‧ <prefix>_coverage.csv → 每隻化合物在此子集的測得數與覆蓋率（核心盤依據）
"""
import argparse
import os
import sys
import pandas as pd

DEFAULT_DB = r"C:\Users\user\Desktop\PFAS database\指紋清單\抽取\PFAS_127-250_彙整_v3.xlsx"

# 各「values 來源表」的欄位配置：靠錨點欄找出「化合物欄區段」，對欄名清單變動較穩。
SHEETS = {
    # raw 原始濃度（含 0=BDL、空白=沒測）
    "raw":  dict(sheet="records",        paper_col="論文編號", phase_col="採集的相態",
                 rowid_col="row_id", country_col="國家", year_col="採樣年",
                 stat_col="stat_type", comp_after="濃度單位", comp_before="哪篇文獻"),
    # norm 單位正規化後的濃度（pg/m³），化合物區段乾淨、最推薦
    "norm": dict(sheet="records_分析數值", paper_col="論文編號", phase_col="採集的相態",
                 rowid_col="row_id", country_col="國家", year_col=None,
                 stat_col="stat_type正規化", comp_after="stat_type正規化", comp_before=None),
    # pct 已封閉成百分比
    "pct":  dict(sheet="records_百分比",   paper_col="哪篇文獻", phase_col="採集的相態",
                 rowid_col=None, country_col="國家", year_col=None,
                 stat_col="stat_type", comp_after="可計%樣本", comp_before=None),
}

# 視為「單筆原始樣本」的 stat_type（其餘 mean/median/max/min/SD/range/percentile 為彙總列，分群不可混入）
INDIVIDUAL_STATS = {"raw", "individual", "measured", "sample"}


def load_category_map(db):
    pm = pd.read_excel(db, sheet_name="paper_meta")
    paper2cats, cat2papers = {}, {}
    for _, r in pm.iterrows():
        try:
            paper = int(r["paper"])
        except Exception:
            continue
        cats = [c.strip() for c in str(r.get("manifest_categories") or "").split("|")
                if c.strip() and c.strip().lower() != "nan"]
        paper2cats[paper] = cats
        for c in cats:
            cat2papers.setdefault(c, []).append(paper)
    return pm, cat2papers, paper2cats


def compound_columns(cols, cfg, drop_x_prefix=True):
    """用錨點欄切出化合物欄區段。"""
    cols = list(cols)
    start = cols.index(cfg["comp_after"]) + 1 if cfg["comp_after"] in cols else 0
    end = cols.index(cfg["comp_before"]) if (cfg["comp_before"] in cols) else len(cols)
    comps = cols[start:end]
    if drop_x_prefix:
        comps = [c for c in comps if not str(c).startswith("X:")]
    return comps


def export_denormalized(args, paper2cats):
    """把整個資料庫攤平成一張 CSV（含 category 欄）→ 上傳網站版用『PFAS 指紋 HCA』線上切。"""
    cfg = SHEETS[args.values]
    df = pd.read_excel(args.db, sheet_name=cfg["sheet"])
    comps = compound_columns(df.columns, cfg)
    stat_col = cfg.get("stat_col")
    if stat_col and stat_col in df.columns and args.stat_type.lower() != "all":
        sv = df[stat_col].astype(str).str.strip().str.lower()
        df = (df[sv.isin(INDIVIDUAL_STATS)] if args.stat_type.lower() == "individual"
              else df[sv == args.stat_type.strip().lower()])
    df = df.reset_index(drop=True)

    def to_int(p):
        try:
            return int(float(p))
        except Exception:
            return None
    pap = df[cfg["paper_col"]]
    primary = [(paper2cats.get(to_int(p)) or [""])[0] for p in pap]
    allcats = ["|".join(paper2cats.get(to_int(p), [])) for p in pap]
    rid = (df[cfg["rowid_col"]].astype(str) if cfg["rowid_col"] and cfg["rowid_col"] in df.columns
           else pd.Series([str(i) for i in range(len(df))]))
    out = pd.DataFrame({
        "sample_id": ["r" + s for s in rid],
        "category": primary, "categories_all": allcats,
        "phase": df[cfg["phase_col"]].values if cfg["phase_col"] in df.columns else "",
        "paper": pap.values,
        "country": df[cfg["country_col"]].values if cfg["country_col"] in df.columns else "",
        "stat_type": df[stat_col].values if (stat_col and stat_col in df.columns) else "",
    })
    comp_df = df[comps].apply(pd.to_numeric, errors="coerce")
    comp_df.index = out.index
    out = pd.concat([out, comp_df], axis=1)

    if args.explode_categories:
        pairs = []
        for i in range(len(out)):
            for c in (paper2cats.get(to_int(pap.iloc[i]), []) or [""]):
                pairs.append((i, c))
        out = out.iloc[[i for i, _ in pairs]].reset_index(drop=True)
        out["category"] = [c for _, c in pairs]
        out["sample_id"] = [f"{s}__{c}" for s, c in zip(out["sample_id"], out["category"])]

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, (args.prefix or "denormalized_all") + ".csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    na = int(out[comps].isna().sum().sum())
    print(f"攤平輸出 ✓：{path}")
    print(f"  {len(out)} 列 × {len(comps)} 化合物（沒測空白 {na} 格）"
          + ("；已依類別展開（一列＝一個樣本×類別）" if args.explode_categories
             else "；category＝主類別、categories_all＝全部標籤"))
    print("  → 上傳網站版，用『PFAS 指紋 HCA』：類別欄選 category、勾要保留的類別值即可線上切子集。")


def list_categories(db):
    pm, cat2papers, _ = load_category_map(db)
    rec = pd.read_excel(db, sheet_name="records_分析數值", usecols=["論文編號", "採集的相態"])
    by_paper = rec["論文編號"].value_counts().to_dict()
    print(f"資料庫：{db}")
    print(f"共 {len(cat2papers)} 大類\n")
    print(f"{'類別':8} {'論文數':>6} {'樣本數':>7}")
    print("-" * 26)
    for c in sorted(cat2papers, key=lambda k: -sum(by_paper.get(p, 0) for p in cat2papers[k])):
        npap = len(set(cat2papers[c]))
        nrec = sum(by_paper.get(p, 0) for p in cat2papers[c])
        print(f"{c:8} {npap:>6} {nrec:>7}")


def main():
    # Windows 主控台預設 cp950，印 ≥ ✓ → 等字元會炸；強制 UTF-8 輸出（無法顯示就替換，不中斷）
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="從 PFAS 彙整資料庫抽子集 → toolkit 可用 CSV")
    ap.add_argument("--db", default=DEFAULT_DB, help="彙整資料庫 xlsx 路徑")
    ap.add_argument("--list-categories", action="store_true", help="列出 24 大類與筆數後結束")
    ap.add_argument("--denormalized", action="store_true",
                    help="攤平整庫成一張 CSV（含 category 欄）→ 上傳網站版用『PFAS 指紋 HCA』線上切")
    ap.add_argument("--explode-categories", action="store_true",
                    help="搭配 --denormalized：一列＝一個(樣本×類別)，讓網頁 category 精確篩選涵蓋多重歸類")
    ap.add_argument("--category", help="要抽的大類代碼（如 URBN、WWTP、POLR…）")
    ap.add_argument("--phase", default="all",
                    help="相態子字串過濾：particle / gas / 固態 / 液態 / all（預設 all 不過濾）")
    ap.add_argument("--stat-type", default="individual",
                    help="統計列過濾：individual 只留單筆原始樣本(預設) / all 全留 / 或指定值如 raw、mean")
    ap.add_argument("--values", choices=list(SHEETS), default="norm",
                    help="數值來源：raw 原始濃度 / norm 正規化濃度(pg/m³，預設) / pct 百分比")
    ap.add_argument("--compounds", default="",
                    help="指定化合物（逗號分隔）；給了就只抽這些，覆蓋率門檻會被忽略")
    ap.add_argument("--core-coverage", type=float, default=0.0,
                    help="核心盤門檻 0~1：只留在此子集中覆蓋率≥此值的化合物（如 0.5）")
    ap.add_argument("--min-compounds", type=int, default=3,
                    help="每個樣本在盤上至少要有幾隻『有測』才保留（預設 3）")
    ap.add_argument("--close", action="store_true",
                    help="輸出前把每列在盤上封閉成 100％（compositional）；沒測仍留空白")
    ap.add_argument("--na-policy", choices=["keep", "zero"], default="keep",
                    help="沒測值處理：keep 留空白(預設、正確) / zero 補 0(不建議，會把沒測當未檢出)")
    ap.add_argument("--id-prefix", default="", help="sample_id 前綴（預設用類別代碼）")
    ap.add_argument("--outdir", default=".", help="輸出資料夾（預設目前目錄）")
    ap.add_argument("--prefix", default="", help="輸出檔名前綴（預設 <類別>_<相態>）")
    ap.add_argument("--xlsx", action="store_true", help="同時輸出一份 xlsx")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        sys.exit(f"找不到資料庫：{args.db}")

    if args.list_categories:
        list_categories(args.db)
        return

    pm, cat2papers, paper2cats = load_category_map(args.db)

    if args.denormalized:
        export_denormalized(args, paper2cats)
        return

    if not args.category:
        sys.exit("請用 --category 指定大類（或 --list-categories 先看有哪些；或 --denormalized 攤平整庫）。")

    cfg = SHEETS[args.values]
    if args.category not in cat2papers:
        sys.exit(f"類別「{args.category}」不存在。可用：{', '.join(sorted(cat2papers))}")
    papers = set(cat2papers[args.category])
    print(f"類別 {args.category} → {len(papers)} 篇論文")

    df = pd.read_excel(args.db, sheet_name=cfg["sheet"])
    comps_all = compound_columns(df.columns, cfg)

    # 1) 依論文（類別）過濾
    sub = df[df[cfg["paper_col"]].isin(papers)].copy()
    # 2) 依相態過濾（子字串、忽略大小寫）
    if args.phase and args.phase.lower() != "all":
        ph = sub[cfg["phase_col"]].astype(str)
        sub = sub[ph.str.contains(args.phase, case=False, na=False)]
    # 3) 依 stat_type 過濾（預設只留單筆原始樣本，排除 mean/median 等彙總列）
    stat_col = cfg.get("stat_col")
    if stat_col and stat_col in sub.columns and args.stat_type.lower() != "all":
        sv = sub[stat_col].astype(str).str.strip().str.lower()
        if args.stat_type.lower() == "individual":
            sub = sub[sv.isin(INDIVIDUAL_STATS)]
        else:
            sub = sub[sv == args.stat_type.strip().lower()]
    print(f"類別＋相態＋統計列（{args.stat_type}）過濾後：{len(sub)} 列")
    if len(sub) == 0:
        sys.exit("過濾後沒有資料，請放寬 --phase / --stat-type 或換 --category。")

    # 3) 化合物欄轉數值（空白→NaN＝沒測；0 維持＝BDL）
    for c in comps_all:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    # 4) 決定要輸出的化合物盤
    if args.compounds.strip():
        want = [c.strip() for c in args.compounds.split(",") if c.strip()]
        panel = [c for c in want if c in comps_all]
        miss = [c for c in want if c not in comps_all]
        if miss:
            print(f"⚠ 這些化合物在資料庫找不到，已略過：{miss}")
    else:
        cov = sub[comps_all].notna().mean()          # 每隻化合物在子集的覆蓋率
        panel = [c for c in comps_all if cov[c] >= args.core_coverage and cov[c] > 0]
        if not panel:
            top = cov.sort_values(ascending=False).head(10)
            hint = "；".join(f"{k} {v:.0%}" for k, v in top.items() if v > 0)
            sys.exit(f"覆蓋率≥{args.core_coverage:.0%} 沒有任何化合物。\n"
                     f"此子集覆蓋率最高的化合物：{hint}\n"
                     f"→ 跨研究 PFAS 覆蓋率本來就低，建議 --core-coverage 設 0.2~0.3，"
                     f"或直接用 --compounds 指定你要的幾隻。")
    if not panel:
        sys.exit("化合物盤是空的：請檢查 --compounds 是否拼對。")
    print(f"化合物盤：{len(panel)} 隻"
          + (f"（覆蓋率≥{args.core_coverage:.0%}）" if not args.compounds.strip() else "（指定）"))

    sub = sub.reset_index(drop=True)

    # 5) 樣本 ID（盡量穩定唯一）
    pre = args.id_prefix or args.category
    if cfg["rowid_col"] and cfg["rowid_col"] in sub.columns:
        sid = sub[cfg["rowid_col"]].astype(str).map(lambda s: f"{pre}_r{s}")
    else:
        sid = [f"{pre}_{int(p)}_{i}" for i, p in enumerate(sub[cfg["paper_col"]])]
    sub.insert(0, "sample_id", list(sid))

    # 6) 丟掉「在盤上有測太少」或「盤上總和為 0」的樣本
    measured = sub[panel].notna().sum(axis=1)
    psum = sub[panel].sum(axis=1, skipna=True)
    keep = (measured >= args.min_compounds) & (psum > 0)
    dropped = int((~keep).sum())
    sub = sub[keep].reset_index(drop=True)
    print(f"丟棄盤上測得<{args.min_compounds} 隻或總和=0 的樣本：{dropped} 列 → 剩 {len(sub)} 列")
    if len(sub) == 0:
        sys.exit("樣本都被門檻濾掉了，請調低 --min-compounds 或 --core-coverage。")

    data = sub[["sample_id"] + panel].copy()

    # 7) 選擇性封閉成 100%（沒測 NaN 不計入分母，仍保持 NaN）
    if args.close:
        vals = data[panel]
        rowsum = vals.sum(axis=1, skipna=True).replace(0, pd.NA)
        data[panel] = vals.div(rowsum, axis=0) * 100.0

    # 8) 沒測值政策
    if args.na_policy == "zero":
        print("⚠ --na-policy zero：把『沒測』補成 0，等於把未測量當未檢出，僅在你確定時使用。")
        data[panel] = data[panel].fillna(0.0)

    # ── 輸出 ──────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)
    prefix = args.prefix or f"{args.category}_{args.phase}"
    main_csv = os.path.join(args.outdir, f"{prefix}.csv")
    data.to_csv(main_csv, index=False, encoding="utf-8-sig")

    meta_cols = {"sample_id": data["sample_id"]}
    meta_cols["paper"] = sub[cfg["paper_col"]].values
    meta_cols["country"] = sub[cfg["country_col"]].values if cfg["country_col"] in sub else ""
    meta_cols["phase"] = sub[cfg["phase_col"]].values if cfg["phase_col"] in sub else ""
    if cfg["year_col"] and cfg["year_col"] in sub:
        meta_cols["year"] = sub[cfg["year_col"]].values
    meta = pd.DataFrame(meta_cols)
    meta_csv = os.path.join(args.outdir, f"{prefix}_meta.csv")
    meta.to_csv(meta_csv, index=False, encoding="utf-8-sig")

    cov = sub[panel].notna()
    cov_df = pd.DataFrame({"compound": panel,
                           "n_measured": cov.sum().values,
                           "coverage_pct": (cov.mean().values * 100).round(1)})
    cov_df = cov_df.sort_values("coverage_pct", ascending=False)
    cov_csv = os.path.join(args.outdir, f"{prefix}_coverage.csv")
    cov_df.to_csv(cov_csv, index=False, encoding="utf-8-sig")

    if args.xlsx:
        xlsx_path = os.path.join(args.outdir, f"{prefix}.xlsx")
        with pd.ExcelWriter(xlsx_path) as xw:
            data.to_excel(xw, sheet_name="data", index=False)
            meta.to_excel(xw, sheet_name="meta", index=False)
            cov_df.to_excel(xw, sheet_name="coverage", index=False)

    na_total = int(data[panel].isna().sum().sum())
    print("\n完成 ✓")
    print(f"  主檔（餵程式）：{main_csv}  （{len(data)} 列 × {len(panel)} 化合物，"
          f"沒測空白 {na_total} 格）")
    print(f"  中繼（診斷用）：{meta_csv}")
    print(f"  覆蓋率報告　 ：{cov_csv}")
    if args.xlsx:
        print(f"  xlsx 一份　 ：{xlsx_path}")


if __name__ == "__main__":
    main()
