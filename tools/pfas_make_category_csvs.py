# -*- coding: utf-8 -*-
"""
pfas_make_category_csvs.py — 為 24 大類各輸出一份「自含中繼欄」的寬表 CSV，
                            可直接丟進 pfas_toolkit 網站版的 HCA 上傳。
================================================================================
每檔欄位：sample_id, category, categories_all, phase, paper, country [, year]
          + 該類別實際量測過的各 PFAS 化合物欄。

嚴格保留資料庫語意：
  ‧ 空白 = 沒測（not measured）→ 留 NaN，不補 0、不補中位數
  ‧ 0    = 有測但未檢出（BDL）→ 維持 0
數值來源：records_分析數值（單位正規化 pg/m³，化合物欄區段最乾淨）。
只取「樣本級」列（stat_type=raw/individual…），排除 mean/median 等彙總列。

相態策略：一個類別＝一個檔，**所有相態（氣/粒/固/液/雪…）放同一檔**，另含 phase 欄，
          上傳後可在網站用『類別欄＝phase』線上篩單一相態再分群（避免跨相態混分）。

類別來源：paper_meta.manifest_categories（與 pfas_extract.py 一致的 24 大類）。
          多重歸類論文的樣本會出現在多個類別檔，categories_all 欄保留全部標籤（透明）。

不動原始庫；只讀 xlsx，輸出到 --outdir。
用法：python tools/pfas_make_category_csvs.py --outdir "<資料夾>"
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfas_extract import (DEFAULT_DB, SHEETS, INDIVIDUAL_STATS,
                          load_category_map, compound_columns, clean_int_col)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="24 大類各輸出一份網站可上傳的寬表 CSV")
    ap.add_argument("--db", default=DEFAULT_DB, help="彙整資料庫 xlsx")
    ap.add_argument("--outdir", default="範例資料集_24類", help="輸出資料夾")
    ap.add_argument("--values", choices=list(SHEETS), default="norm",
                    help="數值來源：norm 正規化 pg/m³（預設）/ raw / pct")
    ap.add_argument("--min-compounds", type=int, default=3,
                    help="每個樣本至少要有幾隻『有測』才保留（預設 3）")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        sys.exit(f"找不到資料庫：{args.db}")

    cfg = SHEETS[args.values]
    pm, cat2papers, paper2cats = load_category_map(args.db)
    df = pd.read_excel(args.db, sheet_name=cfg["sheet"])
    comps = compound_columns(df.columns, cfg)

    # 樣本級過濾（排除 mean/median 等彙總列）
    stat_col = cfg.get("stat_col")
    if stat_col and stat_col in df.columns:
        sv = df[stat_col].astype(str).str.strip().str.lower()
        df = df[sv.isin(INDIVIDUAL_STATS)]
    df = df.reset_index(drop=True)

    # 化合物轉數值（空白→NaN＝沒測；0 維持＝BDL）
    for c in comps:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def to_int(p):
        try:
            return int(float(p))
        except Exception:
            return None

    pap = df[cfg["paper_col"]]
    paper_int = pap.map(to_int)
    allcats = paper_int.map(lambda p: "|".join(paper2cats.get(p, [])))
    if cfg["rowid_col"] and cfg["rowid_col"] in df.columns:
        rid = df[cfg["rowid_col"]].astype(str)
    else:
        rid = pd.Series([str(i) for i in range(len(df))])

    os.makedirs(args.outdir, exist_ok=True)
    index_rows = []
    print(f"[載入] {args.db}\n  樣本級列 {len(df)}；化合物欄 {len(comps)}\n")

    for cat in sorted(cat2papers):
        papers = set(cat2papers[cat])
        mask = paper_int.isin(papers)
        sub = df[mask]
        comp_sub = sub[comps]
        # 丟在盤上有測太少的樣本
        measured = comp_sub.notna().sum(axis=1)
        keep = measured >= args.min_compounds
        sub = sub[keep]
        comp_sub = comp_sub[keep]
        if len(sub) == 0:
            print(f"  {cat:6} —— 0 列（樣本量測<{args.min_compounds} 隻），略過")
            index_rows.append({"category": cat, "rows": 0, "compounds": 0,
                               "papers": 0, "phases": 0, "note": "樣本不足"})
            continue
        # 只留此類別至少被測過一次的化合物（去掉整欄全 NaN）
        nonempty = [c for c in comps if comp_sub[c].notna().any()]

        meta = {
            "sample_id": [f"{cat}_r{r}" for r in rid[sub.index]],
            "category": cat,
            "categories_all": allcats[sub.index].values,
            "phase": sub[cfg["phase_col"]].astype(str).values if cfg["phase_col"] in sub else "",
            "paper": sub[cfg["paper_col"]].values,
            "country": sub[cfg["country_col"]].values if cfg["country_col"] in sub else "",
        }
        if cfg.get("year_col") and cfg["year_col"] in sub:
            meta["year"] = sub[cfg["year_col"]].values
        out = pd.concat([pd.DataFrame(meta).reset_index(drop=True),
                         comp_sub[nonempty].reset_index(drop=True)], axis=1)
        out["paper"] = clean_int_col(out["paper"])          # 論文編號轉整數，避免 190.0
        if "year" in out.columns:
            out["year"] = clean_int_col(out["year"])

        path = os.path.join(args.outdir, f"{cat}.csv")
        out.to_csv(path, index=False, encoding="utf-8-sig")
        nphase = sub[cfg["phase_col"]].astype(str).nunique() if cfg["phase_col"] in sub else 0
        npap = sub[cfg["paper_col"]].nunique()
        na = int(comp_sub[nonempty].isna().sum().sum())
        index_rows.append({"category": cat, "rows": len(out), "compounds": len(nonempty),
                           "papers": npap, "phases": nphase, "note": ""})
        print(f"  {cat:6} {len(out):>4} 列 × {len(nonempty):>3} 化合物 | "
              f"{npap} 篇 | {nphase} 相態 | 沒測空白 {na} 格 → {cat}.csv")

    idx = pd.DataFrame(index_rows).sort_values("rows", ascending=False)
    idx_path = os.path.join(args.outdir, "_索引.csv")
    idx.to_csv(idx_path, index=False, encoding="utf-8-sig")
    print(f"\n[完成] {len([r for r in index_rows if r['rows'] > 0])} 個類別有輸出 → {args.outdir}")
    print(f"       索引：{idx_path}")


if __name__ == "__main__":
    main()
