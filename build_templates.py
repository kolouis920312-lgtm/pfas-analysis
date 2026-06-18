# -*- coding: utf-8 -*-
"""
build_templates.py — 產生每個方法的範例 CSV 到 templates/
用法： python build_templates.py
每個範本用該方法的示範資料前幾列，欄位與示範值都正確，替換成你的資料即可。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pfas_toolkit import methods

TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def main():
    os.makedirs(TPL, exist_ok=True)
    lines = ["# PFAS 分析範本說明", "",
             "每個方法一個範例 CSV，欄位與示範值都已填好；把資料換成你自己的即可。",
             "BDL（未檢出）一律用 0 或留空白，不要寫 ND / <MDL 等文字。", ""]
    for spec in methods.all_specs():
        df = spec.make_demo().head(8)
        path = os.path.join(TPL, f"{spec.key}_template.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        lines += [f"## {spec.name}  ({spec.key}_template.csv)", "", spec.summary, "",
                  f"- 欄位：{', '.join(map(str, df.columns))}",
                  f"- 最少樣本 {spec.schema.min_rows}；最少數值特徵 {spec.schema.min_numeric_cols}"]
        if spec.schema.required_param_cols:
            req = ", ".join(spec.get_param(k).label for k in spec.schema.required_param_cols)
            lines.append(f"- 必填欄：{req}")
        if spec.schema.note:
            lines.append(f"- 注意：{spec.schema.note}")
        lines.append("")
        print(f"[範本] {os.path.basename(path)}  ({df.shape[0]} 列 × {df.shape[1]} 欄)")
    with open(os.path.join(TPL, "範本說明.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("完成。說明檔：", os.path.join(TPL, "範本說明.md"))


if __name__ == "__main__":
    main()
