# -*- coding: utf-8 -*-
"""
selftest.py — 對所有方法跑一次示範資料，驗證環境與整合是否正常。
用法： python -m pfas_toolkit.selftest
"""
import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 主控台預設 cp950，切成 UTF-8 才能印出中文與符號
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pfas_toolkit import methods
from pfas_toolkit.core.spec import OutputSettings
from pfas_toolkit.core.io import RunContext
from pfas_toolkit.core.theme import DEFAULT_THEME
from pfas_toolkit.core.validate import validate


def main():
    outdir = os.path.join(tempfile.gettempdir(), "pfas_selftest")
    ok_all = True
    for spec in methods.all_specs():
        try:
            df = spec.make_demo()
            params = spec.default_params()
            rep = validate(df, spec, params)
            out = OutputSettings(output_dir=os.path.join(outdir, spec.key),
                                 image_format="png", dpi=110, theme=dict(DEFAULT_THEME))
            ctx = RunContext(out)
            res = spec.run(df, params, ctx)
            print(f"[OK]   {spec.key:12} 驗證={'通過' if rep.ok else '不通過'} "
                  f"圖={len(res.figures)} 表={len(res.tables)} 額外={len(res.extras)}")
        except Exception as e:
            ok_all = False
            print(f"[FAIL] {spec.key:12} {type(e).__name__}: {e}")
            traceback.print_exc()
    print("\n" + ("全部通過 ✓" if ok_all else "有方法失敗 ✗"))
    print("輸出資料夾：", outdir)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
