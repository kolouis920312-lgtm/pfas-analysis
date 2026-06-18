# -*- coding: utf-8 -*-
"""
io.py — 讀檔 + 執行情境（RunContext）
=======================================
RunContext 把「輸出設定 + 存圖 + 存表 + 記錄訊息」包成一個物件傳給每個方法，
方法只要呼叫 ctx.save_fig / ctx.save_table / ctx.log，
就會自動套用使用者選的輸出資料夾、圖檔格式、dpi，並蒐集結果路徑給 GUI。
"""
import os
import sys
import numpy as np
import pandas as pd

from .spec import OutputSettings, RunResult
from .theme import merge_theme


def load_csv(path: str) -> pd.DataFrame:
    """讀 CSV；自動去除欄名前後空白（避免 'PFOA ' 這類隱形不一致）。"""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


class RunContext:
    def __init__(self, out: OutputSettings):
        self.out = out
        self.out.theme = merge_theme(out.theme)
        os.makedirs(out.output_dir, exist_ok=True)
        self.figures: list = []     # 交付格式
        self.previews: list = []    # GUI 預覽 PNG
        self.tables: list = []
        self.extras: list = []
        self._log: list = []

    # ── 訊息 ──────────────────────────────────────────
    def log(self, *args):
        s = " ".join(str(a) for a in args)
        self._log.append(s)          # GUI 永遠拿到完整字串
        # console 可能是 cp950，印不出 ⚠ ² ↔ 等字元 → 永不讓它中斷執行
        try:
            print(s)
        except Exception:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            try:
                print(s.encode(enc, "replace").decode(enc, "replace"))
            except Exception:
                pass

    # ── 顏色角色 ──────────────────────────────────────
    @property
    def theme(self) -> dict:
        return self.out.theme

    def color(self, role: str, default=None):
        return self.out.theme.get(role, default)

    # ── 存表 ──────────────────────────────────────────
    def save_table(self, df: pd.DataFrame, name: str, index=True) -> str:
        path = os.path.join(self.out.output_dir, f"{name}.csv")
        df.to_csv(path, index=index, encoding="utf-8-sig")
        self.tables.append(path)
        return path

    def add_extra(self, path: str):
        self.extras.append(path)

    # ── 存圖（自動套格式/dpi，並產生預覽 PNG）──────────
    def save_fig(self, fig, name: str) -> str:
        import matplotlib.pyplot as plt
        fmt = (self.out.image_format or "png").lower()
        path = os.path.join(self.out.output_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=self.out.dpi, bbox_inches="tight")
        self.figures.append(path)
        if fmt in ("png", "jpg", "jpeg"):
            self.previews.append(path)
        else:
            # 向量格式（svg/pdf）無法直接在 Tk 預覽 → 另存一張 PNG 縮圖供預覽
            pdir = os.path.join(self.out.output_dir, "_preview")
            os.makedirs(pdir, exist_ok=True)
            ppath = os.path.join(pdir, f"{name}.png")
            fig.savefig(ppath, dpi=110, bbox_inches="tight")
            self.previews.append(ppath)
        plt.close(fig)
        return path

    # ── 收尾 ──────────────────────────────────────────
    def result(self, summary: str = "") -> RunResult:
        return RunResult(figures=self.figures, previews=self.previews,
                         tables=self.tables, extras=self.extras,
                         log="\n".join(self._log), summary=summary)
