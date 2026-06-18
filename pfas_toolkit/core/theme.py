# -*- coding: utf-8 -*-
"""
theme.py — 字型與顏色主題（集中管理，取代原本散落各檔的硬編設定）
==================================================================
顏色採「角色」制，方法只引用角色名稱，使用者在 GUI 改一次、全部生效：

  primary           主色（長條圖、散佈點、ALE 線）
  accent            強調色（參考線、理想線 y=x、biplot 箭頭）
  cmap_sequential   連續色階（熱圖、SOM component planes / U-matrix、偵測熱圖）
  cmap_diverging    分散色階（相關係數熱圖，正負對稱）
  cmap_categorical  分類色盤（分群、站點、季節）

matplotlib 後端在此統一設為 Agg（不開視窗、可在背景執行緒安全產圖）。
"""
import matplotlib
matplotlib.use("Agg")

DEFAULT_THEME = {
    "font_family": "Microsoft JhengHei",
    "primary": "#4682b4",        # steelblue
    "accent": "#ff6347",         # tomato
    "cmap_sequential": "viridis",
    "cmap_diverging": "RdBu_r",
    "cmap_categorical": "tab10",
}

# 給 GUI 下拉用的常見 colormap 清單
SEQUENTIAL_CMAPS = ["viridis", "plasma", "magma", "cividis", "Greens",
                    "Blues", "Oranges", "Purples", "YlGnBu", "bone_r"]
DIVERGING_CMAPS = ["RdBu_r", "coolwarm", "bwr", "seismic", "PiYG", "PRGn", "Spectral"]
CATEGORICAL_CMAPS = ["tab10", "tab20", "Set1", "Set2", "Set3", "Paired", "Dark2"]

# 常見可放中文的字型（Windows 為主）
FONT_CHOICES = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei",
                "PMingLiU", "DFKai-SB", "Noto Sans CJK TC", "DejaVu Sans"]


def cmap_swatch(name: str, n: int = 16) -> list:
    """把一個 matplotlib colormap 取樣成 hex 色碼清單（給 UI 畫顏色示意圖）。

    分類色盤（ListedColormap）回傳它的離散顏色；連續/分散色階等距取樣 n 個。
    取不到（名稱錯）就回空清單，呼叫端自行容錯。
    """
    from matplotlib.colors import to_hex
    try:
        import matplotlib as mpl
        try:
            cmap = mpl.colormaps[name]            # matplotlib ≥ 3.6
        except Exception:
            cmap = mpl.cm.get_cmap(name)
    except Exception:
        return []
    discrete = getattr(cmap, "colors", None)
    if discrete is not None and len(discrete) <= 24:
        return [to_hex(c) for c in discrete]
    denom = max(n - 1, 1)
    return [to_hex(cmap(i / denom)) for i in range(n)]


def all_cmap_swatches() -> dict:
    """所有內建 colormap 名稱 → 色碼清單（一次給前端，選色時即時預覽）。"""
    out = {}
    for nm in (*SEQUENTIAL_CMAPS, *DIVERGING_CMAPS, *CATEGORICAL_CMAPS):
        out[nm] = cmap_swatch(nm)
    return out


def merge_theme(theme: dict | None) -> dict:
    """把使用者主題疊在預設值上，確保每個角色都有值。"""
    out = dict(DEFAULT_THEME)
    if theme:
        out.update({k: v for k, v in theme.items() if v})
    return out


def apply_theme(theme: dict | None):
    """套用字型設定到 matplotlib（每次產圖前呼叫）。"""
    import matplotlib.pyplot as plt
    t = merge_theme(theme)
    fam = t.get("font_family") or "Microsoft JhengHei"
    # 使用者字型優先，後面接通用備援，避免缺字型時整張圖變方框
    plt.rcParams["font.sans-serif"] = [fam, "Microsoft JhengHei", "SimHei",
                                       "Noto Sans CJK TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def get_plt(theme: dict | None):
    """套用主題並回傳 pyplot（方法內統一用這個取得 plt）。"""
    apply_theme(theme)
    import matplotlib.pyplot as plt
    return plt
