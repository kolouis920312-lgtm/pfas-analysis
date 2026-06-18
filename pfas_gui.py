# -*- coding: utf-8 -*-
"""
pfas_gui.py — PFAS 資料分析工具（Tkinter 桌面程式）
====================================================
雙擊或執行：  python pfas_gui.py

功能：
  ‧ 左側選方法（9 種）
  ‧ 選 CSV 或用示範資料；執行前自動「驗證格式」並列出問題
  ‧ 參數面板依方法自動產生，不必改任何程式碼
  ‧ 顏色：主色/強調色用原生選色盤；連續/分散/分類色階用下拉；可改字型
  ‧ 輸出：自選資料夾、自選格式 (png/svg/pdf/jpg)、dpi
  ‧ 執行後直接預覽圖、看 log、一鍵開啟輸出資料夾
  ‧ 「儲存為預設」會把目前設定寫回 config.json
"""
import os
import sys
import threading
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox

from pfas_toolkit import methods
from pfas_toolkit.core import config as cfgmod
from pfas_toolkit.core.io import load_csv, RunContext
from pfas_toolkit.core.spec import OutputSettings
from pfas_toolkit.core.validate import validate
from pfas_toolkit.core import theme as thememod

IMAGE_FORMATS = ["png", "svg", "pdf", "jpg"]


def open_path(path):
    """用系統預設程式開啟檔案/資料夾。"""
    try:
        os.startfile(path)  # Windows
    except Exception:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(path))


class ScrollFrame(ttk.Frame):
    """可垂直捲動的容器；內容放在 self.inner。"""
    def __init__(self, parent, **kw):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, **kw)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfig(self._win, width=e.width))
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _wheel(self, e):
        self.canvas.yview_scroll(int(-e.delta / 120), "units")


class App:
    def __init__(self, root):
        self.root = root
        root.title("PFAS 資料分析工具")
        root.geometry("1200x840")

        self.cfg = cfgmod.load_config()
        self.theme = dict(thememod.merge_theme(self.cfg.get("theme")))
        self.specs = methods.all_specs()
        self.current_spec = None
        self.df = None
        self.data_label = "（尚未選擇資料）"

        self.param_widgets = {}   # key -> (kind, tk.Variable)
        self.col_combos = {}      # key -> ttk.Combobox（欄位型參數）
        self._photos = []         # 預覽圖參考，避免被回收
        self._result = None
        self._error = None

        self._build_ui()
        if self.specs:
            self.method_list.selection_set(0)
            self._on_method_select()

    # ────────────────────────────────────────────── UI 骨架
    def _build_ui(self):
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        main = ttk.PanedWindow(outer, orient="vertical")
        main.pack(fill="both", expand=True, padx=6, pady=6)

        top = ttk.PanedWindow(main, orient="horizontal")
        main.add(top, weight=3)

        # 左：方法清單
        left = ttk.LabelFrame(top, text="分析方法")
        top.add(left, weight=1)
        self.method_list = tk.Listbox(left, exportselection=False, activestyle="dotbox")
        for s in self.specs:
            self.method_list.insert("end", s.name)
        self.method_list.pack(fill="both", expand=True, padx=6, pady=6)
        self.method_list.bind("<<ListboxSelect>>", lambda e: self._on_method_select())
        self.summary_lbl = ttk.Label(left, text="", wraplength=240, foreground="#444")
        self.summary_lbl.pack(fill="x", padx=8, pady=(0, 8))

        # 右：設定（可捲動）
        self.settings = ScrollFrame(top)
        top.add(self.settings, weight=3)
        self._build_settings(self.settings.inner)

        # 下：結果
        bottom = ttk.PanedWindow(main, orient="horizontal")
        main.add(bottom, weight=2)

        logf = ttk.LabelFrame(bottom, text="執行紀錄 / 摘要")
        bottom.add(logf, weight=1)
        self.log_text = tk.Text(logf, height=12, wrap="word", state="disabled",
                                font=("Consolas", 9))
        logsb = ttk.Scrollbar(logf, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=logsb.set)
        logsb.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        prevf = ttk.LabelFrame(bottom, text="圖像預覽")
        bottom.add(prevf, weight=1)
        self.preview = ScrollFrame(prevf)
        self.preview.pack(fill="both", expand=True)

        # 狀態列
        self.status = ttk.Label(outer, text="就緒", relief="sunken", anchor="w")
        self.status.pack(fill="x", side="bottom")

    def _build_settings(self, p):
        # 資料來源
        d = ttk.LabelFrame(p, text="① 資料來源")
        d.pack(fill="x", padx=8, pady=6)
        self.path_var = tk.StringVar(value=self.data_label)
        ttk.Entry(d, textvariable=self.path_var, state="readonly").grid(
            row=0, column=0, columnspan=4, sticky="ew", padx=6, pady=6)
        ttk.Button(d, text="選擇 CSV…", command=self._choose_file).grid(row=1, column=0, padx=4, pady=4)
        ttk.Button(d, text="用示範資料", command=self._use_demo).grid(row=1, column=1, padx=4, pady=4)
        ttk.Button(d, text="開啟此方法範本", command=self._open_template).grid(row=1, column=2, padx=4, pady=4)
        ttk.Button(d, text="重新驗證", command=self._refresh_validation).grid(row=1, column=3, padx=4, pady=4)
        d.columnconfigure(0, weight=1)

        # 驗證
        v = ttk.LabelFrame(p, text="② 格式驗證")
        v.pack(fill="x", padx=8, pady=6)
        self.valid_text = tk.Text(v, height=6, wrap="word", state="disabled", font=("Consolas", 9))
        self.valid_text.pack(fill="x", padx=6, pady=6)

        # 參數
        self.params_frame = ttk.LabelFrame(p, text="③ 參數（不必改程式碼）")
        self.params_frame.pack(fill="x", padx=8, pady=6)

        # 顏色與字型
        c = ttk.LabelFrame(p, text="④ 顏色與字型（改一次、全部生效）")
        c.pack(fill="x", padx=8, pady=6)
        self._build_theme(c)

        # 輸出
        o = ttk.LabelFrame(p, text="⑤ 輸出設定")
        o.pack(fill="x", padx=8, pady=6)
        self.outdir_var = tk.StringVar(value=self.cfg["output"].get("output_dir", cfgmod.DEFAULT_OUTPUT_DIR))
        ttk.Label(o, text="輸出資料夾").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(o, textvariable=self.outdir_var).grid(row=0, column=1, columnspan=2, sticky="ew", padx=6, pady=4)
        ttk.Button(o, text="瀏覽…", command=self._choose_outdir).grid(row=0, column=3, padx=4, pady=4)
        ttk.Label(o, text="圖檔格式").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.fmt_var = tk.StringVar(value=self.cfg["output"].get("image_format", "png"))
        ttk.Combobox(o, textvariable=self.fmt_var, values=IMAGE_FORMATS, state="readonly",
                     width=8).grid(row=1, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(o, text="dpi").grid(row=1, column=2, sticky="e", padx=6, pady=4)
        self.dpi_var = tk.StringVar(value=str(self.cfg["output"].get("dpi", 150)))
        ttk.Entry(o, textvariable=self.dpi_var, width=8).grid(row=1, column=3, sticky="w", padx=6, pady=4)
        o.columnconfigure(1, weight=1)

        # 動作按鈕
        b = ttk.Frame(p)
        b.pack(fill="x", padx=8, pady=10)
        self.run_btn = ttk.Button(b, text="▶ 執行", command=self._on_run)
        self.run_btn.pack(side="left", padx=4)
        ttk.Button(b, text="儲存為預設", command=self._save_defaults).pack(side="left", padx=4)
        ttk.Button(b, text="開啟輸出資料夾", command=self._open_outdir).pack(side="left", padx=4)

    def _build_theme(self, c):
        ttk.Label(c, text="字型").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.font_var = tk.StringVar(value=self.theme.get("font_family"))
        ttk.Combobox(c, textvariable=self.font_var, values=thememod.FONT_CHOICES,
                     width=22).grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(c, text="主色").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.primary_btn = tk.Button(c, width=12, relief="groove",
                                     command=lambda: self._pick_color("primary", self.primary_btn))
        self.primary_btn.grid(row=1, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(c, text="強調色").grid(row=1, column=2, sticky="e", padx=6, pady=4)
        self.accent_btn = tk.Button(c, width=12, relief="groove",
                                    command=lambda: self._pick_color("accent", self.accent_btn))
        self.accent_btn.grid(row=1, column=3, sticky="w", padx=6, pady=4)
        self._paint_color_btn(self.primary_btn, "primary")
        self._paint_color_btn(self.accent_btn, "accent")

        ttk.Label(c, text="連續色階").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.seq_var = tk.StringVar(value=self.theme.get("cmap_sequential"))
        ttk.Combobox(c, textvariable=self.seq_var, values=thememod.SEQUENTIAL_CMAPS,
                     state="readonly", width=14).grid(row=2, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(c, text="分散色階").grid(row=2, column=2, sticky="e", padx=6, pady=4)
        self.div_var = tk.StringVar(value=self.theme.get("cmap_diverging"))
        ttk.Combobox(c, textvariable=self.div_var, values=thememod.DIVERGING_CMAPS,
                     state="readonly", width=14).grid(row=2, column=3, sticky="w", padx=6, pady=4)
        ttk.Label(c, text="分類色盤").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.cat_var = tk.StringVar(value=self.theme.get("cmap_categorical"))
        ttk.Combobox(c, textvariable=self.cat_var, values=thememod.CATEGORICAL_CMAPS,
                     state="readonly", width=14).grid(row=3, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(c, text="連續→熱圖/SOM；分散→相關；分類→分群/站別",
                  foreground="#777").grid(row=3, column=2, columnspan=2, sticky="w", padx=6, pady=4)

    # ────────────────────────────────────────────── 主色/強調色
    def _paint_color_btn(self, btn, role):
        col = self.theme.get(role, "#888888")
        btn.config(bg=col, activebackground=col, text=col,
                   fg=self._contrast(col))

    def _pick_color(self, role, btn):
        res = colorchooser.askcolor(color=self.theme.get(role), title="選擇顏色")
        if res and res[1]:
            self.theme[role] = res[1]
            self._paint_color_btn(btn, role)

    @staticmethod
    def _contrast(hexcol):
        try:
            h = hexcol.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"
        except Exception:
            return "#000000"

    # ────────────────────────────────────────────── 方法切換 + 參數
    def _on_method_select(self):
        sel = self.method_list.curselection()
        if not sel:
            return
        self.current_spec = self.specs[sel[0]]
        self.summary_lbl.config(text=self.current_spec.summary)
        self._build_params()
        self._refresh_columns()
        self._refresh_validation()

    def _build_params(self):
        for w in self.params_frame.winfo_children():
            w.destroy()
        self.param_widgets.clear()
        self.col_combos.clear()
        overrides = self.cfg.get("methods", {}).get(self.current_spec.key, {})
        self.params_frame.columnconfigure(1, weight=1)

        for r, ps in enumerate(self.current_spec.params):
            ttk.Label(self.params_frame, text=ps.label).grid(
                row=r, column=0, sticky="w", padx=6, pady=3)
            default = overrides.get(ps.key, ps.default)

            if ps.kind == "bool":
                var = tk.BooleanVar(value=bool(default))
                ttk.Checkbutton(self.params_frame, variable=var).grid(
                    row=r, column=1, sticky="w", padx=6, pady=3)
            elif ps.kind == "choice":
                var = tk.StringVar(value=str(default))
                ttk.Combobox(self.params_frame, textvariable=var, values=ps.choices,
                             state="readonly", width=18).grid(
                    row=r, column=1, sticky="w", padx=6, pady=3)
            elif ps.kind == "column":
                var = tk.StringVar(value="" if default is None else str(default))
                combo = ttk.Combobox(self.params_frame, textvariable=var, state="readonly", width=18)
                combo.grid(row=r, column=1, sticky="w", padx=6, pady=3)
                combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_validation())
                self.col_combos[ps.key] = combo
            else:  # int / float / text
                var = tk.StringVar(value="" if default is None else str(default))
                ttk.Entry(self.params_frame, textvariable=var, width=20).grid(
                    row=r, column=1, sticky="w", padx=6, pady=3)

            self.param_widgets[ps.key] = (ps.kind, var)
            if ps.help:
                ttk.Label(self.params_frame, text=ps.help, foreground="#888",
                          wraplength=320).grid(row=r, column=2, sticky="w", padx=6, pady=3)

    def _col_values(self, ps):
        cols = list(self.df.columns) if self.df is not None else []
        return (["(無)"] if ps.optional else []) + cols

    def _refresh_columns(self):
        if not self.current_spec:
            return
        overrides = self.cfg.get("methods", {}).get(self.current_spec.key, {})
        for key, combo in self.col_combos.items():
            ps = self.current_spec.get_param(key)
            vals = self._col_values(ps)
            combo["values"] = vals
            _, var = self.param_widgets[key]
            want = overrides.get(key, ps.default)
            if want in vals:
                var.set(want)
            elif var.get() not in vals:
                var.set("(無)" if ps.optional else (vals[0] if vals else ""))

    # ────────────────────────────────────────────── 蒐集設定
    def collect_params(self):
        p = {}
        for key, (kind, var) in self.param_widgets.items():
            val = var.get()
            if kind == "int":
                try:
                    p[key] = int(float(val)) if str(val).strip() != "" else 0
                except Exception:
                    p[key] = 0
            elif kind == "float":
                try:
                    p[key] = float(val) if str(val).strip() != "" else 0.0
                except Exception:
                    p[key] = 0.0
            elif kind == "bool":
                p[key] = bool(val)
            else:
                s = str(val)
                p[key] = "" if s == "(無)" else s
        return p

    def collect_theme(self):
        t = dict(self.theme)
        t["font_family"] = self.font_var.get().strip() or "Microsoft JhengHei"
        t["cmap_sequential"] = self.seq_var.get()
        t["cmap_diverging"] = self.div_var.get()
        t["cmap_categorical"] = self.cat_var.get()
        return t

    def collect_output(self):
        try:
            dpi = int(float(self.dpi_var.get()))
        except Exception:
            dpi = 150
        return OutputSettings(
            output_dir=self.outdir_var.get().strip() or cfgmod.DEFAULT_OUTPUT_DIR,
            image_format=self.fmt_var.get() or "png", dpi=dpi,
            theme=self.collect_theme())

    # ────────────────────────────────────────────── 資料來源
    def _choose_file(self):
        path = filedialog.askopenfilename(
            title="選擇資料 CSV",
            filetypes=[("CSV 檔", "*.csv"), ("所有檔案", "*.*")])
        if not path:
            return
        try:
            self.df = load_csv(path)
        except Exception as e:
            messagebox.showerror("讀取失敗", f"無法讀取 CSV：\n{e}")
            return
        self.data_label = path
        self.path_var.set(path)
        self._refresh_columns()
        self._refresh_validation()

    def _use_demo(self):
        if not self.current_spec:
            return
        self.df = self.current_spec.make_demo()
        self.data_label = f"（示範資料：{self.current_spec.name}）"
        self.path_var.set(self.data_label)
        self._refresh_columns()
        self._refresh_validation()

    def _open_template(self):
        if not self.current_spec:
            return
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "templates", f"{self.current_spec.key}_template.csv")
        if os.path.exists(path):
            open_path(path)
        else:
            messagebox.showinfo("找不到範本",
                                f"找不到 {path}\n可執行 build_templates.py 重新產生範本。")

    # ────────────────────────────────────────────── 驗證
    def _refresh_validation(self):
        if not self.current_spec:
            return
        self.valid_text.config(state="normal")
        self.valid_text.delete("1.0", "end")
        if self.df is None:
            self.valid_text.insert("end", "請先選擇 CSV 或按「用示範資料」。")
        else:
            rep = validate(self.df, self.current_spec, self.collect_params())
            self.valid_text.insert("end", rep.as_text())
        self.valid_text.config(state="disabled")

    # ────────────────────────────────────────────── 執行
    def _on_run(self):
        if not self.current_spec:
            return
        if self.df is None:
            messagebox.showwarning("尚未選擇資料", "請先選擇 CSV 或按「用示範資料」。")
            return
        params = self.collect_params()
        rep = validate(self.df, self.current_spec, params)
        self._refresh_validation()
        if not rep.ok:
            messagebox.showerror("資料格式有誤",
                                 "請先依「格式驗證」面板修正以下問題：\n\n" + "\n".join(rep.errors))
            return
        out = self.collect_output()
        self._result = None
        self._error = None
        self.run_btn.config(state="disabled")
        self._set_status("執行中…")
        df = self.df.copy()
        spec = self.current_spec
        t = threading.Thread(target=self._worker, args=(spec, df, params, out), daemon=True)
        t.start()
        self.root.after(150, self._poll)

    def _worker(self, spec, df, params, out):
        try:
            ctx = RunContext(out)
            self._result = spec.run(df, params, ctx)
        except Exception:
            self._error = traceback.format_exc()

    def _poll(self):
        if self._result is None and self._error is None:
            self.root.after(150, self._poll)
            return
        self.run_btn.config(state="normal")
        if self._error:
            self._set_status("執行失敗")
            self._show_log(self._error)
            messagebox.showerror("執行失敗", self._error.strip().splitlines()[-1])
        else:
            res = self._result
            self._set_status(f"完成 ✓  圖 {len(res.figures)}、表 {len(res.tables)}")
            self._show_log(res.log + "\n\n【摘要】\n" + res.summary +
                           f"\n\n輸出資料夾：{self.outdir_var.get()}")
            self._show_previews(res.previews)

    # ────────────────────────────────────────────── 結果顯示
    def _set_status(self, msg):
        self.status.config(text=msg)

    def _show_log(self, text):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", text)
        self.log_text.config(state="disabled")

    def _show_previews(self, paths):
        for w in self.preview.inner.winfo_children():
            w.destroy()
        self._photos.clear()
        try:
            from PIL import Image, ImageTk
        except Exception:
            ttk.Label(self.preview.inner, text="（需安裝 Pillow 才能預覽圖像）").pack()
            return
        if not paths:
            ttk.Label(self.preview.inner, text="（此方法沒有圖像輸出）").pack(padx=8, pady=8)
            return
        for pth in paths:
            try:
                img = Image.open(pth)
                img.thumbnail((460, 460))
                photo = ImageTk.PhotoImage(img)
                self._photos.append(photo)
                ttk.Label(self.preview.inner, text=os.path.basename(pth),
                          foreground="#555").pack(anchor="w", padx=6, pady=(8, 0))
                tk.Label(self.preview.inner, image=photo, relief="solid", bd=1).pack(
                    anchor="w", padx=6, pady=2)
            except Exception as e:
                ttk.Label(self.preview.inner, text=f"{os.path.basename(pth)}（預覽失敗：{e}）").pack()

    # ────────────────────────────────────────────── 其他動作
    def _choose_outdir(self):
        d = filedialog.askdirectory(title="選擇輸出資料夾",
                                    initialdir=self.outdir_var.get() or ".")
        if d:
            self.outdir_var.set(d)

    def _open_outdir(self):
        d = self.outdir_var.get().strip() or cfgmod.DEFAULT_OUTPUT_DIR
        os.makedirs(d, exist_ok=True)
        open_path(d)

    def _save_defaults(self):
        self.cfg["theme"] = self.collect_theme()
        out = self.collect_output()
        self.cfg["output"] = {"output_dir": out.output_dir,
                              "image_format": out.image_format, "dpi": out.dpi}
        if self.current_spec:
            self.cfg.setdefault("methods", {})[self.current_spec.key] = self.collect_params()
        try:
            path = cfgmod.save_config(self.cfg)
            self._set_status(f"已儲存預設 → {path}")
            messagebox.showinfo("已儲存", f"目前設定已寫入：\n{path}")
        except Exception as e:
            messagebox.showerror("儲存失敗", str(e))


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
