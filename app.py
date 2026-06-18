# -*- coding: utf-8 -*-
"""
app.py — PFAS 資料分析工具（Flask 網站版）
==================================================
把原本的 Tkinter 桌面程式 (pfas_gui.py) 改成網站，
沿用同一套分析核心 pfas_toolkit，零修改方法程式碼。

本機執行：
    pip install -r requirements.txt
    python app.py
    # 開瀏覽器到 http://127.0.0.1:8000

正式部署（公開網際網路）：
    gunicorn -w 2 -k gthread -t 600 app:app
    （或用 Dockerfile / Procfile，見 網站說明.md）

設計重點
  ‧ 每位使用者上傳的資料各自存成一個 token，互不干擾
  ‧ 每次執行的輸出存進獨立資料夾，用網址提供預覽/下載
  ‧ 不接受使用者指定伺服器路徑（避免任意寫檔），輸出一律在暫存區
  ‧ 舊的暫存檔會自動清掉，避免塞爆磁碟
  ‧ matplotlib 用 Agg、單一處理程序內以 Lock 串行化產圖（pyplot 非執行緒安全）
"""
import io
import os
import re
import sys
import time
import uuid
import shutil
import tempfile
import threading
import traceback

import pandas as pd
from flask import (Flask, request, jsonify, render_template,
                   send_from_directory, abort)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pfas_toolkit import methods
from pfas_toolkit.core.io import load_csv, RunContext
from pfas_toolkit.core.spec import OutputSettings
from pfas_toolkit.core.validate import validate
from pfas_toolkit.core import theme as thememod

# ───────────────────────────────────────── 基本設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")        # 既有的 CSV 範本
EXAMPLES_DIR = os.path.join(BASE_DIR, "web", "static", "examples")  # 說明書範例縮圖
IMAGE_FORMATS = ["png", "svg", "pdf", "jpg"]
MAX_UPLOAD_MB = 32

# 工作區（資料集 + 執行輸出）放在系統暫存資料夾，部署時可用環境變數覆蓋
WORK_DIR = os.environ.get("PFAS_WORK_DIR", os.path.join(tempfile.gettempdir(), "pfas_web"))
DATASETS_DIR = os.path.join(WORK_DIR, "datasets")
RUNS_DIR = os.path.join(WORK_DIR, "runs")
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "web", "templates"),
            static_folder=os.path.join(BASE_DIR, "web", "static"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["JSON_AS_ASCII"] = False  # 讓中文不被轉成 \uXXXX

# pyplot 不是執行緒安全 → 同一個 process 內一次只跑一個分析
_run_lock = threading.Lock()

SPECS = methods.all_specs()
SPEC_BY_KEY = {s.key: s for s in SPECS}

_TOKEN_RE = re.compile(r"^[0-9a-f]{8,40}$")


# ───────────────────────────────────────── 序列化：把 spec 變成前端能用的 JSON
def param_to_dict(p):
    return {
        "key": p.key, "label": p.label, "kind": p.kind,
        "default": p.default, "choices": list(p.choices),
        "minimum": p.minimum, "maximum": p.maximum,
        "help": p.help, "optional": p.optional,
    }


def spec_to_dict(s):
    return {
        "key": s.key,
        "name": s.name,
        "summary": s.summary,
        "uses_colors": list(s.uses_colors),
        "template_columns": list(s.template_columns),
        "schema": {
            "min_rows": s.schema.min_rows,
            "min_numeric_cols": s.schema.min_numeric_cols,
            "note": s.schema.note,
        },
        "params": [param_to_dict(p) for p in s.params],
        "manual": dict(s.manual) if s.manual else {},
        "has_template": os.path.exists(
            os.path.join(TEMPLATES_DIR, f"{s.key}_template.csv")),
    }


def report_to_dict(rep):
    return {"ok": rep.ok, "errors": rep.errors, "warnings": rep.warnings,
            "info": rep.info, "text": rep.as_text()}


def example_urls():
    """掃 web/static/examples，回傳 {方法key: 縮圖網址}（說明書「快速上手」顯示範例輸出用）。"""
    out = {}
    try:
        for fn in sorted(os.listdir(EXAMPLES_DIR)):
            if fn.lower().endswith(".png"):
                out[os.path.splitext(fn)[0]] = f"/static/examples/{fn}"
    except Exception:
        pass
    return out


# ───────────────────────────────────────── 工具函式
def read_upload(file_storage):
    """讀上傳的 CSV，依序嘗試常見編碼（utf-8 / big5 / cp950 …）。"""
    raw = file_storage.read()
    last = None
    for enc in (None, "utf-8-sig", "big5", "cp950", "gb18030", "latin-1"):
        try:
            buf = io.BytesIO(raw)
            df = pd.read_csv(buf) if enc is None else pd.read_csv(buf, encoding=enc)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            last = e
    raise ValueError(f"無法判讀 CSV（已嘗試 utf-8 / big5 / cp950）：{last}")


def df_preview(df, n=8):
    head = df.head(n)
    return {
        "columns": [str(c) for c in df.columns],
        "rows": _json_safe_rows(head),
        "nrows": int(len(df)),
        "ncols": int(df.shape[1]),
    }


def _json_safe_rows(head):
    import json as _json
    return _json.loads(head.to_json(orient="values", date_format="iso"))


def save_dataset(df):
    token = uuid.uuid4().hex
    df.to_csv(os.path.join(DATASETS_DIR, token + ".csv"),
              index=False, encoding="utf-8-sig")
    return token


def load_dataset(token):
    if not token or not _TOKEN_RE.match(str(token)):
        return None
    path = os.path.join(DATASETS_DIR, token + ".csv")
    if not os.path.exists(path):
        return None
    return load_csv(path)


def coerce_params(spec, raw):
    """把前端傳來的字串/布林依參數型別轉好（對應原 GUI 的 collect_params）。"""
    raw = raw or {}
    p = {}
    for ps in spec.params:
        val = raw.get(ps.key, ps.default)
        if ps.kind == "int":
            try:
                p[ps.key] = int(float(val)) if str(val).strip() != "" else 0
            except Exception:
                p[ps.key] = 0
        elif ps.kind == "float":
            try:
                p[ps.key] = float(val) if str(val).strip() != "" else 0.0
            except Exception:
                p[ps.key] = 0.0
        elif ps.kind == "bool":
            p[ps.key] = (val if isinstance(val, bool)
                         else str(val).strip().lower() in ("1", "true", "on", "yes"))
        else:  # choice / column / text
            s = "" if val is None else str(val)
            p[ps.key] = "" if s == "(無)" else s
    return p


def build_theme(raw):
    """對應原 GUI 的 collect_theme，缺值一律回退預設。"""
    out = dict(thememod.DEFAULT_THEME)
    for k in ("font_family", "primary", "accent",
              "cmap_sequential", "cmap_diverging", "cmap_categorical"):
        v = (raw or {}).get(k)
        if v:
            out[k] = v
    return out


def cleanup_old(root, max_age_h):
    """刪掉超過指定時數的暫存檔/資料夾，best-effort。"""
    now = time.time()
    try:
        for name in os.listdir(root):
            p = os.path.join(root, name)
            try:
                if now - os.path.getmtime(p) > max_age_h * 3600:
                    shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
            except Exception:
                pass
    except Exception:
        pass


def files_to_links(paths, run_id, run_dir):
    items = []
    for p in paths:
        rel = os.path.relpath(p, run_dir).replace(os.sep, "/")
        items.append({"name": os.path.basename(p),
                      "url": f"/api/file/{run_id}/{rel}"})
    return items


# ───────────────────────────────────────── 路由
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return "ok"


@app.route("/api/meta")
def meta():
    return jsonify({
        "methods": [spec_to_dict(s) for s in SPECS],
        "glossary": methods.GLOSSARY,
        "examples": example_urls(),
        "theme_default": dict(thememod.DEFAULT_THEME),
        "fonts": list(thememod.FONT_CHOICES),
        "cmap_sequential": list(thememod.SEQUENTIAL_CMAPS),
        "cmap_diverging": list(thememod.DIVERGING_CMAPS),
        "cmap_categorical": list(thememod.CATEGORICAL_CMAPS),
        "image_formats": IMAGE_FORMATS,
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "沒有收到檔案。"}), 400
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "只接受 .csv 檔。"}), 400
    try:
        df = read_upload(f)
    except Exception as e:
        return jsonify({"error": f"無法讀取 CSV：{e}"}), 400
    if df.shape[1] == 0:
        return jsonify({"error": "讀不到任何欄位，請確認第一列是欄名。"}), 400
    token = save_dataset(df)
    cleanup_old(DATASETS_DIR, max_age_h=12)
    return jsonify({"dataset": token, "label": f.filename, "preview": df_preview(df)})


@app.route("/api/demo", methods=["POST"])
def demo():
    body = request.get_json(silent=True) or {}
    spec = SPEC_BY_KEY.get(body.get("method"))
    if not spec:
        return jsonify({"error": "未知的方法。"}), 400
    try:
        df = spec.make_demo()
    except Exception as e:
        return jsonify({"error": f"產生示範資料失敗：{e}"}), 500
    token = save_dataset(df)
    return jsonify({"dataset": token,
                    "label": f"（示範資料：{spec.name}）",
                    "preview": df_preview(df)})


@app.route("/api/validate", methods=["POST"])
def do_validate():
    body = request.get_json(silent=True) or {}
    spec = SPEC_BY_KEY.get(body.get("method"))
    if not spec:
        return jsonify({"error": "未知的方法。"}), 400
    df = load_dataset(body.get("dataset"))
    if df is None:
        return jsonify({"report": {"ok": False, "errors": [], "warnings": [],
                                   "info": [], "text": "請先選擇 CSV 或用示範資料。"}})
    params = coerce_params(spec, body.get("params"))
    rep = validate(df, spec, params)
    return jsonify({"report": report_to_dict(rep)})


@app.route("/api/run", methods=["POST"])
def do_run():
    body = request.get_json(silent=True) or {}
    spec = SPEC_BY_KEY.get(body.get("method"))
    if not spec:
        return jsonify({"error": "未知的方法。"}), 400
    df = load_dataset(body.get("dataset"))
    if df is None:
        return jsonify({"error": "請先選擇 CSV 或用示範資料。"}), 400

    params = coerce_params(spec, body.get("params"))
    rep = validate(df, spec, params)
    if not rep.ok:
        return jsonify({"error": "資料格式有誤，請先依驗證面板修正。",
                        "report": report_to_dict(rep)}), 400

    theme = build_theme(body.get("theme"))
    fmt = str(body.get("image_format") or "png").lower()
    if fmt not in IMAGE_FORMATS:
        fmt = "png"
    try:
        dpi = int(float(body.get("dpi") or 150))
    except Exception:
        dpi = 150
    dpi = max(50, min(dpi, 600))

    run_id = uuid.uuid4().hex
    run_dir = os.path.join(RUNS_DIR, run_id)
    out = OutputSettings(output_dir=run_dir, image_format=fmt, dpi=dpi, theme=theme)

    cleanup_old(RUNS_DIR, max_age_h=6)

    with _run_lock:
        try:
            ctx = RunContext(out)
            res = spec.run(df.copy(), params, ctx)
        except Exception:
            tb = traceback.format_exc()
            msg = tb.strip().splitlines()[-1] if tb.strip() else "執行失敗"
            return jsonify({"error": msg, "trace": tb,
                            "report": report_to_dict(rep)}), 500

    return jsonify({
        "ok": True,
        "run": run_id,
        "log": res.log,
        "summary": res.summary,
        "previews": files_to_links(res.previews, run_id, run_dir),
        "figures": files_to_links(res.figures, run_id, run_dir),
        "tables": files_to_links(res.tables, run_id, run_dir),
        "extras": files_to_links(res.extras, run_id, run_dir),
        "report": report_to_dict(rep),
    })


@app.route("/api/file/<run_id>/<path:relpath>")
def serve_file(run_id, relpath):
    if not _TOKEN_RE.match(run_id):
        abort(404)
    run_dir = os.path.join(RUNS_DIR, run_id)
    if not os.path.isdir(run_dir):
        abort(404)
    as_attach = request.args.get("dl") == "1"
    return send_from_directory(run_dir, relpath, as_attachment=as_attach)


@app.route("/api/template/<key>")
def template_csv(key):
    if not re.match(r"^[a-z_]+$", key):
        abort(404)
    fname = f"{key}_template.csv"
    if not os.path.exists(os.path.join(TEMPLATES_DIR, fname)):
        abort(404)
    return send_from_directory(TEMPLATES_DIR, fname, as_attachment=True)


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": f"檔案太大，上限 {MAX_UPLOAD_MB} MB。"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
