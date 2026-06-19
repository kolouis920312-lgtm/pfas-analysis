# PFAS 資料分析工具 — 網站版容器
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# fonts-noto-cjk：讓 matplotlib 在 Linux 上能正確顯示中文
# libgomp1：xgboost / scikit-learn 的 OpenMP 執行期
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-noto-cjk \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

# 用 $PORT（雲端平台會注入）；沒有就預設 8000
# 512MB 免費方案：-w 1（單一 worker 省記憶體；分析本來就被 app.py 的 _run_lock 串行化，
#   多 worker 對運算沒幫助，只會讓 numpy/matplotlib/plotly 的記憶體翻倍 → OOM）
# --threads 4：用執行緒處理並發（載預覽圖、靜態檔、互動地圖 iframe）；-t 600：複雜方法給 10 分鐘
# --max-requests：每處理約 120 個請求就回收並重啟 worker，釋放 matplotlib/numpy 累積的記憶體碎片
CMD gunicorn -w 1 -k gthread --threads 4 -t 600 --max-requests 120 --max-requests-jitter 20 -b 0.0.0.0:${PORT:-8000} app:app
