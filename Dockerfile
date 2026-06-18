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
# -w 2：2 個 worker；--threads 4：每個 worker 4 執行緒；-t 600：複雜方法給 10 分鐘
CMD gunicorn -w 2 -k gthread --threads 4 -t 600 -b 0.0.0.0:${PORT:-8000} app:app
