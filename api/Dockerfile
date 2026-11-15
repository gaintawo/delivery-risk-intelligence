# DRI scoring API.
# Build (from the repository root):  docker build -f api/Dockerfile -t dri-api .
# Run:                                docker run --rm -p 8000:8000 dri-api   ->  http://localhost:8000/docs
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DRI_EXPLANATION_DB=/app/logs/explanations.db

WORKDIR /app
# libgomp: OpenMP runtime needed by XGBoost (delivery-days model)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src/__init__.py src/__init__.py
COPY src/models/__init__.py src/models/explain.py src/models/
COPY src/actions/ src/actions/
COPY api/ api/
COPY models/ models/

RUN useradd --create-home --uid 1000 dri && mkdir -p logs && chown -R dri:dri /app
USER dri
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT','8000') + '/health'); sys.exit(0)"
# Render and most clouds pass $PORT; default 8000 locally
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
