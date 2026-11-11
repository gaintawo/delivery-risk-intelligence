# DRI data pipeline container.
# Build:  docker build -t dri-pipeline .
# Run:    docker run --rm -v "$PWD/data:/app/data" -v "$PWD/logs:/app/logs" dri-pipeline
# Test:   docker run --rm --entrypoint pytest dri-pipeline -q --ignore=tests/test_structure.py
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DRI_ROOT=/app \
    DRI_ACTOR=docker-pipeline \
    PREFECT_HOME=/tmp/prefect \
    PREFECT_SERVER_ANALYTICS_ENABLED=false \
    GX_ANALYTICS_ENABLED=false

WORKDIR /app

COPY requirements-pipeline.txt .
RUN pip install -r requirements-pipeline.txt

COPY src/ src/
COPY tests/ tests/
COPY gx/great_expectations.yml gx/great_expectations.yml

# Run as a non-root user; data, logs and evidence are mounted or written at runtime.
RUN useradd --create-home --uid 1000 dri \
    && mkdir -p data/raw data/processed logs docs/evidence \
    && chown -R dri:dri /app
USER dri

ENTRYPOINT ["python", "-m", "src.pipeline.flow"]
