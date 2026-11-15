# Monitoring job: replays scored orders, writes drift/fairness/performance signals and the
# static monitoring site (Evidently AI reports) to build/monitoring_site.
# Build (from the repository root): docker build -f monitoring/Dockerfile -t dri-monitor .
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DRI_ROOT=/app MPLBACKEND=Agg
WORKDIR /app
COPY requirements-pipeline.txt requirements-monitoring.txt ./
RUN pip install -r requirements-monitoring.txt
COPY src/ src/
COPY models/ models/
RUN useradd --create-home --uid 1000 dri \
    && mkdir -p data docs/evidence/monitoring dashboard/data build/monitoring_site \
    && chown -R dri:dri /app
USER dri
ENTRYPOINT ["python", "-m", "src.monitoring.monitor"]
