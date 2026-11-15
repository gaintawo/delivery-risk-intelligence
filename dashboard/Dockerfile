# DRI stakeholder dashboard (Streamlit).
# Build (from the repository root):  docker build -f dashboard/Dockerfile -t dri-dashboard .
# Run:                                docker run --rm -p 8501:8501 dri-dashboard
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY dashboard/requirements.txt dashboard/requirements.txt
RUN pip install -r dashboard/requirements.txt
COPY .streamlit/ .streamlit/
COPY src/__init__.py src/__init__.py
COPY src/models/__init__.py src/models/explain.py src/models/
COPY models/ models/
COPY dashboard/ dashboard/
RUN useradd --create-home --uid 1000 dri && chown -R dri:dri /app
USER dri
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
