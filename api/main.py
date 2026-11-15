"""DRI scoring API.

    uvicorn api.main:app --reload        # then open http://127.0.0.1:8000/docs

POST /predict scores one order at the moment of purchase and returns the
late-delivery risk, whether it falls in the team's daily review list, the top
three reasons in plain language (SHAP), the expected delivery time and
suggested low-cost actions. Every response is written to the explanation store
(GET /explanations/{prediction_id}) and counted in Prometheus metrics (/metrics).
"""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from api.store import ExplanationStore
from src.actions.recommend import suggest
from src.models.explain import Explainer

MODEL_DIR = Path(os.getenv("DRI_MODEL_DIR", Path(__file__).resolve().parents[1] / "models"))
STATE: dict = {}

PREDICTIONS = Counter("dri_predictions_total", "Orders scored", ["flagged", "region"])
LATENCY = Histogram("dri_predict_seconds", "Time to score one order",
                    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
SCORE = Histogram("dri_late_probability", "Distribution of risk scores (drift signal)",
                  buckets=tuple(i / 10 for i in range(1, 11)))
MODEL_INFO = Gauge("dri_model_info", "Model in service", ["model", "version", "algorithm"])


class Order(BaseModel):
    """Facts known at the moment of purchase. Field names match the data dictionary."""
    promised_days: float = Field(..., ge=0, le=200, description="Days between purchase and promised date")
    purchase_dow: int = Field(..., ge=0, le=6, description="0 = Monday")
    purchase_month: int = Field(..., ge=1, le=12)
    purchase_hour: int = Field(..., ge=0, le=23)
    is_peak_season: int = Field(0, ge=0, le=1, description="1 between 20 Nov and 24 Dec")
    distance_km: float = Field(..., ge=0, le=5000)
    interstate: int = Field(..., ge=0, le=1)
    n_items: int = Field(1, ge=1)
    n_sellers: int = Field(1, ge=1)
    price_total: float = Field(..., gt=0)
    freight_total: float = Field(..., ge=0)
    freight_ratio: float | None = Field(None, ge=0, description="Computed from freight and price if omitted")
    weight_total_g: float = Field(..., ge=0)
    volume_total_cm3: float = Field(..., ge=0)
    any_size_imputed: int = Field(0, ge=0, le=1)
    is_boleto: int | None = Field(None, ge=0, le=1, description="Derived from payment_type if omitted")
    payment_installments: int = Field(1, ge=1)
    n_payment_methods: int = Field(1, ge=1)
    seller_hist_orders: int = Field(0, ge=0)
    seller_hist_late_rate: float | None = Field(None, ge=0, le=1, description="Null for a new seller")
    seller_hist_handling_days: float | None = Field(None, ge=0)
    # Recent market conditions: supplied by the daily batch job from the warehouse.
    # If omitted, the model imputes the training median (and the reasons say less).
    state_late_rate_30d: float | None = Field(None, ge=0, le=1, description="Late rate in the customer's state, last 30 days")
    state_delivery_days_30d: float | None = Field(None, ge=0, description="Mean delivery days in the customer's state, last 30 days")
    promise_slack_days: float | None = Field(None, description="promised_days - state_delivery_days_30d (computed if omitted)")
    market_late_rate_14d: float | None = Field(None, ge=0, le=1, description="Late rate across Brazil, last 14 days")
    seller_orders_7d: int = Field(0, ge=0, description="Orders the seller received in the previous 7 days")
    customer_state: str = Field(..., min_length=2, max_length=2)
    seller_state: str = Field(..., min_length=2, max_length=2)
    customer_region: Literal["North", "Northeast", "Central-West", "South", "Southeast"]
    primary_category: str = "unknown"
    payment_type: Literal["credit_card", "boleto", "voucher", "debit_card"] = "credit_card"
    seller_size: Literal["new", "small", "large"] | None = None

    model_config = {"json_schema_extra": {"example": {
        "promised_days": 18, "purchase_dow": 4, "purchase_month": 11, "purchase_hour": 21,
        "is_peak_season": 1, "distance_km": 2110, "interstate": 1, "n_items": 1, "n_sellers": 1,
        "price_total": 129.9, "freight_total": 38.5, "weight_total_g": 1800, "volume_total_cm3": 9000,
        "payment_installments": 3, "seller_hist_orders": 42, "seller_hist_late_rate": 0.19,
        "seller_hist_handling_days": 3.4, "state_late_rate_30d": 0.09, "state_delivery_days_30d": 19.6,
        "market_late_rate_14d": 0.07, "seller_orders_7d": 12, "customer_state": "BA", "seller_state": "SP",
        "customer_region": "Northeast", "primary_category": "health_beauty", "payment_type": "boleto"}}}


class Prediction(BaseModel):
    prediction_id: str = Field(..., description="Key for GET /explanations/{prediction_id} (audit trail)")
    late_probability: float
    flagged: bool
    flagged_fairness_adjusted: bool | None = Field(
        None, description="Decision with region-specific thresholds (Fairlearn ThresholdOptimizer, recall parity)")
    risk_band: Literal["low", "medium", "high"]
    threshold: float
    top_reasons: list[dict]
    expected_delivery_days: float | None = Field(None, description="Delivery-days model (goal G3)")
    expected_days_late: float | None = Field(None, description="Expected days after the promised date (0 = on time)")
    suggested_actions: list[dict] = Field(default_factory=list, description="Advice only; a person decides")
    model: str


def load() -> None:
    meta = json.loads((MODEL_DIR / "dri_late_classifier.json").read_text())
    model = joblib.load(MODEL_DIR / "dri_late_classifier.joblib")
    post = MODEL_DIR / "dri_fairness_postprocessor.joblib"
    STATE["postprocessor"] = joblib.load(post) if post.exists() else None
    reg = MODEL_DIR / "dri_delivery_days_regressor.joblib"
    try:
        STATE["regressor"] = joblib.load(reg) if reg.exists() else None
    except ModuleNotFoundError:  # xgboost not installed: serve without the delivery-days estimate
        STATE["regressor"] = None
    STATE["store"] = ExplanationStore()
    MODEL_INFO.labels(meta["model_name"], str(meta["version"]), meta["algorithm"]).set(1)
    STATE.update(meta=meta, model=model, features=meta["features"],
                 explainer=Explainer(model, None, meta["features"], mean_encoded=meta.get("encoded_mean")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load()
    yield


app = FastAPI(title="Delivery Risk Intelligence API", version="1.1.0", lifespan=lifespan,
              description="Late-delivery risk for Olist orders at the moment of purchase (BAN6800). "
                          "Shadow mode: scores are advice for review, not decisions.")
app.mount("/metrics", make_asgi_app())


def to_frame(order: Order) -> pd.DataFrame:
    d = order.model_dump()
    if d["freight_ratio"] is None:
        d["freight_ratio"] = d["freight_total"] / d["price_total"]
    if d["promise_slack_days"] is None and d["state_delivery_days_30d"] is not None:
        d["promise_slack_days"] = d["promised_days"] - d["state_delivery_days_30d"]
    if d["is_boleto"] is None:
        d["is_boleto"] = int(d["payment_type"] == "boleto")
    if d["seller_size"] is None:
        n = d["seller_hist_orders"]
        d["seller_size"] = "new" if n == 0 else ("small" if n < 20 else "large")
    df = pd.DataFrame([d])[STATE["features"]]
    for c in STATE["meta"]["numeric"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in STATE["meta"]["categorical"]:
        df[c] = df[c].astype("object")
    return df


@app.get("/")
def root() -> dict:
    return {"service": "Delivery Risk Intelligence API", "docs": "/docs", "health": "/health",
            "predict": "POST /predict", "audit": "GET /explanations/{prediction_id}", "metrics": "/metrics",
            "status": "shadow mode: scores are advice for review, not decisions"}


@app.get("/health")
def health() -> dict:
    m = STATE.get("meta", {})
    return {"status": "ok" if "model" in STATE else "loading", "model": m.get("model_name"),
            "version": m.get("version"), "algorithm": m.get("algorithm"),
            "delivery_days_model": STATE.get("regressor") is not None,
            "explanations_stored": STATE["store"].count() if "store" in STATE else 0}


@app.post("/predict", response_model=Prediction)
def predict(order: Order) -> Prediction:
    if "model" not in STATE:
        raise HTTPException(503, "Model not loaded")
    t0 = time.perf_counter()
    X = to_frame(order)
    p = float(STATE["model"].predict_proba(X)[0, 1])
    thr = STATE["meta"]["threshold"]
    flagged = p >= thr
    band = "high" if flagged else ("medium" if p >= thr * 0.6 else "low")
    fair = None
    if STATE.get("postprocessor") is not None:
        fair = bool(STATE["postprocessor"].predict(X, sensitive_features=X["customer_region"], random_state=0)[0])
    days = late_days = None
    if STATE.get("regressor") is not None:
        days = round(float(STATE["regressor"].predict(X)[0]), 1)
        late_days = round(max(0.0, days - order.promised_days), 1)
    reasons = STATE["explainer"].reasons(X)
    result = {"late_probability": round(p, 4), "flagged": flagged, "flagged_fairness_adjusted": fair,
              "risk_band": band, "threshold": round(thr, 4), "top_reasons": reasons,
              "expected_delivery_days": days, "expected_days_late": late_days,
              "suggested_actions": suggest(reasons, flagged or bool(fair), late_days),
              "model": f"{STATE['meta']['model_name']} v{STATE['meta']['version']} ({STATE['meta']['algorithm']})"}
    pid = STATE["store"].record(order.model_dump(), result)
    LATENCY.observe(time.perf_counter() - t0)
    SCORE.observe(p)
    PREDICTIONS.labels(str(flagged).lower(), order.customer_region).inc()
    return Prediction(prediction_id=pid, **result)


@app.get("/explanations/{prediction_id}")
def explanation(prediction_id: str) -> dict:
    """The stored record behind one score, for audit or a seller's challenge."""
    rec = STATE["store"].get(prediction_id)
    if rec is None:
        raise HTTPException(404, "No explanation stored with that id")
    return rec


@app.get("/explanations")
def explanations(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    """Most recent stored explanations (audit view)."""
    return STATE["store"].recent(limit)
