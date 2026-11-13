"""FastAPI /predict tests. Skipped if the trained model file is not present."""
from pathlib import Path

import pytest

MODEL = Path(__file__).resolve().parents[1] / "models" / "dri_late_classifier.joblib"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="trained model not present")

ORDER = {"promised_days": 18, "purchase_dow": 4, "purchase_month": 11, "purchase_hour": 21,
         "is_peak_season": 1, "distance_km": 2110, "interstate": 1, "price_total": 129.9,
         "freight_total": 38.5, "weight_total_g": 1800, "volume_total_cm3": 9000,
         "seller_hist_orders": 42, "seller_hist_late_rate": 0.19, "seller_hist_handling_days": 3.4,
         "state_late_rate_30d": 0.09, "state_delivery_days_30d": 19.6, "market_late_rate_14d": 0.07,
         "seller_orders_7d": 12,
         "customer_state": "BA", "seller_state": "SP", "customer_region": "Northeast",
         "primary_category": "health_beauty", "payment_type": "boleto"}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_probability_flag_and_reasons(client):
    r = client.post("/predict", json=ORDER)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["late_probability"] <= 1
    assert body["flagged"] == (body["late_probability"] >= body["threshold"])
    assert body["risk_band"] in {"low", "medium", "high"}
    assert len(body["top_reasons"]) <= 3
    assert body["flagged_fairness_adjusted"] in (True, False, None)


def test_new_seller_without_history_is_accepted(client):
    order = {**ORDER, "seller_hist_orders": 0, "seller_hist_late_rate": None, "seller_hist_handling_days": None}
    r = client.post("/predict", json=order)
    assert r.status_code == 200


def test_market_context_is_optional(client):
    order = {k: v for k, v in ORDER.items() if not k.startswith(("state_", "market_"))}
    assert client.post("/predict", json=order).status_code == 200


def test_invalid_region_is_rejected(client):
    r = client.post("/predict", json={**ORDER, "customer_region": "Atlantis"})
    assert r.status_code == 422


def test_riskier_order_scores_higher(client):
    safe = {**ORDER, "promised_days": 40, "distance_km": 50, "interstate": 0, "seller_state": "BA",
            "seller_hist_late_rate": 0.0, "is_peak_season": 0, "purchase_month": 6, "state_delivery_days_30d": 9.0}
    risky = {**ORDER, "promised_days": 8}
    p_safe = client.post("/predict", json=safe).json()["late_probability"]
    p_risky = client.post("/predict", json=risky).json()["late_probability"]
    assert p_risky > p_safe
