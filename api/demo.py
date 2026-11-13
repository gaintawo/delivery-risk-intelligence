"""API demo: score the three Module 5 example orders through POST /predict.

    uvicorn api.main:app                       # terminal 1
    python -m api.demo                         # terminal 2 (or: python -m api.demo --in-process)

The orders come from the de-identified test-period file used by the dashboard.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["promised_days", "purchase_dow", "purchase_month", "purchase_hour", "is_peak_season", "distance_km",
          "interstate", "n_items", "n_sellers", "price_total", "freight_total", "weight_total_g", "volume_total_cm3",
          "any_size_imputed", "payment_installments", "n_payment_methods", "seller_hist_orders",
          "seller_hist_late_rate", "seller_hist_handling_days", "state_late_rate_30d", "state_delivery_days_30d",
          "market_late_rate_14d", "seller_orders_7d", "customer_state", "seller_state", "customer_region",
          "primary_category", "payment_type"]
INTS = {"purchase_dow", "purchase_month", "purchase_hour", "is_peak_season", "interstate", "n_items", "n_sellers",
        "any_size_imputed", "payment_installments", "n_payment_methods", "seller_hist_orders", "seller_orders_7d"}


def example_orders() -> list[tuple[str, dict, int]]:
    d = pd.read_csv(ROOT / "dashboard" / "data" / "test_scored.csv.gz", dtype={"example": "string"})
    out = []
    for _, r in d[d["example"].notna()].sort_values("example").iterrows():
        body = {}
        for f in FIELDS:
            v = r[f]
            if isinstance(v, float) and math.isnan(v):
                v = None
            elif f in INTS:
                v = int(v)
            body[f] = v
        out.append((r["example"], body, int(r["is_late"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--in-process", action="store_true", help="run the API inside this script (no server needed)")
    a = ap.parse_args()
    if a.in_process:
        from fastapi.testclient import TestClient

        from api.main import app
        client = TestClient(app).__enter__()
    else:
        import httpx
        client = httpx.Client(base_url=a.url, timeout=10)

    print("GET /health ->", client.get("/health").json())
    for name, body, late in example_orders():
        r = client.post("/predict", json=body)
        r.raise_for_status()
        p = r.json()
        print(f"\n=== Order {name} ({body['customer_state']}, {body['customer_region']}; "
              f"actually {'late' if late else 'on time'}) ===")
        print(f"POST /predict -> risk {p['late_probability']:.2f} · band {p['risk_band']} · "
              f"flagged {p['flagged']} · fairness-adjusted flag {p['flagged_fairness_adjusted']}")
        for i, why in enumerate(p["top_reasons"], 1):
            print(f"  reason {i}: {why['reason']} = {why['value']} (+{why['impact_log_odds']:.2f})")
    print("\nFull response for the last order:\n" + json.dumps(p, indent=2))


if __name__ == "__main__":
    main()
