"""Modelling dataset for the late-delivery classifier (Module 4).

Unit of analysis : one delivered order.
Prediction point : the moment of purchase (stage-1 scoring, Module 2).
Target           : is_late = 1 if the order reached the customer after the
                   promised (estimated) delivery date.
Horizon          : the promise window itself - median 23 days ahead.

Only features tagged P (known at purchase) in src/features/build.py are
allowed; select_features() enforces this, so outcome or hand-over fields can
never reach the model (leakage prevention, FR2).
"""
from __future__ import annotations

import pandas as pd

from src.features.build import FEATURE_TAGS
from src.pipeline import config

TARGET = "is_late"
REGRESSION_TARGET = "delivery_days"
TIME_COL = "order_purchase_timestamp"
SENSITIVE = ["customer_region", "seller_size"]

CATEGORICAL = ["customer_state", "seller_state", "customer_region", "primary_category",
               "payment_type", "seller_size"]
NUMERIC = ["promised_days", "purchase_dow", "purchase_month", "purchase_hour", "is_peak_season",
           "distance_km", "interstate", "n_items", "n_sellers", "price_total", "freight_total",
           "freight_ratio", "weight_total_g", "volume_total_cm3", "any_size_imputed",
           "is_boleto", "payment_installments", "n_payment_methods", "seller_hist_orders",
           "seller_hist_late_rate", "seller_hist_handling_days",
           "state_late_rate_30d", "state_delivery_days_30d", "promise_slack_days",
           "market_late_rate_14d", "seller_orders_7d"]
FEATURES = NUMERIC + CATEGORICAL
# Geography identity columns - removed in the "unaware" variant (proxy test, ER2).
GEO_IDENTITY = ["customer_state", "seller_state", "customer_region"]

# Time-aware split (purchase date). 2016 is dropped: 267 orders, launch period.
SPLITS = {
    "train": ("2017-01-01", "2018-03-01"),
    "validation": ("2018-03-01", "2018-06-01"),
    "test": ("2018-06-01", "2018-09-01"),
}


def check_no_leakage(features: list[str]) -> None:
    bad = [f for f in features if FEATURE_TAGS.get(f) != "P"]
    if bad:
        raise ValueError(f"Not known at purchase, cannot be a model input: {bad}")


def load_modelling_frame(path=None) -> pd.DataFrame:
    path = path or config.PROCESSED_DIR / "orders_model_ready.parquet"
    df = pd.read_parquet(path)
    df = df[(df["is_delivered"] == 1) & df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(int)
    for c in CATEGORICAL:
        df[c] = df[c].astype("object").where(df[c].notna(), "missing")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df.sort_values(TIME_COL).reset_index(drop=True)


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for name, (lo, hi) in SPLITS.items():
        m = (df[TIME_COL] >= lo) & (df[TIME_COL] < hi)
        out[name] = df[m].reset_index(drop=True)
    return out


check_no_leakage(FEATURES)
