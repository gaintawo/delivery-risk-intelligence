"""Stage 6: anonymisation and PII removal.

Implements docs/data_anonymization_plan.md:
  * pseudonymise identifiers with a salted SHA-256 hash (the salt lives outside Git),
  * delete direct location data once distance has been derived from it,
  * keep only the columns the model and the fairness checks need (allow-list).
Anything not on the allow-list never reaches data/processed/.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from src.features.build import FEATURE_TAGS

from . import config
from .audit import audit

PSEUDONYMISE = ["order_id", "customer_unique_id", "seller_id"]

# Quasi-identifiers and raw location: used to derive features, then deleted.
DELETE_AFTER_USE = [
    "customer_id", "customer_zip_code_prefix", "seller_zip_code_prefix",
    "customer_lat", "customer_lng", "seller_lat", "seller_lng",
]

# Timestamps kept for time-based validation splits in Module 4.
KEEP_TIMESTAMPS = ["order_purchase_timestamp"]

OUTPUT_COLUMNS = (["order_key", "customer_key", "seller_key"] + KEEP_TIMESTAMPS
                  + ["order_status", "ts_anomaly"] + list(FEATURE_TAGS))


def pseudonymise(values: pd.Series, salt: str = config.HASH_SALT) -> pd.Series:
    """Stable, salted, one-way 16-character key. Same input + salt = same key."""
    def h(v):
        if pd.isna(v):
            return pd.NA
        return hashlib.sha256(f"{salt}|{v}".encode()).hexdigest()[:16]
    return values.map(h).astype("string")


def anonymise(df: pd.DataFrame, salt: str = config.HASH_SALT) -> tuple[pd.DataFrame, dict]:
    rep = {"rows_in": len(df), "cols_in": df.shape[1]}
    out = df.copy()
    out["order_key"] = pseudonymise(out["order_id"], salt)
    out["customer_key"] = pseudonymise(out["customer_unique_id"], salt)
    out["seller_key"] = pseudonymise(out["seller_id"], salt)
    audit("pii_pseudonymise", "orders_integrated", purpose="Replace identifiers with salted hashes",
          rows=len(out), columns=PSEUDONYMISE)

    removed = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    out = out[OUTPUT_COLUMNS]
    audit("pii_remove", "orders_integrated", purpose="Drop identifiers, location and unused columns",
          rows=len(out), cols=out.shape[1], removed=removed)
    rep.update(cols_out=out.shape[1], columns_removed=removed)
    return out, rep


def k_anonymity(df: pd.DataFrame, quasi: list[str]) -> dict:
    """Smallest group size over the quasi-identifiers left in the output."""
    sizes = df.groupby(quasi, dropna=False).size()
    return {"quasi_identifiers": quasi, "k_min": int(sizes.min()),
            "groups": len(sizes), "groups_below_5": int((sizes < 5).sum()),
            "rows_in_groups_below_5": int(sizes[sizes < 5].sum())}
