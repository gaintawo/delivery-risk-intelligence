"""Stage 1-2: data sources and ingestion.

Downloads the nine public Olist CSV files if they are missing, then loads each
one with an explicit schema so that types are never guessed.
"""
from __future__ import annotations

import urllib.request

import pandas as pd

from . import config
from .audit import audit

# name -> (file, expected columns, date columns, string columns)
TABLES: dict[str, dict] = {
    "orders": {
        "file": "olist_orders_dataset.csv",
        "dates": ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                  "order_delivered_customer_date", "order_estimated_delivery_date"],
        "strings": ["order_id", "customer_id", "order_status"],
    },
    "items": {
        "file": "olist_order_items_dataset.csv",
        "dates": ["shipping_limit_date"],
        "strings": ["order_id", "product_id", "seller_id"],
    },
    "payments": {
        "file": "olist_order_payments_dataset.csv",
        "dates": [],
        "strings": ["order_id", "payment_type"],
    },
    "reviews": {
        "file": "olist_order_reviews_dataset.csv",
        "dates": ["review_creation_date", "review_answer_timestamp"],
        "strings": ["review_id", "order_id"],
    },
    "customers": {
        "file": "olist_customers_dataset.csv",
        "dates": [],
        "strings": ["customer_id", "customer_unique_id", "customer_zip_code_prefix",
                    "customer_city", "customer_state"],
    },
    "sellers": {
        "file": "olist_sellers_dataset.csv",
        "dates": [],
        "strings": ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
    },
    "products": {
        "file": "olist_products_dataset.csv",
        "dates": [],
        "strings": ["product_id", "product_category_name"],
    },
    "geolocation": {
        "file": "olist_geolocation_dataset.csv",
        "dates": [],
        "strings": ["geolocation_zip_code_prefix", "geolocation_city", "geolocation_state"],
    },
    "category_translation": {
        "file": "product_category_name_translation.csv",
        "dates": [],
        "strings": ["product_category_name", "product_category_name_english"],
    },
}


def download_missing() -> list[str]:
    """Fetch any raw file that is not already in data/raw/."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetched = []
    for spec in TABLES.values():
        target = config.RAW_DIR / spec["file"]
        if target.exists():
            continue
        urllib.request.urlretrieve(f"{config.SOURCE_URL}/{spec['file']}", target)
        audit("download", spec["file"], purpose="Acquire public Olist source file",
              path=target, source=config.SOURCE_URL)
        fetched.append(spec["file"])
    return fetched


def load_table(name: str) -> pd.DataFrame:
    spec = TABLES[name]
    path = config.RAW_DIR / spec["file"]
    df = pd.read_csv(path, dtype={c: "string" for c in spec["strings"]}, low_memory=False)
    for c in spec["dates"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    audit("read", name, purpose="Ingest raw source table", rows=len(df), cols=df.shape[1], path=path)
    return df


def load_all() -> dict[str, pd.DataFrame]:
    return {name: load_table(name) for name in TABLES}
