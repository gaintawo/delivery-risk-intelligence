"""Stage 3: cleaning.

One pure function per table: DataFrame in, (DataFrame, report) out. Each
report counts exactly what was changed, so every cleaning decision is visible
in the lineage file and testable in tests/test_clean.py.
"""
from __future__ import annotations

import pandas as pd

from . import config

# Columns dropped at the door because they are personal or not needed (ER4,
# data minimisation). See docs/data_anonymization_plan.md.
DROP_AT_INGEST = {
    "customers": ["customer_city"],
    "sellers": ["seller_city"],
    "reviews": ["review_comment_title", "review_comment_message"],
    "geolocation": ["geolocation_city"],
}


def _drop_pii(name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in DROP_AT_INGEST.get(name, []) if c in df.columns]
    return df.drop(columns=cols), cols


def _zip5(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.zfill(5)


def clean_orders(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep = {"rows_in": len(df)}
    df = df.drop_duplicates()
    rep["duplicate_order_ids"] = int(df["order_id"].duplicated().sum())
    df = df.drop_duplicates("order_id")
    df["order_status"] = df["order_status"].str.lower().str.strip()
    # Timestamps that run backwards are kept but flagged; they are a known
    # data-quality issue (RAID R01), not a reason to lose the order.
    approved_before_purchase = df["order_approved_at"] < df["order_purchase_timestamp"]
    carrier_before_approved = df["order_delivered_carrier_date"] < df["order_approved_at"]
    delivered_before_carrier = df["order_delivered_customer_date"] < df["order_delivered_carrier_date"]
    df["ts_anomaly"] = (approved_before_purchase | carrier_before_approved | delivered_before_carrier).astype("int8")
    rep["ts_anomaly"] = int(df["ts_anomaly"].sum())
    rep["delivered_missing_date"] = int(
        ((df["order_status"] == "delivered") & df["order_delivered_customer_date"].isna()).sum())
    rep["rows_out"] = len(df)
    return df, rep


def clean_items(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep = {"rows_in": len(df)}
    df = df.drop_duplicates(["order_id", "order_item_id"])
    bad = (df["price"] <= 0) | (df["freight_value"] < 0)
    rep["invalid_price_or_freight"] = int(bad.sum())
    df = df[~bad]
    rep["rows_out"] = len(df)
    return df, rep


def clean_payments(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep = {"rows_in": len(df)}
    df = df.drop_duplicates()
    df["payment_type"] = df["payment_type"].str.lower().str.strip()
    rep["not_defined_type"] = int((df["payment_type"] == "not_defined").sum())
    df = df[df["payment_type"] != "not_defined"]
    df.loc[df["payment_installments"] < 1, "payment_installments"] = 1
    rep["rows_out"] = len(df)
    return df, rep


def clean_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep = {"rows_in": len(df)}
    df, dropped = _drop_pii("reviews", df)
    rep["pii_columns_dropped"] = dropped
    # Some orders have several reviews and some reviews repeat: keep the latest
    # answer per order so the outcome measure (G4) is one value per order.
    df = df.sort_values("review_answer_timestamp")
    rep["extra_reviews_per_order"] = int(df["order_id"].duplicated(keep="last").sum())
    df = df.drop_duplicates("order_id", keep="last")
    df = df[df["review_score"].between(1, 5)]
    rep["rows_out"] = len(df)
    return df, rep


def clean_customers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep = {"rows_in": len(df)}
    df, dropped = _drop_pii("customers", df)
    rep["pii_columns_dropped"] = dropped
    df = df.drop_duplicates("customer_id")
    df["customer_zip_code_prefix"] = _zip5(df["customer_zip_code_prefix"])
    df["customer_state"] = df["customer_state"].str.upper().str.strip()
    rep["unknown_state"] = int((~df["customer_state"].isin(config.STATE_TO_REGION)).sum())
    rep["rows_out"] = len(df)
    return df, rep


def clean_sellers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep = {"rows_in": len(df)}
    df, dropped = _drop_pii("sellers", df)
    rep["pii_columns_dropped"] = dropped
    df = df.drop_duplicates("seller_id")
    df["seller_zip_code_prefix"] = _zip5(df["seller_zip_code_prefix"])
    df["seller_state"] = df["seller_state"].str.upper().str.strip()
    rep["rows_out"] = len(df)
    return df, rep


def clean_products(df: pd.DataFrame, translation: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep = {"rows_in": len(df)}
    # Fix the misspelt source column names ("lenght").
    df = df.rename(columns=lambda c: c.replace("lenght", "length"))
    df = df.drop_duplicates("product_id")
    rep["missing_category"] = int(df["product_category_name"].isna().sum())
    df["product_category_name"] = df["product_category_name"].fillna("unknown")
    df = df.merge(translation, on="product_category_name", how="left")
    df["product_category"] = df["product_category_name_english"].fillna(df["product_category_name"])
    # Missing or zero size/weight: impute the category median, then the global
    # median, and keep a flag so the model can tell imputed values apart.
    size_cols = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
    df[size_cols] = df[size_cols].where(df[size_cols] > 0)
    df["size_imputed"] = df[size_cols].isna().any(axis=1).astype("int8")
    rep["size_imputed"] = int(df["size_imputed"].sum())
    for c in size_cols:
        df[c] = df[c].fillna(df.groupby("product_category")[c].transform("median")).fillna(df[c].median())
    keep = ["product_id", "product_category", "size_imputed"] + size_cols
    rep["rows_out"] = len(df)
    return df[keep], rep


def clean_geolocation(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse ~1M coordinate rows to one median point per postcode prefix."""
    rep = {"rows_in": len(df)}
    df, dropped = _drop_pii("geolocation", df)
    rep["pii_columns_dropped"] = dropped
    df = df.drop_duplicates()
    rep["rows_after_dedupe"] = len(df)
    lat_ok = df["geolocation_lat"].between(*config.BRAZIL_LAT)
    lng_ok = df["geolocation_lng"].between(*config.BRAZIL_LNG)
    rep["outside_brazil"] = int((~(lat_ok & lng_ok)).sum())
    df = df[lat_ok & lng_ok].copy()
    df["zip"] = _zip5(df["geolocation_zip_code_prefix"])
    out = (df.groupby("zip")
             .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"),
                  state=("geolocation_state", lambda s: s.mode().iat[0]))
             .reset_index())
    rep["rows_out"] = len(out)
    return out, rep


def clean_all(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    out, reports = {}, {}
    for name, fn in (("orders", clean_orders), ("items", clean_items), ("payments", clean_payments),
                     ("reviews", clean_reviews), ("customers", clean_customers),
                     ("sellers", clean_sellers), ("geolocation", clean_geolocation)):
        out[name], reports[name] = fn(raw[name].copy())
    out["products"], reports["products"] = clean_products(raw["products"].copy(),
                                                          raw["category_translation"])
    return out, reports
