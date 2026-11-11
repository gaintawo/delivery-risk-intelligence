"""Stage 4: integration.

Joins the eight cleaned tables into one row per order. Item-level and
payment-level tables are aggregated to order level first, so no join can
multiply rows; the final assertion checks that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _items_per_order(items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    it = items.merge(products, on="product_id", how="left")
    it["volume_cm3"] = it["product_length_cm"] * it["product_height_cm"] * it["product_width_cm"]
    # The "primary" seller and category are those of the most expensive item.
    top = (it.sort_values(["order_id", "price"], ascending=[True, False])
             .drop_duplicates("order_id")[["order_id", "seller_id", "product_category"]]
             .rename(columns={"product_category": "primary_category"}))
    agg = it.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
        price_total=("price", "sum"),
        freight_total=("freight_value", "sum"),
        weight_total_g=("product_weight_g", "sum"),
        volume_total_cm3=("volume_cm3", "sum"),
        any_size_imputed=("size_imputed", "max"),
        shipping_limit_date=("shipping_limit_date", "max"),
    ).reset_index()
    return agg.merge(top, on="order_id", how="left")


def _payments_per_order(payments: pd.DataFrame) -> pd.DataFrame:
    main = (payments.sort_values(["order_id", "payment_value"], ascending=[True, False])
                    .drop_duplicates("order_id")[["order_id", "payment_type"]])
    agg = payments.groupby("order_id").agg(
        n_payment_methods=("payment_type", "nunique"),
        payment_installments=("payment_installments", "max"),
        payment_value_total=("payment_value", "sum"),
    ).reset_index()
    return agg.merge(main, on="order_id", how="left")


def _haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _attach_coordinates(df: pd.DataFrame, geo: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, int]:
    """Add lat/lng for customer or seller; fall back to the state's median point."""
    zip_col, state_col = f"{prefix}_zip_code_prefix", f"{prefix}_state"
    df = df.merge(geo[["zip", "lat", "lng"]].rename(columns={"zip": zip_col, "lat": f"{prefix}_lat",
                                                             "lng": f"{prefix}_lng"}),
                  on=zip_col, how="left")
    centroids = geo.groupby("state")[["lat", "lng"]].median()
    missing = df[f"{prefix}_lat"].isna()
    df.loc[missing, f"{prefix}_lat"] = df.loc[missing, state_col].map(centroids["lat"])
    df.loc[missing, f"{prefix}_lng"] = df.loc[missing, state_col].map(centroids["lng"])
    return df, int(missing.sum())


def build_order_table(c: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    rep = {}
    df = c["orders"].merge(c["customers"], on="customer_id", how="left")
    df = df.merge(_items_per_order(c["items"], c["products"]), on="order_id", how="left")
    rep["orders_without_items"] = int(df["n_items"].isna().sum())
    df = df.merge(c["sellers"], on="seller_id", how="left")
    df = df.merge(_payments_per_order(c["payments"]), on="order_id", how="left")
    rep["orders_without_payment"] = int(df["payment_type"].isna().sum())
    df = df.merge(c["reviews"][["order_id", "review_score"]], on="order_id", how="left")

    df, rep["customer_geo_fallback"] = _attach_coordinates(df, c["geolocation"], "customer")
    df, rep["seller_geo_fallback"] = _attach_coordinates(df, c["geolocation"], "seller")
    df["distance_km"] = _haversine_km(df["customer_lat"], df["customer_lng"],
                                      df["seller_lat"], df["seller_lng"]).round(1)

    assert df["order_id"].is_unique, "integration multiplied orders"
    assert len(df) == len(c["orders"]), "integration lost orders"
    rep["rows_out"] = len(df)
    return df, rep
