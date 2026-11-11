"""Stage 9: outputs.

1. data/processed/orders_model_ready.parquet - one row per order, pseudonymised;
   the training table for Module 4 (restricted access, see governance framework).
2. Warehouse table orders_model_ready + view v_training_delivered in the
   database at DATABASE_URL (SQLite by default, PostgreSQL in production).
3. data/processed/dashboard_feed.csv - region x month x category aggregates for
   the Module 5 dashboard, with every cell of fewer than 5 orders suppressed,
   so it can be shared more widely.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from . import config
from .audit import audit

K_SUPPRESS = 5


def dashboard_feed(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = df.assign(month=df["order_purchase_timestamp"].dt.to_period("M").astype(str))
    feed = (d.groupby(["customer_region", "month", "primary_category"], dropna=False)
              .agg(orders=("order_key", "count"), delivered=("is_delivered", "sum"),
                   late=("is_late", "sum"), avg_promised_days=("promised_days", "mean"),
                   avg_review=("review_score", "mean"))
              .reset_index())
    small = feed["orders"] < K_SUPPRESS
    orders_suppressed = int(feed.loc[small, "orders"].sum())
    feed.loc[small, ["orders", "delivered", "late", "avg_promised_days", "avg_review"]] = pd.NA
    feed["suppressed"] = small.astype("int8")
    feed["late_rate"] = (feed["late"] / feed["delivered"]).round(4)
    feed["avg_promised_days"] = feed["avg_promised_days"].round(2)
    feed["avg_review"] = feed["avg_review"].round(2)
    return feed, {"cells": len(feed), "cells_suppressed": int(small.sum()),
                  "orders_in_suppressed_cells": orders_suppressed,
                  "share_of_orders_suppressed": round(orders_suppressed / len(df), 4)}


def write_outputs(df: pd.DataFrame) -> dict:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parquet = config.PROCESSED_DIR / "orders_model_ready.parquet"
    df.to_parquet(parquet, index=False)
    audit("write", "orders_model_ready", purpose="Model-ready training table (restricted)",
          rows=len(df), cols=df.shape[1], path=parquet)

    engine = create_engine(config.DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP VIEW IF EXISTS v_training_delivered"))
    df.to_sql("orders_model_ready", engine, if_exists="replace", index=False, chunksize=10000)
    with engine.begin() as conn:
        conn.execute(text("CREATE VIEW v_training_delivered AS "
                          "SELECT * FROM orders_model_ready WHERE is_delivered = 1"))
        n_view = conn.execute(text("SELECT COUNT(*) FROM v_training_delivered")).scalar()
    audit("write", "warehouse.orders_model_ready", purpose="Load warehouse table and training view",
          rows=len(df), cols=df.shape[1], target=config.DATABASE_URL.split("@")[-1], view_rows=n_view)

    feed, frep = dashboard_feed(df)
    feed_path = config.PROCESSED_DIR / "dashboard_feed.csv"
    feed.to_csv(feed_path, index=False)
    audit("write", "dashboard_feed", purpose="Aggregated, small-cell-suppressed dashboard feed",
          rows=len(feed), cols=feed.shape[1], path=feed_path, **frep)
    return {"parquet": str(parquet.relative_to(config.ROOT)), "rows": len(df),
            "training_view_rows": int(n_view), "dashboard_feed": frep}
