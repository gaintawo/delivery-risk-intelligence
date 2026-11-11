"""Stage 5: transformation into model inputs.

Every column produced here carries an availability tag from the data
dictionary: P (known at purchase), H (known at carrier hand-over) or
O (outcome, never a model input). FEATURE_TAGS is the single source of truth
that the leakage test checks against.
"""
from __future__ import annotations

import pandas as pd

from src.pipeline import config

FEATURE_TAGS: dict[str, str] = {
    # labels and outcomes
    "is_late": "O", "days_late": "O", "delivery_days": "O", "review_score": "O",
    "is_delivered": "O",
    # known at purchase
    "promised_days": "P", "purchase_dow": "P", "purchase_month": "P", "purchase_hour": "P",
    "is_peak_season": "P", "distance_km": "P", "interstate": "P", "customer_region": "P",
    "customer_state": "P", "seller_state": "P", "n_items": "P", "n_sellers": "P",
    "price_total": "P", "freight_total": "P", "freight_ratio": "P", "weight_total_g": "P",
    "volume_total_cm3": "P", "any_size_imputed": "P", "primary_category": "P",
    "payment_type": "P", "is_boleto": "P", "payment_installments": "P",
    "n_payment_methods": "P", "seller_hist_orders": "P", "seller_hist_late_rate": "P",
    "seller_hist_handling_days": "P", "seller_size": "P",
    # known at hand-over (stage-2 scoring only)
    "approval_lag_h": "H", "handling_days": "H", "handover_vs_limit_days": "H",
}


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    delivered = (df["order_status"] == "delivered") & df["order_delivered_customer_date"].notna()
    df["is_delivered"] = delivered.astype("int8")
    late = df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    df["is_late"] = late.where(delivered).astype("Int8")
    gap = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.total_seconds() / 86400
    df["days_late"] = gap.clip(lower=0).where(delivered).round(2)
    span = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400
    df["delivery_days"] = span.where(delivered).round(2)
    return df


def add_purchase_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["order_purchase_timestamp"]
    df["promised_days"] = ((df["order_estimated_delivery_date"] - t).dt.total_seconds() / 86400).round(2)
    df["purchase_dow"] = t.dt.dayofweek.astype("int8")
    df["purchase_month"] = t.dt.month.astype("int8")
    df["purchase_hour"] = t.dt.hour.astype("int8")
    # Black Friday week (last week of November) to Christmas Eve.
    nov_peak = (t.dt.month == 11) & (t.dt.day >= 20)
    dec_peak = (t.dt.month == 12) & (t.dt.day <= 24)
    df["is_peak_season"] = (nov_peak | dec_peak).astype("int8")
    df["interstate"] = (df["customer_state"] != df["seller_state"]).astype("Int8")  # NA when no seller
    df["customer_region"] = df["customer_state"].map(config.STATE_TO_REGION)
    df["freight_ratio"] = (df["freight_total"] / df["price_total"]).round(4)
    df["is_boleto"] = (df["payment_type"] == "boleto").fillna(False).astype("int8")
    return df


def add_handover_features(df: pd.DataFrame) -> pd.DataFrame:
    df["approval_lag_h"] = ((df["order_approved_at"] - df["order_purchase_timestamp"])
                            .dt.total_seconds() / 3600).round(2)
    df["handling_days"] = ((df["order_delivered_carrier_date"] - df["order_approved_at"])
                           .dt.total_seconds() / 86400).round(2)
    df["handover_vs_limit_days"] = ((df["order_delivered_carrier_date"] - df["shipping_limit_date"])
                                    .dt.total_seconds() / 86400).round(2)
    # A negative duration means the timestamps are out of order (flagged in
    # ts_anomaly); the value is unusable, so it becomes missing, not zero.
    for c in ("approval_lag_h", "handling_days"):
        df.loc[df[c] < 0, c] = float("nan")
    return df


def add_seller_history(df: pd.DataFrame) -> pd.DataFrame:
    """Seller track record as it stood at the moment of each purchase.

    Only orders already *delivered* before this order was placed count, so the
    feature never uses information from the future (FR2, RAID R02).
    """
    events = df.loc[df["is_delivered"] == 1, ["seller_id", "order_delivered_customer_date",
                                              "is_late", "handling_days"]].copy()
    events = events.dropna(subset=["seller_id"]).sort_values("order_delivered_customer_date")
    events["is_late"] = events["is_late"].astype("int32")  # avoid Int8 overflow in cumsum
    g = events.groupby("seller_id")
    events["cum_orders"] = g.cumcount() + 1
    events["cum_late"] = g["is_late"].cumsum()
    events["cum_handling"] = g["handling_days"].transform(lambda s: s.fillna(0).cumsum())
    events = events.rename(columns={"order_delivered_customer_date": "known_at"})

    left = df[["order_id", "seller_id", "order_purchase_timestamp"]].dropna().sort_values(
        "order_purchase_timestamp")
    hist = pd.merge_asof(left, events[["seller_id", "known_at", "cum_orders", "cum_late", "cum_handling"]]
                         .sort_values("known_at"),
                         left_on="order_purchase_timestamp", right_on="known_at",
                         by="seller_id", direction="backward", allow_exact_matches=False)
    hist["seller_hist_orders"] = hist["cum_orders"].fillna(0).astype("int32")
    n = hist["cum_orders"]
    hist["seller_hist_late_rate"] = (hist["cum_late"] / n).round(4)
    hist["seller_hist_handling_days"] = (hist["cum_handling"] / n).round(2)
    hist["seller_size"] = pd.cut(hist["seller_hist_orders"], [-1, 0, 19, 10**9],
                                 labels=["new", "small", "large"]).astype("string")
    cols = ["order_id", "seller_hist_orders", "seller_hist_late_rate", "seller_hist_handling_days",
            "seller_size"]
    return df.merge(hist[cols], on="order_id", how="left")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_labels(df)
    df = add_purchase_features(df)
    df = add_handover_features(df)
    df = add_seller_history(df)
    return df
