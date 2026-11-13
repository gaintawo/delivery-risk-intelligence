import pandas as pd

from src.features.build import FEATURE_TAGS, add_seller_history, build_features
from src.pipeline import clean, integrate


def _integrated(raw):
    c, _ = clean.clean_all(raw)
    df, _ = integrate.build_order_table(c)
    return df


def test_integration_keeps_one_row_per_order(raw):
    df = _integrated(raw)
    assert df["order_id"].is_unique and len(df) == len(raw["orders"])


def test_late_label_matches_dates(raw):
    df = build_features(_integrated(raw))
    expected = (df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]).astype(int)
    assert (df["is_late"].astype(int) == expected).all()


def test_seller_history_uses_only_earlier_deliveries():
    t = pd.Timestamp
    df = pd.DataFrame({
        "order_id": ["a", "b", "c"],
        "seller_id": ["s1", "s1", "s1"],
        "order_purchase_timestamp": [t("2017-01-01"), t("2017-01-05"), t("2017-01-20")],
        "order_delivered_customer_date": [t("2017-01-10"), t("2017-01-25"), t("2017-01-30")],
        "is_delivered": [1, 1, 1],
        "is_late": pd.array([1, 0, 0], dtype="Int8"),
        "handling_days": [2.0, 3.0, 1.0],
    })
    out = add_seller_history(df).set_index("order_id")
    # b was bought before a was delivered: the seller had no known history yet.
    assert out.loc["b", "seller_hist_orders"] == 0
    assert pd.isna(out.loc["b", "seller_hist_late_rate"])
    # c was bought after a (late) was delivered but before b was delivered.
    assert out.loc["c", "seller_hist_orders"] == 1
    assert out.loc["c", "seller_hist_late_rate"] == 1.0


def test_negative_durations_become_missing(raw):
    df = _integrated(raw)
    df.loc[0, "order_delivered_carrier_date"] = df.loc[0, "order_approved_at"] - pd.Timedelta(days=1)
    out = build_features(df)
    assert pd.isna(out.loc[out["order_id"] == df.loc[0, "order_id"], "handling_days"]).all()


def test_every_output_feature_is_tagged(raw):
    df = build_features(_integrated(raw))
    assert set(FEATURE_TAGS) <= set(df.columns)
    assert set(FEATURE_TAGS.values()) <= {"P", "H", "O"}
    # Outcome columns must never be tagged as model inputs.
    for col in ("is_late", "days_late", "delivery_days", "review_score"):
        assert FEATURE_TAGS[col] == "O"


def test_market_context_counts_only_deliveries_before_purchase():
    from src.features.build import _window_stats
    t = pd.Timestamp
    orders = pd.DataFrame({"order_id": ["late_buy", "early_buy"],
                           "order_purchase_timestamp": [t("2018-03-01"), t("2018-01-20")],
                           "customer_state": ["SP", "SP"]})
    events = pd.DataFrame({"known_at": [t("2017-12-10"), t("2018-01-05"), t("2018-01-19"), t("2018-01-25")],
                           "customer_state": ["SP"] * 4, "late": [1.0, 1.0, 0.0, 1.0],
                           "days": [10.0, 20.0, 30.0, 5.0]})
    w = _window_stats(orders, events, "customer_state", 30, ["late", "days"])
    # early_buy: only the deliveries of 5 Jan (late) and 19 Jan (on time) fall in its 30-day window
    assert w.loc["early_buy", "n"] == 2 and w.loc["early_buy", "cum_late"] == 1
    # late_buy: the 25 Jan delivery is more than 30 days before 1 March
    assert w.loc["late_buy", "n"] == 0
