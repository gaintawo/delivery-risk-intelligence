import pandas as pd

from src.pipeline import clean


def test_orders_duplicates_removed_and_anomalies_flagged(raw):
    orders = pd.concat([raw["orders"], raw["orders"].head(3)], ignore_index=True)
    orders.loc[0, "order_approved_at"] = orders.loc[0, "order_purchase_timestamp"] - pd.Timedelta(hours=2)
    out, rep = clean.clean_orders(orders)
    assert out["order_id"].is_unique
    assert len(out) == len(raw["orders"])
    assert out["ts_anomaly"].sum() == 1 and rep["ts_anomaly"] == 1


def test_free_text_and_city_columns_dropped(raw):
    reviews, rep = clean.clean_reviews(raw["reviews"])
    assert "review_comment_message" not in reviews and "review_comment_title" not in reviews
    assert rep["pii_columns_dropped"] == ["review_comment_title", "review_comment_message"]
    customers, _ = clean.clean_customers(raw["customers"])
    assert "customer_city" not in customers


def test_one_review_per_order_keeps_latest(raw):
    r = raw["reviews"].head(2).copy()
    r.loc[1, "order_id"] = r.loc[0, "order_id"]
    r.loc[1, "review_answer_timestamp"] = r.loc[0, "review_answer_timestamp"] + pd.Timedelta(days=1)
    out, rep = clean.clean_reviews(r)
    assert len(out) == 1 and rep["extra_reviews_per_order"] == 1
    assert out.iloc[0]["review_id"] == r.loc[1, "review_id"]


def test_zip_codes_padded_to_five_digits(raw):
    c = raw["customers"].copy()
    c.loc[0, "customer_zip_code_prefix"] = "1234"
    out, _ = clean.clean_customers(c)
    assert out.loc[0, "customer_zip_code_prefix"] == "01234"


def test_product_sizes_imputed_and_flagged(raw):
    out, rep = clean.clean_products(raw["products"], raw["category_translation"])
    assert out["product_weight_g"].isna().sum() == 0
    assert (out["product_weight_g"] > 0).all()
    assert rep["size_imputed"] == 2          # one missing, one zero
    assert rep["missing_category"] == 2
    assert "health_beauty" in set(out["product_category"])


def test_geolocation_outside_brazil_dropped_and_collapsed(raw):
    out, rep = clean.clean_geolocation(raw["geolocation"])
    assert rep["outside_brazil"] == 1
    assert out["zip"].is_unique
    assert out["lat"].between(-34, 5.5).all()


def test_not_defined_payments_removed(raw):
    p = raw["payments"].copy()
    p.loc[0, "payment_type"] = "not_defined"
    p.loc[1, "payment_installments"] = 0
    out, rep = clean.clean_payments(p)
    assert rep["not_defined_type"] == 1
    assert (out["payment_installments"] >= 1).all()
