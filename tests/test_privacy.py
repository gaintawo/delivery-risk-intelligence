import json

import pandas as pd

from src.features.build import build_features
from src.pipeline import anonymize, clean, config, integrate, load
from src.pipeline.audit import audit

PII = {"order_id", "customer_id", "customer_unique_id", "seller_id", "customer_zip_code_prefix",
       "seller_zip_code_prefix", "customer_city", "seller_city", "customer_lat", "customer_lng",
       "review_comment_message", "review_comment_title"}


def _ready(raw):
    c, _ = clean.clean_all(raw)
    df, _ = integrate.build_order_table(c)
    return anonymize.anonymise(build_features(df))


def test_output_has_no_pii_columns(raw):
    out, rep = _ready(raw)
    assert not PII & set(out.columns)
    assert list(out.columns) == anonymize.OUTPUT_COLUMNS
    assert "customer_zip_code_prefix" in rep["columns_removed"]


def test_pseudonyms_are_stable_salted_and_one_way():
    s = pd.Series(["abc", "abc", "xyz"], dtype="string")
    a = anonymize.pseudonymise(s, salt="one")
    b = anonymize.pseudonymise(s, salt="two")
    assert a.iloc[0] == a.iloc[1] and a.iloc[0] != a.iloc[2]
    assert a.iloc[0] != b.iloc[0]
    assert a.str.fullmatch(r"[0-9a-f]{16}").all()
    assert "abc" not in a.iloc[0]


def test_k_anonymity_reports_smallest_group():
    df = pd.DataFrame({"state": ["SP", "SP", "SP", "AM"], "m": [1, 1, 1, 2]})
    k = anonymize.k_anonymity(df, ["state", "m"])
    assert k["k_min"] == 1 and k["groups_below_5"] == 2


def test_dashboard_feed_suppresses_small_cells(raw):
    out, _ = _ready(raw)
    feed, rep = load.dashboard_feed(out)
    small = feed[feed["suppressed"] == 1]
    assert small["orders"].isna().all() and small["late_rate"].isna().all()
    assert rep["cells_suppressed"] == len(small)


def test_audit_log_writes_one_json_line_per_event():
    audit("read", "orders", purpose="test", rows=10, cols=3)
    audit("pii_remove", "orders", purpose="test", removed=["customer_city"])
    lines = (config.LOG_DIR / "audit.log").read_text().strip().splitlines()
    events = [json.loads(line) for line in lines]
    assert [e["action"] for e in events] == ["read", "pii_remove"]
    for e in events:
        assert {"ts", "run_id", "actor", "action", "dataset", "purpose"} <= set(e)
