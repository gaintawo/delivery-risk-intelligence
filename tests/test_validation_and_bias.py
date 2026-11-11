import pandas as pd

from src.fairness.bias import region_report
from src.features.build import build_features
from src.pipeline import anonymize, clean, integrate, validate


def _ready(raw):
    c, _ = clean.clean_all(raw)
    df, _ = integrate.build_order_table(c)
    out, _ = anonymize.anonymise(build_features(df))
    return out


def test_validation_gate_rejects_a_leaked_pii_column(raw):
    out = _ready(raw)
    leaked = out.assign(customer_zip_code_prefix="01234")
    summary = validate.run_validation(raw["orders"], raw["items"], leaked)
    assert summary["orders_model_ready"]["success"] is False
    assert "expect_table_columns_to_match_set" in summary["orders_model_ready"]["failed"]


def test_representation_ratio_uses_population_share():
    rows = (["Southeast"] * 70 + ["South"] * 15 + ["Northeast"] * 9 + ["Central-West"] * 4
            + ["North"] * 2)
    df = pd.DataFrame({"customer_region": rows, "is_delivered": 1,
                       "is_late": pd.array([0] * 100, dtype="Int8")})
    rep = region_report(df)
    assert round(rep.loc["North", "representation_ratio"], 3) == round(0.02 / 0.085, 3)
    assert bool(rep.loc["North", "under_represented"]) is True
    assert bool(rep.loc["Southeast", "under_represented"]) is False
