import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from src.monitoring.monitor import TH, alerts, psi


def test_psi_is_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.02
    assert psi(a, rng.normal(loc=1.5, size=5000)) > TH["psi"]


def test_alerts_fire_on_each_threshold():
    row = {"week": "2018-08-06", "recall_4w": 0.30, "auc_4w": 0.65, "score_psi": 0.4, "max_feature_psi": 0.5,
           "max_psi_feature": "promised_days", "flag_rate_threshold": 0.05, "recall_gap_4w": 0.3,
           "flag_rate_gap_regions": 0.9, "flag_rate_gap_reference": 0.6}
    a = alerts(pd.DataFrame([row]))
    assert set(a["type"]) == {"performance", "drift", "fairness"}
    assert len(a) == 7


def test_no_alerts_when_healthy():
    row = {"week": "2018-08-06", "recall_4w": 0.6, "auc_4w": 0.8, "score_psi": 0.05, "max_feature_psi": 0.05,
           "max_psi_feature": "promised_days", "flag_rate_threshold": 0.25, "recall_gap_4w": 0.05,
           "flag_rate_gap_regions": 0.2, "flag_rate_gap_reference": 0.2}
    assert alerts(pd.DataFrame([row])).empty
