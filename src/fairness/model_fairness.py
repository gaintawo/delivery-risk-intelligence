"""Fairness metrics and mitigation for the late-delivery model (Fairlearn).

Being flagged is *helpful* here: a flagged order gets proactive action
(seller nudged, faster courier, honest new date). The harm is a late order
that is NOT flagged. So the primary metric, set in Module 2, is equal
opportunity: recall on late orders should not differ by more than 10 points
between regions (goal G5, risk ER1). Demographic parity, equalized odds and
the disparate impact ratio are reported too, as the brief requires, but
demographic parity is not used as a gate: regions truly differ in lateness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import precision_score

GATE_RECALL_GAP = 0.10


def _base_rate(y_true, y_pred):
    return float(np.mean(y_true))


def _precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)


def by_group(y, pred, groups) -> pd.DataFrame:
    mf = MetricFrame(metrics={"orders": count, "late_rate": _base_rate, "selection_rate": selection_rate,
                              "recall_tpr": true_positive_rate, "fpr": false_positive_rate,
                              "precision": _precision},
                     y_true=y, y_pred=pred, sensitive_features=groups)
    out = mf.by_group.copy()
    out["late_orders"] = (out["orders"] * out["late_rate"]).round().astype(int)
    return out


def summary(y, pred, groups, min_late: int = 30) -> dict:
    """Aggregate fairness metrics. Groups with fewer than `min_late` late orders
    are reported but excluded from the recall-gap gate (too noisy to judge)."""
    g = by_group(y, pred, groups)
    judged = g[g["late_orders"] >= min_late]
    sr = g["selection_rate"]
    return {
        "demographic_parity_difference": round(float(demographic_parity_difference(y, pred, sensitive_features=groups)), 4),
        "demographic_parity_ratio": round(float(demographic_parity_ratio(y, pred, sensitive_features=groups)), 4),
        "disparate_impact_ratio": round(float(sr.min() / sr.max()), 4),
        "equalized_odds_difference": round(float(equalized_odds_difference(y, pred, sensitive_features=groups)), 4),
        "equalized_odds_ratio": round(float(equalized_odds_ratio(y, pred, sensitive_features=groups)), 4),
        "recall_gap_all_groups": round(float(g["recall_tpr"].max() - g["recall_tpr"].min()), 4),
        "recall_gap_judged_groups": round(float(judged["recall_tpr"].max() - judged["recall_tpr"].min()), 4),
        "groups_too_small_to_judge": g.index[g["late_orders"] < min_late].tolist(),
        "passes_g5_gate": bool((judged["recall_tpr"].max() - judged["recall_tpr"].min()) <= GATE_RECALL_GAP),
    }


def bootstrap_recall_ci(y, pred, groups, n=500, seed=0) -> pd.DataFrame:
    """95% bootstrap interval for recall in each group (shows how uncertain small groups are)."""
    rng = np.random.default_rng(seed)
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups)
    rows = []
    for g in pd.unique(groups):
        idx = np.where((groups == g) & (y == 1))[0]
        if len(idx) == 0:
            continue
        draws = [pred[rng.choice(idx, len(idx))].mean() for _ in range(n)]
        rows.append({"group": g, "recall": pred[idx].mean(), "ci_low": np.quantile(draws, 0.025),
                     "ci_high": np.quantile(draws, 0.975), "late_orders": len(idx)})
    return pd.DataFrame(rows).set_index("group")


def fit_threshold_optimizer(model, X_val, y_val, groups_val, seed=0) -> ThresholdOptimizer:
    """Post-processing mitigation: region-specific thresholds chosen on the
    validation set so that recall on late orders is equal across regions."""
    to = ThresholdOptimizer(estimator=model, constraints="true_positive_rate_parity",
                            objective="balanced_accuracy_score", predict_method="predict_proba",
                            prefit=True)
    to.fit(X_val, y_val, sensitive_features=groups_val)
    return to
