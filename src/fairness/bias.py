"""Stage 8: representation-bias detection (Fairlearn).

Checks the *data* before any model exists, for the groups named in the
Module 1 Ethical AI Charter: customer region (ER1) and seller size (ER3).

For each group it reports:
  * representation ratio - share of orders / share of Brazil's population (region only);
    below 0.8 means under-represented (four-fifths rule);
  * label coverage      - share of orders that end up with a delivered/late label;
  * base rate           - share of delivered orders that were late;
  * sample size         - delivered orders available for training.
Group statistics are computed with fairlearn.metrics.MetricFrame.
"""
from __future__ import annotations

import json

import pandas as pd
from fairlearn.metrics import MetricFrame, count

from src.pipeline import config
from src.pipeline.audit import audit

MIN_GROUP_SIZE = 1000  # delivered orders; below this, model metrics per group are unreliable


def _base_rate(y_true, y_pred):
    return float(pd.Series(y_true).mean())


def group_frame(df: pd.DataFrame, by: str) -> pd.DataFrame:
    labelled = df[df["is_delivered"] == 1]
    y = labelled["is_late"].astype(int)
    mf = MetricFrame(metrics={"delivered_orders": count, "late_rate": _base_rate},
                     y_true=y, y_pred=y, sensitive_features=labelled[by])
    out = mf.by_group.copy()
    out["orders"] = df.groupby(by).size()
    out["order_share"] = out["orders"] / out["orders"].sum()
    out["label_coverage"] = out["delivered_orders"] / out["orders"]
    out["late_rate_diff_vs_overall"] = out["late_rate"] - float(y.mean())
    out["small_sample"] = out["delivered_orders"] < MIN_GROUP_SIZE
    return out


def region_report(df: pd.DataFrame) -> pd.DataFrame:
    out = group_frame(df, "customer_region")
    out["population_share"] = pd.Series(config.POPULATION_SHARE_2022)
    out["representation_ratio"] = out["order_share"] / out["population_share"]
    out["under_represented"] = out["representation_ratio"] < config.REPRESENTATION_THRESHOLD
    return out.sort_values("representation_ratio")


def run_bias_checks(df: pd.DataFrame) -> dict:
    region = region_report(df)
    seller = group_frame(df.dropna(subset=["seller_size"]), "seller_size")
    config.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    region.round(4).to_csv(config.EVIDENCE_DIR / "bias_region.csv")
    seller.round(4).to_csv(config.EVIDENCE_DIR / "bias_seller_size.csv")

    flags = {
        "under_represented_regions": region.index[region["under_represented"]].tolist(),
        "small_sample_regions": region.index[region["small_sample"]].tolist(),
        "late_rate_gap_regions_pts": round(100 * float(region["late_rate"].max() - region["late_rate"].min()), 1),
        "late_rate_gap_seller_size_pts": round(100 * float(seller["late_rate"].max() - seller["late_rate"].min()), 1),
    }
    (config.EVIDENCE_DIR / "bias_summary.json").write_text(json.dumps(flags, indent=2))
    audit("bias_check", "orders_model_ready", purpose="Representation bias check (ER1, ER3)",
          rows=len(df), **flags)
    return {"region": region, "seller_size": seller, "flags": flags}
