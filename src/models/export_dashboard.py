"""Export the files the Streamlit dashboard reads (Module 5).

    python -m src.models.export_dashboard

Writes dashboard/data/. The dashboard is deployed on Streamlit Community Cloud,
which has no access to the warehouse, so it reads a small, de-identified copy
of the held-out test period (June-August 2018) scored by the Module 4 champion:

* no order, customer or seller keys (privacy plan, ER4);
* location only as state and region;
* Olist data is CC BY-NC-SA 4.0, so the copy keeps that licence.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models import evaluate as ev
from src.models.data import FEATURES, TARGET, TIME_COL, load_modelling_frame, split
from src.models.explain import Explainer
from src.models.train import FIG_DIR, MODEL_DIR

OUT = Path(__file__).resolve().parents[2] / "dashboard" / "data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((MODEL_DIR / "dri_late_classifier.json").read_text())
    model = joblib.load(MODEL_DIR / "dri_late_classifier.joblib")
    post = joblib.load(MODEL_DIR / "dri_fairness_postprocessor.joblib")
    te = split(load_modelling_frame())["test"].reset_index(drop=True)

    score = model.predict_proba(te[FEATURES])[:, 1]
    flag = ev.daily_capacity_flags(score, te[TIME_COL])
    fair = post.predict(te[FEATURES], sensitive_features=te["customer_region"], random_state=0)

    out = te[FEATURES].copy()
    out.insert(0, "purchase_date", pd.to_datetime(te[TIME_COL]).dt.date.astype(str))
    out["is_late"] = te[TARGET].astype(int)
    out["delivery_days"] = te["delivery_days"].round(1)
    out["review_score"] = te["review_score"]
    out["risk_score"] = score.round(5)
    out["flagged"] = flag
    out["flagged_fair"] = np.asarray(fair).astype(int)

    # Three example orders: a typical correct flag (Northeast), a typical low-risk order, a typical miss
    out["example"] = ""
    tp = out[(out.is_late == 1) & (out.flagged == 1) & (out.customer_region == "Northeast")]
    tn = out[(out.is_late == 0) & (out.flagged == 0)]
    fn = out[(out.is_late == 1) & (out.flagged == 0)]
    out.loc[int(tp.risk_score.sub(tp.risk_score.median()).abs().idxmin()), "example"] = "A: flagged, arrived late"
    out.loc[int(tn.risk_score.sub(tn.risk_score.median()).abs().idxmin()), "example"] = "B: low risk, arrived on time"
    # Module 5 uses a typical missed order (median score), not the near-miss Module 4 plotted
    out.loc[int(fn.risk_score.sub(fn.risk_score.median()).abs().idxmin()), "example"] = "C: missed, arrived late"

    # Local explanations for the dashboard order explorer (exact linear SHAP)
    expl = Explainer(model, None, meta["features"], mean_encoded=meta.get("encoded_mean"))
    g = expl.grouped(out[FEATURES])
    g.round(4).add_prefix("shap__").assign(base=round(expl.base_value(), 4)).to_csv(
        OUT / "test_shap.csv.gz", index=False, compression="gzip")
    out.to_csv(OUT / "test_scored.csv.gz", index=False, compression="gzip")

    for f in ("shap_global_importance.csv", "fairness_region_unaware.csv", "sensitivity_monthly.csv",
              "sensitivity_perturbation.csv", "backtest_summary.csv", "training_summary.json",
              "analysis_summary.json"):
        shutil.copy(FIG_DIR / f, OUT / f)
    print(f"wrote {len(out):,} scored orders to {OUT}")


if __name__ == "__main__":
    main()
