"""Rolling-origin backtest for champion selection (Module 4).

The first champion choice used one validation window (Mar-May 2018), the most
atypical period in the data (11.7% late vs 7.7% in training). Selection is
made more robust here by refitting each tuned model on all data before a
cutoff and scoring the next three months, for three cutoffs. Only training
and validation periods are used; the test period (Jun-Aug 2018) is untouched.

Disclosure: this step was added after the test results of MLflow model v3
had been seen (see the report, section 4).
"""
from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.models import evaluate as ev
from src.models.data import FEATURES, TARGET, TIME_COL, load_modelling_frame
from src.models.train import FIG_DIR, tracking_setup

WINDOWS = [("2017-09-01", "2017-12-01"), ("2017-12-01", "2018-03-01"), ("2018-03-01", "2018-06-01")]


def main():
    tracking_setup()
    df = load_modelling_frame()
    df = df[df[TIME_COL] >= "2017-01-01"]
    summary = json.loads((FIG_DIR / "training_summary.json").read_text())
    client = mlflow.MlflowClient()
    rows = []
    with mlflow.start_run(run_name="backtest-champion-selection"):
        mlflow.set_tags({"purpose": "rolling-origin selection; test period not used",
                         "windows": json.dumps(WINDOWS)})
        for name in ("Logistic Regression", "Random Forest", "XGBoost"):
            run_id = summary["results"][name]["run_id"]
            tuned = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            for lo, hi in WINDOWS:
                tr = df[df[TIME_COL] < lo]
                ho = df[(df[TIME_COL] >= lo) & (df[TIME_COL] < hi)]
                m = clone(tuned).fit(tr[FEATURES], tr[TARGET])
                s = m.predict_proba(ho[FEATURES])[:, 1]
                met = ev.classification_metrics(ho[TARGET].to_numpy(), s,
                                                pred=ev.daily_capacity_flags(s, ho[TIME_COL]))
                rows.append({"model": name, "window": f"{lo} to {hi}", "train_rows": len(tr),
                             "holdout_rows": len(ho), "late_rate": round(float(ho[TARGET].mean()), 4), **met})
                print(name, lo, met["recall"], met["roc_auc"])
        res = pd.DataFrame(rows)
        agg = res.groupby("model")[["recall", "precision", "roc_auc", "pr_auc"]].mean().round(4)
        agg["recall_min"] = res.groupby("model")["recall"].min().round(4)
        agg = agg.sort_values(["recall", "pr_auc"], ascending=False)
        champ = agg.index[0]
        res.to_csv(FIG_DIR / "backtest_windows.csv", index=False)
        agg.to_csv(FIG_DIR / "backtest_summary.csv")
        mlflow.log_artifact(str(FIG_DIR / "backtest_windows.csv"), "backtest")
        mlflow.log_artifact(str(FIG_DIR / "backtest_summary.csv"), "backtest")
        for n, r in agg.iterrows():
            key = n.lower().replace(" ", "_")
            mlflow.log_metrics({f"{key}_mean_recall": r["recall"], f"{key}_mean_roc_auc": r["roc_auc"]})
        mlflow.set_tag("selected", champ)
    print(agg.to_string())
    print("SELECTED:", champ)
    (FIG_DIR / "backtest_selected.json").write_text(json.dumps(
        {"selected": champ, "run_id": summary["results"][champ]["run_id"], "summary": agg.to_dict()}, indent=2))
    _ = client
    return champ


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
