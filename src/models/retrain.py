"""Automated monthly retraining with promotion gates (Final Project).

    python -m src.models.retrain                     # replay Jun, Jul, Aug 2018 month-ends
    python -m src.models.retrain --cutoff 2018-08-01 # one run, as the scheduled job does

At each month-end cutoff the job:

1. uses only orders whose outcome is already known: purchases at least 30 days
   before the cutoff (most promise windows have closed);
2. trains a candidate with the champion's recipe (logistic regression, C =
   0.0042, balanced class weights, same leakage-safe pipeline) on the last 13
   labelled months;
3. scores candidate and current model on the most recent labelled month
   (the "gate month", which neither model was trained on for the candidate);
4. applies the promotion gates below and logs everything to MLflow
   (experiment dri-retraining), registering the candidate with alias
   "candidate" and tag gates_passed;
5. never promotes itself: the Review Board moves the "champion" alias
   (human-in-the-loop, Ethical AI Charter: Accountability).

For the replay it also scores both models on the month after the cutoff, to
show what retraining would actually have changed. That month is never used
for the decision.

Promotion gates (all must pass on the gate month)
* recall within capacity >= current model's recall - 0.02
* ROC-AUC >= current model's ROC-AUC - 0.01
* regional recall gap (regions with >= 15 late orders) <= current gap + 0.05
* leakage check passes (every input known at purchase)
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.models import evaluate as ev
from src.models.data import (
    FEATURES,
    TARGET,
    TIME_COL,
    check_no_leakage,
    load_modelling_frame,
)
from src.models.train import MODEL_DIR, MODEL_NAME, preprocessor
from src.pipeline import config

EXPERIMENT = "dri-retraining"
OUT = config.ROOT / "docs" / "evidence" / "retraining"
CANDIDATES = config.ROOT / "build" / "candidates"
LABEL_LAG_DAYS = 30
TRAIN_MONTHS = 13
C_CHAMPION = 0.0042
MIN_LATE_REGION = 15
GATES = {"recall_drop": 0.02, "auc_drop": 0.01, "gap_increase": 0.05}
CUTOFFS = ["2018-06-01", "2018-07-01", "2018-08-01"]


def metrics(y: pd.Series, score: np.ndarray, dates: pd.Series, region: pd.Series) -> dict:
    flag = ev.daily_capacity_flags(score, dates)
    late = y.to_numpy() == 1
    rec = pd.Series(flag[late]).groupby(region.to_numpy()[late]).agg(["mean", "size"])
    judged = rec[rec["size"] >= MIN_LATE_REGION]["mean"]
    return {"recall": float(flag[late].mean()) if late.any() else float("nan"),
            "precision": float(y.to_numpy()[flag == 1].mean()),
            "roc_auc": float(roc_auc_score(y, score)) if y.nunique() == 2 else float("nan"),
            "pr_auc": float(average_precision_score(y, score)) if y.nunique() == 2 else float("nan"),
            "regional_gap": float(judged.max() - judged.min()) if len(judged) >= 2 else float("nan"),
            "orders": len(y), "late_orders": int(late.sum())}


def gates(cand: dict, cur: dict) -> dict:
    g = {
        "recall": cand["recall"] >= cur["recall"] - GATES["recall_drop"],
        "roc_auc": cand["roc_auc"] >= cur["roc_auc"] - GATES["auc_drop"],
        "fairness": (np.isnan(cand["regional_gap"]) or np.isnan(cur["regional_gap"])
                     or cand["regional_gap"] <= cur["regional_gap"] + GATES["gap_increase"]),
    }
    g["all"] = all(g.values())
    return g


def setup_mlflow() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{config.ROOT / 'mlflow.db'}"))
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(config.ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def run(df: pd.DataFrame, cutoff: str, current, current_name: str, register: bool = True) -> dict:
    c = pd.Timestamp(cutoff)
    t = pd.to_datetime(df[TIME_COL])
    labelled_until = c - pd.Timedelta(days=LABEL_LAG_DAYS)
    gate_from = labelled_until - pd.DateOffset(months=1)
    train_from = gate_from - pd.DateOffset(months=TRAIN_MONTHS)
    tr = df[(t >= train_from) & (t < gate_from)]
    gm = df[(t >= gate_from) & (t < labelled_until)]
    nxt = df[(t >= c) & (t < c + pd.DateOffset(months=1))]
    check_no_leakage(FEATURES)

    cand = Pipeline([("prep", preprocessor()),
                     ("model", LogisticRegression(C=C_CHAMPION, class_weight="balanced", max_iter=2000))])
    cand.fit(tr[FEATURES], tr[TARGET])

    def m(model, d):
        return metrics(d[TARGET], model.predict_proba(d[FEATURES])[:, 1], d[TIME_COL], d["customer_region"])

    res = {"cutoff": cutoff, "train_from": train_from.date().isoformat(), "gate_from": gate_from.date().isoformat(),
           "labelled_until": labelled_until.date().isoformat(), "train_orders": len(tr),
           "current_model": current_name,
           "gate_candidate": m(cand, gm), "gate_current": m(current, gm),
           "next_candidate": m(cand, nxt) if len(nxt) else None, "next_current": m(current, nxt) if len(nxt) else None}
    res["gates"] = gates(res["gate_candidate"], res["gate_current"])

    with mlflow.start_run(run_name=f"retrain-{cutoff}"):
        mlflow.set_tags({"cutoff": cutoff, "trigger": os.getenv("DRI_RETRAIN_TRIGGER", "manual replay"),
                         "gates_passed": str(res["gates"]["all"]), "current_model": current_name,
                         "promotion": "requires Review Board approval (alias champion is never set automatically)"})
        mlflow.log_params({"C": C_CHAMPION, "train_from": res["train_from"], "gate_from": res["gate_from"],
                           "labelled_until": res["labelled_until"], "train_orders": len(tr)})
        for k in ("gate_candidate", "gate_current", "next_candidate", "next_current"):
            if res[k]:
                mlflow.log_metrics({f"{k}_{mk}": v for mk, v in res[k].items() if not pd.isna(v)})
        mlflow.log_dict(res, "retrain_result.json")
        if register:
            info = mlflow.sklearn.log_model(cand, name="model", serialization_format="cloudpickle")
            mv = mlflow.register_model(info.model_uri, MODEL_NAME)
            client = mlflow.MlflowClient()
            client.set_model_version_tag(MODEL_NAME, mv.version, "gates_passed", str(res["gates"]["all"]))
            client.set_model_version_tag(MODEL_NAME, mv.version, "cutoff", cutoff)
            if res["gates"]["all"]:
                client.set_registered_model_alias(MODEL_NAME, "candidate", mv.version)
            res["registered_version"] = int(mv.version)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    joblib.dump(cand, CANDIDATES / f"candidate_{cutoff}.joblib", compress=3)
    return res, cand


def main(cutoffs: list[str], register: bool = True) -> list[dict]:
    setup_mlflow()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_modelling_frame()
    df = df[pd.to_datetime(df[TIME_COL]) >= "2017-01-01"]
    current = joblib.load(MODEL_DIR / "dri_late_classifier.joblib")
    static = current
    current_name = "v3 champion (trained to Feb 2018)"
    results = []
    for c in cutoffs:
        res, cand = run(df, c, current, current_name, register)
        # also record what the never-retrained champion would have done next month
        nxt = df[(pd.to_datetime(df[TIME_COL]) >= c) & (pd.to_datetime(df[TIME_COL]) < pd.Timestamp(c) + pd.DateOffset(months=1))]
        res["next_static_v3"] = metrics(nxt[TARGET], static.predict_proba(nxt[FEATURES])[:, 1], nxt[TIME_COL],
                                        nxt["customer_region"]) if len(nxt) else None
        results.append(res)
        # Simulation of the approval step: a gate-passing candidate is what the Review Board would review;
        # the replay assumes approval so the next month compares against it.
        if res["gates"]["all"]:
            current, current_name = cand, f"retrained at {c}"
        print(json.dumps({k: res[k] for k in ("cutoff", "gates", "train_orders")}, default=str))
    (OUT / "retraining_results.json").write_text(json.dumps(results, indent=2, default=str))
    rows = []
    for r in results:
        for k, label in (("gate_candidate", "gate: candidate"), ("gate_current", "gate: current"),
                         ("next_candidate", "next month: candidate"), ("next_current", "next month: current"),
                         ("next_static_v3", "next month: never retrained (v3)")):
            if r.get(k):
                rows.append({"cutoff": r["cutoff"], "model": label, **r[k], "gates_passed": r["gates"]["all"]})
    pd.DataFrame(rows).round(4).to_csv(OUT / "retraining_results.csv", index=False)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", action="append", help="month-end cutoff date (YYYY-MM-DD); repeatable")
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    main(a.cutoff or CUTOFFS, register=not a.no_register)
