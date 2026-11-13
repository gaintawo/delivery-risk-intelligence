"""Explainability, fairness, counterfactual and sensitivity analysis (Module 4).

    python -m src.models.analyze

Runs on the registered champion model and the frozen time-aware test set.
Writes figures and tables to docs/evidence/model/ and logs them to MLflow as
the run "analysis-xai-fairness".
"""
from __future__ import annotations

import json
import os
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.fairness import model_fairness as mf
from src.models import evaluate as ev
from src.models.data import (
    CATEGORICAL,
    FEATURES,
    GEO_IDENTITY,
    NUMERIC,
    TARGET,
    TIME_COL,
    load_modelling_frame,
    split,
)
from src.models.explain import PLAIN, Explainer
from src.models.train import FIG_DIR, MODEL_DIR, preprocessor, tracking_setup

warnings.filterwarnings("ignore")
REGION_ORDER = ["North", "Northeast", "Central-West", "South", "Southeast"]


def _save(fig, name):
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    return FIG_DIR / name


# ------------------------------------------------------------------ SHAP
def shap_global(expl: Explainer, X: pd.DataFrame) -> pd.DataFrame:
    G = expl.grouped(X)
    imp = G.abs().mean().sort_values(ascending=False)
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    ax.barh([PLAIN.get(f, f) for f in top.index], top.values, color=ev.TEAL, height=0.62)
    for i, v in enumerate(top.values):
        ax.text(v + top.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=ev.INK)
    ax.set_xlabel("Mean |SHAP value| (log-odds of late delivery)", fontsize=9, color=ev.MUTED)
    ev._style(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=ev.INK)
    _save(fig, "shap_global_importance.png")

    # beeswarm for the numeric drivers (colour = feature value)
    num_top = [f for f in imp.index if f in NUMERIC][:10]
    e = shap.Explanation(values=G[num_top].to_numpy(), data=X[num_top].to_numpy(dtype=float),
                         feature_names=[PLAIN.get(f, f) for f in num_top])
    plt.figure(figsize=(7.5, 4.8), dpi=200)
    shap.plots.beeswarm(e, max_display=10, show=False, color_bar=True)
    plt.xlabel("SHAP value (impact on log-odds of late delivery)", fontsize=9)
    plt.gcf().savefig(FIG_DIR / "shap_beeswarm.png", bbox_inches="tight")
    plt.close("all")
    return imp.round(4).to_frame("mean_abs_shap")


def shap_local(expl: Explainer, X: pd.DataFrame, idx: int, name: str, title: str) -> dict:
    row = X.iloc[[idx]]
    g = expl.grouped(row).iloc[0]
    e = shap.Explanation(values=g.to_numpy(), base_values=expl.base_value(),
                         data=np.array([row.iloc[0][f] for f in FEATURES], dtype=object),
                         feature_names=[PLAIN.get(f, f) for f in FEATURES])
    plt.figure(figsize=(7.5, 4.6), dpi=200)
    shap.plots.waterfall(e, max_display=9, show=False)
    plt.title(title, fontsize=10)
    plt.gcf().savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close("all")
    return {"reasons": expl.reasons(row), "base_log_odds": round(expl.base_value(), 3),
            "score": float(expl.pipe.predict_proba(row[FEATURES])[0, 1])}


# ------------------------------------------------------------------ fairness
def fairness(model, thr, s) -> dict:
    va, te = s["validation"], s["test"]
    yte = te[TARGET].to_numpy()
    pred = ev.daily_capacity_flags(model.predict_proba(te[FEATURES])[:, 1], te[TIME_COL])
    out = {}
    for attr in ("customer_region", "seller_size"):
        g = mf.by_group(yte, pred, te[attr])
        out[attr] = {"by_group": g.round(4), "summary": mf.summary(yte, pred, te[attr])}
    ci = mf.bootstrap_recall_ci(yte, pred, te["customer_region"]).reindex(REGION_ORDER)

    # Mitigation 1: post-processing, recall parity across regions (thresholds fitted on validation)
    to = mf.fit_threshold_optimizer(model, va[FEATURES], va[TARGET], va["customer_region"])
    joblib.dump(to, MODEL_DIR / "dri_fairness_postprocessor.joblib", compress=3)
    pred_to = to.predict(te[FEATURES], sensitive_features=te["customer_region"], random_state=0)
    g_to = mf.by_group(yte, pred_to, te["customer_region"])
    ci_to = mf.bootstrap_recall_ci(yte, pred_to, te["customer_region"]).reindex(REGION_ORDER)
    out["mitigated_threshold_optimizer"] = {"by_group": g_to.round(4),
                                            "summary": mf.summary(yte, pred_to, te["customer_region"]),
                                            "overall_recall": round(float(pred_to[yte == 1].mean()), 4),
                                            "overall_precision": round(float(yte[pred_to == 1].mean()), 4),
                                            "flag_rate": round(float(pred_to.mean()), 4)}

    # Mitigation 2 / proxy test (ER2): same model without geography identity columns
    feats = [f for f in FEATURES if f not in GEO_IDENTITY]
    cats = [c for c in CATEGORICAL if c not in GEO_IDENTITY]
    nums = [c for c in NUMERIC]
    unaware = Pipeline([("prep", preprocessor(nums, cats)), ("model", clone(model.named_steps["model"]))])
    unaware.fit(s["train"][feats], s["train"][TARGET])
    ste = unaware.predict_proba(te[feats])[:, 1]
    pred_u = ev.daily_capacity_flags(ste, te[TIME_COL])
    out["unaware_no_geography"] = {"metrics": ev.classification_metrics(yte, ste, pred=pred_u),
                                   "by_group": mf.by_group(yte, pred_u, te["customer_region"]).round(4),
                                   "summary": mf.summary(yte, pred_u, te["customer_region"])}

    # figure: recall by region, before vs after mitigation, with 95% CIs
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    x = np.arange(len(REGION_ORDER))
    w = 0.36
    for k, (frame, label, color) in enumerate([(ci, "Champion, top 25% of each day", ev.TEAL),
                                                (ci_to, "After ThresholdOptimizer (recall parity)", ev.ORANGE)]):
        xs = x + (k - 0.5) * w
        ax.bar(xs, frame["recall"], width=w * 0.92, color=color, label=label)
        ax.errorbar(xs, frame["recall"], yerr=[frame["recall"] - frame["ci_low"], frame["ci_high"] - frame["recall"]],
                    fmt="none", ecolor=ev.INK, elinewidth=0.9, capsize=3)
        for xi, v in zip(xs, frame["recall"]):
            ax.text(xi, 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=8,
                    color="white", fontweight="bold")
    ax.set_xticks(x, [f"{r}\n({int(ci.loc[r, 'late_orders'])} late)" for r in REGION_ORDER])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Recall on late orders (test)", fontsize=9, color=ev.MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    ev._style(ax)
    _save(fig, "fairness_recall_by_region.png")
    out["recall_ci"] = ci.round(4)
    out["recall_ci_mitigated"] = ci_to.round(4)
    return out


# ------------------------------------------------------------------ counterfactuals
def _extend_promise(X: pd.DataFrame, days: float) -> pd.DataFrame:
    """A longer promise changes the promised days AND the slack against recent delivery times."""
    X = X.copy()
    X["promised_days"] = X["promised_days"] + days
    X["promise_slack_days"] = X["promise_slack_days"] + days
    return X


def promise_extension_needed(model, thr, X: pd.DataFrame, max_days: int = 30) -> pd.Series:
    """Smallest whole number of extra promise days that brings each order below the threshold."""
    need = pd.Series(np.nan, index=X.index)
    for d in range(max_days + 1):
        p = model.predict_proba(_extend_promise(X, d)[FEATURES])[:, 1]
        hit = need.isna() & (p < thr)
        need[hit] = d
    return need


def counterfactuals(model, thr, s, n_queries=3) -> dict:
    """DiCE counterfactuals for flagged late orders, restricted to actions Olist controls:
    a longer promise (slack), or routing to a seller with a better track record."""
    import dice_ml
    from raiutils.exceptions import UserConfigValidationException
    tr, te = s["train"], s["test"]
    vary = ["promise_slack_days", "seller_hist_late_rate", "seller_hist_handling_days"]
    dtypes = tr[FEATURES].dtypes.to_dict()
    data = dice_ml.Data(dataframe=tr[FEATURES + [TARGET]].dropna(subset=vary), continuous_features=NUMERIC,
                        outcome_name=TARGET)
    dice = dice_ml.Dice(data, dice_ml.Model(model=model, backend="sklearn", model_type="classifier"),
                        method="random")
    sc = model.predict_proba(te[FEATURES])[:, 1]
    flagged = te.assign(score=sc, flag=ev.daily_capacity_flags(sc, te[TIME_COL])).dropna(subset=vary)
    flagged = flagged[(flagged["flag"] == 1) & (flagged["score"] >= thr) & (flagged[TARGET] == 1)].sort_values("score")
    picks = flagged.iloc[np.linspace(0, len(flagged) - 1, n_queries).astype(int)]
    examples = []
    for _, q in picks.iterrows():
        qdf = q[FEATURES].to_frame().T.astype(dtypes)
        try:
            cf = dice.generate_counterfactuals(
                qdf, total_CFs=3, desired_class=0, features_to_vary=vary, random_seed=1,
                stopping_threshold=1 - thr,
                permitted_range={"promise_slack_days": [float(q["promise_slack_days"]), float(q["promise_slack_days"]) + 30],
                                 "seller_hist_late_rate": [0.0, float(q["seller_hist_late_rate"])],
                                 "seller_hist_handling_days": [0.5, float(q["seller_hist_handling_days"])]})
            cfs = cf.cf_examples_list[0].final_cfs_df
        except UserConfigValidationException as e:  # DiCE raises when nothing is found in range
            cfs = None
            print("DiCE:", str(e)[:120])
        if cfs is None or cfs.empty:
            examples.append({"order_key": q["order_key"], "region": q["customer_region"],
                             "score": round(float(q["score"]), 3), "counterfactuals": [], "note": "none found"})
            continue
        cfs = cfs[FEATURES].astype(dtypes).copy()
        # keep the promise coherent: promised days move with the slack
        cfs["promised_days"] = float(q["promised_days"]) + (cfs["promise_slack_days"] - float(q["promise_slack_days"]))
        new_scores = model.predict_proba(cfs[FEATURES])[:, 1]
        changes = []
        for (_, c), ns in zip(cfs.iterrows(), new_scores):
            ch = {f: (round(float(q[f]), 2), round(float(c[f]), 2))
                  for f in vary + ["promised_days"] if abs(float(c[f]) - float(q[f])) > 1e-6}
            changes.append({"changes": ch, "new_score": round(float(ns), 3), "below_threshold": bool(ns < thr)})
        examples.append({"order_key": q["order_key"], "region": q["customer_region"],
                         "score": round(float(q["score"]), 3), "counterfactuals": changes})

    # population view: how much longer would the promise need to be?
    flagged_all = te.assign(score=sc, flag=ev.daily_capacity_flags(sc, te[TIME_COL]))
    flagged_all = flagged_all[(flagged_all["flag"] == 1) & (flagged_all["score"] >= thr)].dropna(subset=["promise_slack_days"])
    need = promise_extension_needed(model, thr, flagged_all)
    by_region = need.groupby(flagged_all["customer_region"]).median()
    return {"examples": examples,
            "promise_extension": {"orders": len(need), "median_days": float(need.median()),
                                  "share_resolved_within_7_days": round(float((need <= 7).mean()), 4),
                                  "share_not_resolved_by_30_days": round(float(need.isna().mean()), 4),
                                  "median_days_by_region": by_region.round(1).to_dict()}}


# ------------------------------------------------------------------ sensitivity
def sensitivity(model, thr, s) -> dict:
    te = s["test"]
    X, y = te[FEATURES].copy(), te[TARGET].to_numpy()
    base = model.predict_proba(X)[:, 1]
    base_flag = base >= thr
    rows = []
    tests = {"promised_days": [("-2 days", lambda v: v - 2), ("+2 days", lambda v: v + 2)],
             "distance_km": [("-10%", lambda v: v * 0.9), ("+10%", lambda v: v * 1.1)],
             "freight_total": [("-10%", lambda v: v * 0.9), ("+10%", lambda v: v * 1.1)],
             "price_total": [("-10%", lambda v: v * 0.9), ("+10%", lambda v: v * 1.1)],
             "weight_total_g": [("-10%", lambda v: v * 0.9), ("+10%", lambda v: v * 1.1)],
             "seller_hist_late_rate": [("-0.05", lambda v: (v - 0.05).clip(lower=0)),
                                       ("+0.05", lambda v: (v + 0.05).clip(upper=1))]}
    for f, variants in tests.items():
        for label, fn in variants:
            Xp = X.copy()
            Xp[f] = fn(Xp[f])
            p = model.predict_proba(Xp)[:, 1]
            rows.append({"feature": PLAIN[f], "change": label,
                         "mean_abs_score_change": round(float(np.abs(p - base).mean()), 4),
                         "mean_score_change": round(float((p - base).mean()), 4),
                         "decisions_flipped": round(float(((p >= thr) != base_flag).mean()), 4),
                         "roc_auc": round(float(roc_auc_score(y, p)), 4)})
    rng = np.random.default_rng(0)
    Xn = X.copy()
    for f in NUMERIC:
        sd = X[f].std()
        if sd and not np.isnan(sd):
            Xn[f] = X[f] + rng.normal(0, 0.05 * sd, len(X))
            if X[f].min() >= 0:  # counts, sizes, rates and distances cannot go below zero
                Xn[f] = Xn[f].clip(lower=0)
    pn = model.predict_proba(Xn)[:, 1]
    rows.append({"feature": "All numeric features", "change": "Gaussian noise, 5% of SD",
                 "mean_abs_score_change": round(float(np.abs(pn - base).mean()), 4),
                 "mean_score_change": round(float((pn - base).mean()), 4),
                 "decisions_flipped": round(float(((pn >= thr) != base_flag).mean()), 4),
                 "roc_auc": round(float(roc_auc_score(y, pn)), 4)})
    Xc = X.copy()
    Xc[["seller_hist_orders", "seller_hist_late_rate", "seller_hist_handling_days"]] = [0, np.nan, np.nan]
    Xc["seller_size"] = "new"
    pc = model.predict_proba(Xc)[:, 1]
    rows.append({"feature": "Seller history removed", "change": "Every seller treated as new",
                 "mean_abs_score_change": round(float(np.abs(pc - base).mean()), 4),
                 "mean_score_change": round(float((pc - base).mean()), 4),
                 "decisions_flipped": round(float(((pc >= thr) != base_flag).mean()), 4),
                 "roc_auc": round(float(roc_auc_score(y, pc)), 4)})
    monthly = (te.assign(score=base, flag=ev.daily_capacity_flags(base, te[TIME_COL])).groupby(te["order_purchase_timestamp"].dt.to_period("M"))
                 .apply(lambda d: pd.Series({"orders": len(d), "late_rate": d[TARGET].mean(),
                                             "roc_auc": roc_auc_score(d[TARGET], d["score"]),
                                             "recall": d.loc[d[TARGET] == 1, "flag"].mean(),
                                             "flag_rate": d["flag"].mean()})))
    monthly.index = monthly.index.astype(str)
    return {"perturbation": pd.DataFrame(rows), "monthly": monthly.round(4),
            "baseline_roc_auc": round(float(roc_auc_score(y, base)), 4)}


def main():
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    tracking_setup()
    meta = json.loads((MODEL_DIR / "dri_late_classifier.json").read_text())
    model = joblib.load(MODEL_DIR / "dri_late_classifier.joblib")
    thr = meta["threshold"]
    s = split(load_modelling_frame())
    te = s["test"]
    expl = Explainer(model, None, FEATURES, mean_encoded=meta.get("encoded_mean"))

    sample = te.sample(4000, random_state=0).reset_index(drop=True)
    imp = shap_global(expl, sample)
    sc = model.predict_proba(te[FEATURES])[:, 1]
    te_s = te.assign(score=sc, flag=ev.daily_capacity_flags(sc, te[TIME_COL]))
    tp = te_s[(te_s[TARGET] == 1) & (te_s["flag"] == 1) & (te_s["customer_region"] == "Northeast")]
    tn = te_s[(te_s[TARGET] == 0) & (te_s["flag"] == 0)]
    fn = te_s[(te_s[TARGET] == 1) & (te_s["flag"] == 0)]
    local = {
        "true_positive_northeast": shap_local(expl, te, int(tp["score"].sub(tp["score"].median()).abs().idxmin()),
                                              "shap_local_flagged.png", "Order correctly flagged as likely late"),
        "true_negative": shap_local(expl, te, int(tn["score"].sub(tn["score"].median()).abs().idxmin()),
                                    "shap_local_low_risk.png", "Low-risk order, delivered on time"),
        "false_negative": shap_local(expl, te, int(fn["score"].idxmax()),
                                     "shap_local_missed.png", "Late order the model missed"),
    }
    fair = fairness(model, thr, s)
    cfs = counterfactuals(model, thr, s)
    sens = sensitivity(model, thr, s)

    # tables to disk
    imp.to_csv(FIG_DIR / "shap_global_importance.csv")
    for k in ("customer_region", "seller_size"):
        fair[k]["by_group"].to_csv(FIG_DIR / f"fairness_{k}.csv")
    fair["mitigated_threshold_optimizer"]["by_group"].to_csv(FIG_DIR / "fairness_region_mitigated.csv")
    fair["unaware_no_geography"]["by_group"].to_csv(FIG_DIR / "fairness_region_unaware.csv")
    fair["recall_ci"].to_csv(FIG_DIR / "fairness_recall_ci.csv")
    sens["perturbation"].to_csv(FIG_DIR / "sensitivity_perturbation.csv", index=False)
    sens["monthly"].to_csv(FIG_DIR / "sensitivity_monthly.csv")
    summary = {
        "local_explanations": local, "counterfactuals": cfs,
        "fairness": {k: (v["summary"] if isinstance(v, dict) and "summary" in v else None)
                     for k, v in fair.items() if isinstance(v, dict)},
        "mitigated": {k: v for k, v in fair["mitigated_threshold_optimizer"].items() if k != "by_group"},
        "unaware_metrics": fair["unaware_no_geography"]["metrics"],
        "sensitivity_baseline_auc": sens["baseline_roc_auc"],
    }
    (FIG_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    with mlflow.start_run(run_name="analysis-xai-fairness"):
        mlflow.set_tags({"champion_run_id": meta["mlflow_run_id"], "model_version": meta["version"]})
        for k, v in fair["customer_region"]["summary"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                mlflow.log_metric(f"region_{k}", v)
        for k, v in fair["mitigated_threshold_optimizer"]["summary"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                mlflow.log_metric(f"mitigated_region_{k}", v)
        mlflow.log_artifacts(str(FIG_DIR), artifact_path="evidence")
    print(json.dumps({"fairness_region": fair["customer_region"]["summary"],
                      "mitigated": summary["mitigated"], "unaware": summary["unaware_metrics"],
                      "n_counterfactual_orders": len(cfs)}, indent=1, default=str))


if __name__ == "__main__":
    main()
