"""Model monitoring: performance, data drift, prediction drift and fairness drift.

    python -m src.monitoring.monitor          # needs the pipeline output and models/

Replays the held-out test period (June-August 2018) week by week as if the
champion model were live, and checks every signal in the Model Card's
monitoring plan against its action threshold:

| Signal                                  | Window             | Alert when            |
| --------------------------------------- | ------------------ | --------------------- |
| Recall within capacity                  | rolling 4 weeks    | below 0.45            |
| ROC-AUC                                 | rolling 4 weeks    | below 0.70 -> retrain |
| Score drift (PSI vs validation period)  | weekly             | above 0.20            |
| Feature drift (max PSI, top features)   | weekly             | above 0.20            |
| Flag rate at the serving threshold      | weekly             | outside 15-35%        |
| Regional recall gap (>= 30 late each)   | rolling 4 weeks    | above 0.10            |
| Regional flag-rate gap (label-free)     | weekly             | 0.15 above reference  |

Labels (late or not) only arrive once the promise window closes (median 23
days), so in live use the performance and recall-gap rows lag by about three
weeks; the drift and flag-rate rows are available on the day of scoring.

Outputs
* docs/evidence/monitoring/  weekly table, alerts, summary JSON, charts
* dashboard/data/            copies for the dashboard's Monitoring page
* build/monitoring_site/     static monitoring site (GitHub Pages) with
                             Evidently AI drift and classification reports
"""
from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import evaluate as ev
from src.models.data import (
    FEATURES,
    TARGET,
    TIME_COL,
    load_modelling_frame,
    split,
)
from src.pipeline import config

ROOT = config.ROOT
EVID = ROOT / "docs" / "evidence" / "monitoring"
DASH = ROOT / "dashboard" / "data"
SITE = ROOT / "build" / "monitoring_site"
MODEL_DIR = ROOT / "models"
REGIONS = ["North", "Northeast", "Central-West", "South", "Southeast"]
# Order-level inputs: a shift here means the incoming orders have changed -> alert.
ORDER_FEATURES = ["promised_days", "distance_km", "seller_hist_handling_days", "seller_hist_late_rate",
                  "freight_total", "price_total", "weight_total_g"]
# Market-context inputs describe delivery "weather" and are meant to move with it (Module 4); their
# drift is reported as context, not alerted on, or every delivery wave would page the team.
CONTEXT_FEATURES = ["state_delivery_days_30d", "promise_slack_days", "state_late_rate_30d", "market_late_rate_14d"]
DRIFT_FEATURES = ORDER_FEATURES + CONTEXT_FEATURES
# Only complete Monday-Sunday weeks: partial weeks at the edges of the period distort PSI.
TH = {"recall": 0.45, "auc": 0.70, "psi": 0.20, "flag_lo": 0.15, "flag_hi": 0.35, "gap": 0.10,
      "flag_gap_increase": 0.15, "min_late": 30}


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population stability index with quantile bins from the reference."""
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def regional_recall_gap(d: pd.DataFrame, flag: str) -> tuple[float, dict]:
    late = d[d[TARGET] == 1]
    by = late.groupby("customer_region")[flag].agg(["mean", "size"])
    judged = by[by["size"] >= TH["min_late"]]
    rec = {r: (round(float(by.loc[r, "mean"]), 3) if r in by.index else None) for r in REGIONS}
    gap = float(judged["mean"].max() - judged["mean"].min()) if len(judged) >= 2 else float("nan")
    return gap, rec


def flag_rate_gap(d: pd.DataFrame, flag: str) -> float:
    sel = d.groupby("customer_region")[flag].mean()
    return float(sel.max() - sel.min())


def score(df: pd.DataFrame, model, post, thr: float) -> pd.DataFrame:
    out = df.copy()
    out["risk_score"] = model.predict_proba(df[FEATURES])[:, 1]
    out["flag_capacity"] = ev.daily_capacity_flags(out["risk_score"], out[TIME_COL])
    out["flag_threshold"] = (out["risk_score"] >= thr).astype(int)
    out["flag_fair"] = np.asarray(post.predict(df[FEATURES], sensitive_features=df["customer_region"],
                                               random_state=0)).astype(int)
    return out


def weekly_table(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    cur = cur.assign(week=pd.to_datetime(cur[TIME_COL]).dt.to_period("W-SUN").dt.start_time)
    days = cur.groupby("week")[TIME_COL].apply(lambda t: pd.to_datetime(t).dt.normalize().nunique())
    weeks = sorted(days[days == 7].index)
    cur = cur[cur["week"].isin(weeks)]
    ref_gap = flag_rate_gap(ref, "flag_threshold")
    rows = []
    for i, w in enumerate(weeks):
        wk = cur[cur["week"] == w]
        win = cur[cur["week"].isin(weeks[max(0, i - 3): i + 1])]
        late = win[win[TARGET] == 1]
        feat_psi = {f: psi(ref[f].to_numpy(float), wk[f].to_numpy(float)) for f in DRIFT_FEATURES}
        gap_std, rec_std = regional_recall_gap(win, "flag_capacity")
        gap_fair, _ = regional_recall_gap(win, "flag_fair")
        rows.append({
            "week": pd.Timestamp(w).date().isoformat(), "orders": len(wk), "late_rate": wk[TARGET].mean(),
            "flag_rate_threshold": wk["flag_threshold"].mean(),
            "recall_4w": late["flag_capacity"].mean() if len(late) else np.nan,
            "recall_fair_4w": late["flag_fair"].mean() if len(late) else np.nan,
            "precision_4w": win.loc[win["flag_capacity"] == 1, TARGET].mean(),
            "auc_4w": roc_auc_score(win[TARGET], win["risk_score"]) if win[TARGET].nunique() == 2 else np.nan,
            "late_orders_4w": len(late),
            "score_psi": psi(ref["risk_score"].to_numpy(), wk["risk_score"].to_numpy()),
            "max_feature_psi": max(feat_psi[f] for f in ORDER_FEATURES),
            "max_psi_feature": max(ORDER_FEATURES, key=feat_psi.get),
            "max_context_psi": max(feat_psi[f] for f in CONTEXT_FEATURES),
            "market_late_rate_14d_mean": wk["market_late_rate_14d"].mean(),
            "recall_gap_4w": gap_std, "recall_gap_fair_4w": gap_fair,
            "flag_rate_gap_regions": flag_rate_gap(wk, "flag_threshold"),
            "flag_rate_gap_reference": ref_gap,
            **{f"recall_4w_{r}": rec_std[r] for r in REGIONS},
            **{f"psi_{f}": v for f, v in feat_psi.items()},
        })
    return pd.DataFrame(rows)


def alerts(wt: pd.DataFrame) -> pd.DataFrame:
    out = []

    def add(w, signal, value, rule, action, kind):
        out.append({"week": w, "signal": signal, "value": round(float(value), 3), "rule": rule,
                    "action": action, "type": kind})

    for _, r in wt.iterrows():
        w = r["week"]
        if r["recall_4w"] < TH["recall"]:
            add(w, "Recall within capacity (4 weeks)", r["recall_4w"], "< 0.45", "Analytics review", "performance")
        if r["auc_4w"] < TH["auc"]:
            add(w, "ROC-AUC (4 weeks)", r["auc_4w"], "< 0.70", "Retrain (src.models.retrain)", "performance")
        if r["score_psi"] > TH["psi"]:
            add(w, "Score drift (PSI)", r["score_psi"], "> 0.20", "Recalibrate threshold; investigate", "drift")
        if r["max_feature_psi"] > TH["psi"]:
            add(w, f"Feature drift: {r['max_psi_feature']}", r["max_feature_psi"], "> 0.20",
                "Investigate data change", "drift")
        if not TH["flag_lo"] <= r["flag_rate_threshold"] <= TH["flag_hi"]:
            add(w, "Flag rate at serving threshold", r["flag_rate_threshold"], "outside 0.15-0.35",
                "Recalibrate threshold", "drift")
        if r["recall_gap_4w"] > TH["gap"]:
            add(w, "Regional recall gap (4 weeks)", r["recall_gap_4w"], "> 0.10",
                "Fairness incident review (Review Board)", "fairness")
        if r["flag_rate_gap_regions"] - r["flag_rate_gap_reference"] > TH["flag_gap_increase"]:
            add(w, "Regional flag-rate gap rising (bias drift)", r["flag_rate_gap_regions"],
                "+0.15 vs reference", "Fairness incident review", "fairness")
    return pd.DataFrame(out, columns=["week", "signal", "value", "rule", "action", "type"])


def charts(wt: pd.DataFrame, path_dir: Path) -> list[str]:
    teal, orange, slate, grey = "#00927F", "#D9691A", "#3E5466", "#9AA7B2"
    x = pd.to_datetime(wt["week"])
    files = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(x, wt["recall_4w"], color=teal, lw=2, marker="o", ms=4, label="Recall, standard flags")
    ax[0].plot(x, wt["recall_fair_4w"], color=orange, lw=2, marker="o", ms=4, label="Recall, fairness-adjusted")
    ax[0].plot(x, wt["auc_4w"], color=slate, lw=1.5, ls="--", label="ROC-AUC")
    ax[0].axhline(TH["recall"], color=teal, lw=1, ls=":")
    ax[0].axhline(TH["auc"], color=slate, lw=1, ls=":")
    ax[0].set_ylim(0, 1)
    ax[0].set_title("Performance (rolling 4 weeks)", loc="left", fontsize=11)
    ax[0].legend(fontsize=8, frameon=False, loc="lower left")
    ax[1].plot(x, wt["score_psi"], color=teal, lw=2, marker="o", ms=4, label="Risk score PSI")
    ax[1].plot(x, wt["max_feature_psi"], color=orange, lw=2, marker="o", ms=4, label="Largest order-feature PSI")
    ax[1].axhline(TH["psi"], color=grey, lw=1, ls=":", label="Alert at 0.20")
    ax[1].set_ylim(0, max(0.5, float(np.nanmax(wt[["score_psi", "max_feature_psi"]].to_numpy())) * 1.1))
    ax[1].set_yscale("symlog", linthresh=0.5)
    ax[1].set_title("Drift vs validation period (weekly)", loc="left", fontsize=11)
    ax[1].legend(fontsize=8, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8)
        a.grid(axis="y", color="#E7ECEF")
    fig.autofmt_xdate()
    fig.tight_layout()
    f = path_dir / "monitoring_performance_drift.png"
    fig.savefig(f, dpi=160)
    plt.close(fig)
    files.append(f.name)

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(x, wt["recall_gap_4w"], color=teal, lw=2, marker="o", ms=4, label="Recall gap, standard flags")
    ax.plot(x, wt["recall_gap_fair_4w"], color=orange, lw=2, marker="o", ms=4, label="Recall gap, fairness-adjusted")
    ax.plot(x, wt["flag_rate_gap_regions"], color=slate, lw=1.5, ls="--", label="Flag-rate gap (label-free)")
    ax.axhline(TH["gap"], color=grey, lw=1, ls=":", label="G5 gate 0.10")
    ax.set_ylim(0, 1)
    ax.set_title("Fairness across regions (judged regions: 30+ late orders in window)", loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="#E7ECEF")
    fig.autofmt_xdate()
    fig.tight_layout()
    f = path_dir / "monitoring_fairness.png"
    fig.savefig(f, dpi=160)
    plt.close(fig)
    files.append(f.name)
    return files


def evidently_reports(ref: pd.DataFrame, cur: pd.DataFrame, out_dir: Path) -> list[dict]:
    """Evidently AI data/prediction drift and classification reports, one per month."""
    from evidently import BinaryClassification, DataDefinition, Dataset, Report
    from evidently.presets import ClassificationPreset, DataDriftPreset

    cols = DRIFT_FEATURES + ["customer_region", "seller_size", "risk_score"]
    keep = cols + [TARGET, "flag_capacity"]
    dd = DataDefinition(numerical_columns=DRIFT_FEATURES + ["risk_score"],
                        categorical_columns=["customer_region", "seller_size"],
                        classification=[BinaryClassification(target=TARGET, prediction_labels="flag_capacity",
                                                              prediction_probas="risk_score")])
    refd = Dataset.from_pandas(ref[keep].reset_index(drop=True), data_definition=dd)
    made = []
    cur = cur.assign(month=pd.to_datetime(cur[TIME_COL]).dt.strftime("%Y-%m"))
    for m, g in cur.groupby("month"):
        snap = Report([DataDriftPreset(columns=cols), ClassificationPreset()]).run(
            Dataset.from_pandas(g[keep].reset_index(drop=True), data_definition=dd), refd)
        name = f"evidently_{m}.html"
        snap.save_html(str(out_dir / name))
        drift = next((x["value"] for x in snap.dict()["metrics"]
                      if x["metric_name"].startswith("DriftedColumnsCount")), {})
        made.append({"month": m, "file": name, "orders": len(g),
                     "drifted_columns": int(drift.get("count", 0)), "drift_share": round(float(drift.get("share", 0)), 3)})
    return made


def site(wt: pd.DataFrame, al: pd.DataFrame, reports: list[dict], summary: dict, images: list[str]) -> None:
    last = wt.iloc[-1]

    def pct(v):
        return "–" if pd.isna(v) else f"{100 * v:.0f}%"

    def tile(label, value, ok):
        cls = "ok" if ok else "bad"
        return f'<div class="tile {cls}"><div class="v">{value}</div><div class="l">{label}</div></div>'

    tiles = "".join([
        tile("Recall, last 4 weeks (alert below 45%)", pct(last["recall_4w"]), last["recall_4w"] >= TH["recall"]),
        tile("ROC-AUC, last 4 weeks (retrain below 0.70)", f"{last['auc_4w']:.2f}", last["auc_4w"] >= TH["auc"]),
        tile("Risk-score drift, PSI (alert above 0.20)", f"{last['score_psi']:.2f}", last["score_psi"] <= TH["psi"]),
        tile("Regional recall gap (gate 10 pts)", f"{100 * last['recall_gap_4w']:.0f} pts",
             last["recall_gap_4w"] <= TH["gap"]),
    ])
    n_cols = len(DRIFT_FEATURES) + 3
    rep = "".join(f"<tr><td>{r['month']}</td><td>{r['orders']:,}</td><td>{r['drifted_columns']} of {n_cols} "
                  f"({100 * r['drift_share']:.0f}%)</td><td><a href='{r['file']}'>Open Evidently report</a></td></tr>"
                  for r in reports)
    alert_rows = "".join(f"<tr><td>{a.week}</td><td>{a.signal}</td><td>{a.value}</td><td>{a.rule}</td>"
                         f"<td>{a.action}</td></tr>" for a in al.itertuples())
    weekly = wt[["week", "orders", "late_rate", "recall_4w", "recall_fair_4w", "auc_4w", "score_psi",
                 "max_feature_psi", "flag_rate_threshold", "recall_gap_4w", "recall_gap_fair_4w"]].copy()
    fmt = {"late_rate": pct, "recall_4w": pct, "recall_fair_4w": pct, "flag_rate_threshold": pct,
           "auc_4w": "{:.2f}".format, "score_psi": "{:.2f}".format, "max_feature_psi": "{:.2f}".format,
           "recall_gap_4w": "{:.2f}".format, "recall_gap_fair_4w": "{:.2f}".format}
    for c, f in fmt.items():
        weekly[c] = weekly[c].map(lambda v, f=f: "–" if pd.isna(v) else f(v))
    weekly.columns = ["Week", "Orders", "Late rate", "Recall (4w)", "Recall fair (4w)", "AUC (4w)", "Score PSI",
                      "Max feature PSI", "Flag rate", "Recall gap", "Recall gap fair"]
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>DRI Model Monitoring</title>
<style>
:root{{--ink:#1B2430;--muted:#5B6B7A;--teal:#00927F;--orange:#D9691A;--bg:#FFFFFF;--soft:#F3F6F7;--line:#DDE3E7;--bad:#B42318;--good:#1E7B34}}
body{{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg)}}
main{{max-width:1100px;margin:0 auto;padding:28px 16px 60px}} h1{{margin:0 0 4px;font-size:28px}}
h2{{margin:34px 0 10px;font-size:20px}} .sub{{color:var(--muted);margin:0 0 18px}}
.banner{{background:#FBF0E7;border-left:4px solid var(--orange);padding:10px 14px;border-radius:4px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0}}
.tile{{background:var(--soft);border-radius:8px;padding:14px;border-top:4px solid var(--good)}}
.tile.bad{{border-top-color:var(--bad)}} .tile .v{{font-size:28px;font-weight:700}} .tile .l{{color:var(--muted);font-size:13px}}
img{{max-width:100%;border:1px solid var(--line);border-radius:6px}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap}} th{{background:var(--soft)}}
a{{color:var(--teal)}} footer{{color:var(--muted);font-size:12px;margin-top:40px}}
</style></head><body><main>
<h1>Delivery Risk Intelligence: model monitoring</h1>
<p class="sub">Champion {summary['model']} · reference: validation period (Mar–May 2018) · monitored: Jun–Aug 2018,
replayed week by week · generated {summary['generated_at']}</p>
<div class="banner"><b>Shadow mode.</b> Scores are not used for decisions. This page checks every monitoring signal
in the Model Card against its action threshold.</div>
<div class="tiles">{tiles}</div>
<h2>Performance and drift</h2><img src="{images[0]}" alt="Recall, ROC-AUC and PSI by week">
<h2>Continuous fairness monitoring (Fairlearn)</h2><img src="{images[1]}" alt="Regional recall gap by week">
<p class="sub">The flag-rate gap needs no labels, so bias drift is visible on the day of scoring. Recall gaps need
outcomes, which arrive about three weeks later.</p>
<h2>Evidently AI reports (data drift, prediction drift, classification quality)</h2>
<div class="tw"><table><tr><th>Month</th><th>Orders</th><th>Drifted columns</th><th>Report</th></tr>{rep}</table></div>
<h2>Alerts raised ({len(al)})</h2>
<div class="tw"><table><tr><th>Week</th><th>Signal</th><th>Value</th><th>Rule</th><th>Action</th></tr>{alert_rows}</table></div>
<h2>Weekly signals</h2><div class="tw">{weekly.to_html(index=False, border=0)}</div>
<footer>Code: src/monitoring/monitor.py ·
<a href="https://github.com/gaintawo/delivery-risk-intelligence">github.com/gaintawo/delivery-risk-intelligence</a> ·
Data: Olist (2018), CC BY-NC-SA 4.0 · BAN6800 Final Project, Gain Tawodzera</footer>
</main></body></html>"""
    (SITE / "index.html").write_text(html, encoding="utf-8")


def main(with_evidently: bool = True) -> dict:
    for d in (EVID, DASH, SITE):
        d.mkdir(parents=True, exist_ok=True)
    meta = json.loads((MODEL_DIR / "dri_late_classifier.json").read_text())
    model = joblib.load(MODEL_DIR / "dri_late_classifier.joblib")
    post = joblib.load(MODEL_DIR / "dri_fairness_postprocessor.joblib")
    parts = split(load_modelling_frame())
    thr = meta["threshold"]
    ref = score(parts["validation"], model, post, thr)
    cur = score(parts["test"], model, post, thr)

    wt = weekly_table(ref, cur)
    al = alerts(wt)
    images = charts(wt, EVID)
    reports = evidently_reports(ref, cur, SITE) if with_evidently else []
    first_retrain = al.loc[al["signal"].str.startswith("ROC-AUC"), "week"].min()
    summary = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "model": f"{meta['model_name']} v{meta['version']}",
        "weeks": len(wt), "alerts": len(al), "alerts_by_type": al["type"].value_counts().to_dict(),
        "first_retrain_alert_week": None if pd.isna(first_retrain) else first_retrain,
        "last_week": wt.iloc[-1][["week", "recall_4w", "recall_fair_4w", "auc_4w", "score_psi", "max_feature_psi",
                                  "max_psi_feature", "max_context_psi", "recall_gap_4w", "recall_gap_fair_4w"]].to_dict(),
        "first_alert_week_by_type": al.groupby("type")["week"].min().to_dict(),
        "reference_market_late_rate_14d": round(float(ref["market_late_rate_14d"].mean()), 4),
        "reference_flag_rate_gap": round(float(wt["flag_rate_gap_reference"].iloc[0]), 3),
        "evidently_reports": reports, "thresholds": TH,
    }
    wt.round(4).to_csv(EVID / "monitoring_weekly.csv", index=False)
    al.to_csv(EVID / "monitoring_alerts.csv", index=False)
    (EVID / "monitoring_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    for f in ("monitoring_weekly.csv", "monitoring_alerts.csv", "monitoring_summary.json"):
        shutil.copy(EVID / f, DASH / f)
    for img in images:
        shutil.copy(EVID / img, SITE / img)
    site(wt, al, reports, summary, images)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("evidently_reports", "thresholds")},
                     indent=1, default=str))
    return summary


if __name__ == "__main__":
    import sys
    main(with_evidently="--no-evidently" not in sys.argv)
