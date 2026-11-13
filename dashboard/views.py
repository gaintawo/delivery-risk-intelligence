"""Pages of the Delivery Risk Intelligence dashboard (BAN6800 Module 5). Entry point: dashboard/app.py.

Reads the Module 4 champion model (models/) and a de-identified, scored copy of
the held-out test period (dashboard/data/, built by src.models.export_dashboard).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.explain import PLAIN, Explainer

DATA, MODELS = ROOT / "dashboard" / "data", ROOT / "models"
TEAL, ORANGE, SLATE, GREY, INK = "#00927F", "#D9691A", "#3E5466", "#C9D2DA", "#1B2430"
REGIONS = ["North", "Northeast", "Central-West", "South", "Southeast"]
SELLER_SIZES = ["new", "small", "large"]
MIN_LATE = 30          # a group needs at least this many late orders to be judged (G5 rule)
G5_GATE = 0.10



# ------------------------------------------------------------------ data
@st.cache_resource
def load_model():
    meta = json.loads((MODELS / "dri_late_classifier.json").read_text())
    model = joblib.load(MODELS / "dri_late_classifier.joblib")
    post = joblib.load(MODELS / "dri_fairness_postprocessor.joblib")
    expl = Explainer(model, None, meta["features"], mean_encoded=meta.get("encoded_mean"))
    return meta, model, post, expl


@st.cache_data
def load_orders() -> pd.DataFrame:
    d = pd.read_csv(DATA / "test_scored.csv.gz", dtype={"example": "string"}, parse_dates=["purchase_date"])
    d["example"] = d["example"].fillna("")
    d["month"] = d["purchase_date"].dt.strftime("%b %Y")
    return d


@st.cache_data
def load_shap() -> pd.DataFrame:
    return pd.read_csv(DATA / "test_shap.csv.gz")


meta, model, post, expl = load_model()
orders = load_orders()
FEATURES = meta["features"]


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


def group_table(df: pd.DataFrame, by: str, decision: str, order: list[str]) -> pd.DataFrame:
    """Fairness metrics per group, computed with Fairlearn's MetricFrame."""
    from fairlearn.metrics import (
        MetricFrame,
        false_positive_rate,
        selection_rate,
        true_positive_rate,
    )

    mf = MetricFrame(metrics={"Share of orders flagged": selection_rate,
                              "Late orders caught (recall)": true_positive_rate,
                              "On-time orders flagged (false alarms)": false_positive_rate},
                     y_true=df["is_late"], y_pred=df[decision], sensitive_features=df[by])
    t = mf.by_group.reindex([g for g in order if g in mf.by_group.index])
    counts = df.groupby(by).agg(orders=("is_late", "size"), late_orders=("is_late", "sum"))
    return t.join(counts)


def bar_layout(fig: go.Figure, height: int = 360, pct_axis: bool = True, **kw) -> go.Figure:
    fig.update_layout(height=height, margin={"l": 10, "r": 10, "t": 40, "b": 10}, plot_bgcolor="white",
                      paper_bgcolor="white", font={"color": INK, "size": 13}, bargap=0.3, bargroupgap=0.08,
                      legend={"orientation": "h", "y": 1.12, "x": 0}, **kw)
    fig.update_yaxes(gridcolor="#E7ECEF", zeroline=False, tickformat=".0%" if pct_axis else None)
    fig.update_xaxes(showgrid=False)
    return fig


def status_banner():
    st.markdown('<div class="warn"><b>Status: shadow mode.</b> The model scores orders, but no one acts on the '
                "scores yet. It has not met the recall target (G1) or the regional fairness gate (G5).</div>",
                unsafe_allow_html=True)


# ------------------------------------------------------------------ pages
def page_overview():
    st.title("Delivery Risk Intelligence")
    st.caption("Which Olist orders are likely to arrive late, scored on the day they are placed · "
               "held-out test period, June–August 2018 · model v3 (logistic regression)")
    status_banner()

    c1, c2, c3 = st.columns([2, 2, 3])
    months = ["All months"] + list(orders["month"].unique())
    month = c1.selectbox("Month", months)
    region = c2.selectbox("Customer region", ["All regions"] + REGIONS)
    decision = c3.radio("Flag rule", ["Standard (top 25% each day)", "Fairness-adjusted thresholds"],
                        horizontal=True)
    col = "flagged" if decision.startswith("Standard") else "flagged_fair"
    d = orders
    if month != "All months":
        d = d[d["month"] == month]
    if region != "All regions":
        d = d[d["customer_region"] == region]

    late = d[d["is_late"] == 1]
    caught = late[col].mean() if len(late) else np.nan
    flagged = d[d[col] == 1]
    m = st.columns(4)
    m[0].metric("Orders", f"{len(d):,}", help="Delivered orders purchased in the selected period")
    m[1].metric("Arrived late", f"{len(late):,}", pct(len(late) / max(len(d), 1), 1) + " of orders",
                delta_color="off", delta_arrow="off")
    m[2].metric("Late orders flagged in advance", pct(caught) if len(late) else "–",
                "goal G1: 70% or more", delta_color="off", delta_arrow="off")
    m[3].metric("Flagged orders that were late", pct(flagged["is_late"].mean()) if len(flagged) else "–",
                f"vs {pct(d['is_late'].mean(), 1)} of all orders", delta_color="off", delta_arrow="off",
                help="Precision: most flagged orders still arrive on time, so actions must be cheap and helpful")

    st.subheader("Of the late orders, how many did the model flag in advance?")
    g = (orders if region == "All regions" else orders[orders["customer_region"] == region])
    g = g[g["is_late"] == 1].groupby("month", sort=False).agg(late=("is_late", "size"), caught=(col, "sum"))
    fig = go.Figure()
    fig.add_bar(x=g.index, y=g["caught"], name="Flagged in advance", marker_color=TEAL,
                hovertemplate="%{x}: %{y} late orders flagged<extra></extra>")
    fig.add_bar(x=g.index, y=g["late"] - g["caught"], name="Missed", marker_color=GREY,
                hovertemplate="%{x}: %{y} late orders missed<extra></extra>")
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    bar_layout(fig, 330, pct_axis=False, yaxis_title="Late orders")
    st.plotly_chart(fig, width="stretch")
    rate = (g["caught"] / g["late"]).round(2)
    st.caption("Share caught by month: " + ", ".join(f"{k} {pct(v)}" for k, v in rate.items())
               + ". The share falls each month after training: the model needs regular retraining.")

    st.subheader("Progress against the Module 1 goals")
    lt = orders[orders["is_late"] == 1]
    gaps = {}
    for c in ("flagged", "flagged_fair"):
        t = group_table(orders, "customer_region", c, REGIONS)
        r = t.loc[t["late_orders"] >= MIN_LATE, "Late orders caught (recall)"]
        gaps[c] = 100 * (r.max() - r.min())
    goals = pd.DataFrame([
        ["G1 Spot late orders early", "≥ 70% of late orders flagged",
         f"{pct(lt['flagged'].mean(), 1)} ({pct(lt['flagged_fair'].mean(), 1)} fairness-adjusted)", "Not met"],
        ["G2 Fewer late deliveries", "≤ 5% late", "Needs live actions: not measurable in shadow mode", "Not yet tested"],
        ["G3 More honest dates", "Promise gap 20% smaller", "Delivery-days model error 3.4 days vs 5.0 naive", "Partly ready"],
        ["G4 Happier customers", "25% fewer 1–2★ reviews on flagged orders", "Needs live actions", "Not yet tested"],
        ["G5 Fair to every region", "Recall gap ≤ 10 points",
         f"{gaps['flagged']:.1f} points ({gaps['flagged_fair']:.1f} fairness-adjusted)", "Not met"],
    ], columns=["Goal", "Target", "Result on test period", "Status"])
    st.dataframe(goals, hide_index=True, width="stretch")


def page_drivers():
    st.title("What drives a late-delivery risk score?")
    st.markdown('<div class="note">Each bar shows how much a factor moves the risk score on average '
                "(mean absolute SHAP value, 4,000 test orders). Longer bars matter more. The direction "
                "column says which way the factor usually pushes.</div>", unsafe_allow_html=True)
    imp = pd.read_csv(DATA / "shap_global_importance.csv", index_col=0)["mean_abs_shap"]
    sh = load_shap()
    sample = orders[FEATURES].sample(2000, random_state=0)
    base_p = model.predict_proba(sample)[:, 1].mean()
    direction = {}
    for f in imp.index[:10]:
        x = orders[f]
        if pd.api.types.is_numeric_dtype(x):
            hi = sample.copy()
            binary = set(x.dropna().unique()) <= {0, 1}
            hi[f] = 1 if binary else hi[f] + x.std()      # what the model does when the factor goes up
            up = model.predict_proba(hi)[:, 1].mean() > (base_p if not binary else
                                                         model.predict_proba(sample.assign(**{f: 0}))[:, 1].mean())
            direction[f] = ("Yes" if binary else "Higher value") + (" → higher risk" if up else " → lower risk")
        else:
            top = sh[f"shap__{f}"].groupby(x).mean().sort_values()
            direction[f] = f"Highest: {top.index[-1]} · lowest: {top.index[0]}"
    top = imp.head(10)[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=[PLAIN.get(f, f) for f in top.index], orientation="h",
                           marker_color=TEAL, customdata=[direction[f] for f in top.index],
                           hovertemplate="%{y}<br>Average impact %{x:.2f}<br>%{customdata}<extra></extra>"))
    bar_layout(fig, 430, pct_axis=False, xaxis_title="Average impact on risk (log-odds)")
    fig.update_xaxes(showgrid=True, gridcolor="#E7ECEF")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame({"Factor": [PLAIN.get(f, f) for f in imp.index[:10]],
                               "Usual direction": [direction[f] for f in imp.index[:10]]}),
                 hide_index=True, width="stretch")
    st.markdown("**In business terms:** risk is highest when the promised date is tight compared with how long "
                "deliveries to that state have recently taken, when the route is long or crosses states, in the "
                "Black Friday–Christmas peak, and when the seller has been slow to hand parcels over.")


def explain_row(row: pd.DataFrame) -> tuple[float, pd.Series]:
    p = float(model.predict_proba(row[FEATURES])[0, 1])
    return p, expl.grouped(row).iloc[0]


def reasons_chart(g: pd.Series, k: int = 6) -> go.Figure:
    top = g.reindex(g.abs().sort_values(ascending=False).index).head(k)[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=[PLAIN.get(f, f) for f in top.index], orientation="h",
                           marker_color=[ORANGE if v > 0 else TEAL for v in top.values],
                           hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
    bar_layout(fig, 300, pct_axis=False, xaxis_title="← lowers risk   ·   raises risk →")
    fig.update_xaxes(zeroline=True, zerolinecolor=SLATE, showgrid=True, gridcolor="#E7ECEF")
    return fig


def page_explorer():
    st.title("Order explorer and what-if analysis")
    st.caption("Pick an order, see why it scored as it did, then change the facts Olist controls.")
    thr = meta["threshold"]
    ex = orders[orders["example"] != ""].sort_values("example")
    c1, c2 = st.columns([3, 2])
    choice = c1.selectbox("Order", list(ex["example"]) + ["Random order from the test period"])
    if choice.startswith("Random"):
        seed = c2.number_input("Order number (0–18,602)", 0, len(orders) - 1, 1234)
        idx = int(seed)
    else:
        idx = int(ex.index[ex["example"] == choice][0])
    base = orders.loc[[idx], FEATURES].copy()
    o = orders.loc[idx]
    p0, g0 = explain_row(base)

    a, b, c, d = st.columns(4)
    a.metric("Risk score", f"{p0:.2f}", "flagged" if p0 >= thr else "not flagged", delta_color="off", delta_arrow="off")
    b.metric("Customer", f"{o['customer_state']} · {o['customer_region']}")
    c.metric("Promised", f"{o['promised_days']:.0f} days", f"route {o['distance_km']:,.0f} km", delta_color="off", delta_arrow="off")
    d.metric("What happened", "Late" if o["is_late"] else "On time", f"delivered in {o['delivery_days']:.0f} days",
             delta_color="off", delta_arrow="off")
    st.plotly_chart(reasons_chart(g0), width="stretch")

    st.subheader("What if…?")
    st.caption("Move the sliders: the model re-scores the order instantly. Promise slack moves with the promise.")
    w1, w2, w3 = st.columns(3)
    pdays = w1.slider("Days promised to the customer", 3, 60, round(o["promised_days"]))
    late_rate = w2.slider("Seller's past late rate", 0.0, 0.6,
                          float(round(o["seller_hist_late_rate"], 2)) if pd.notna(o["seller_hist_late_rate"]) else 0.08,
                          0.01)
    handling = w3.slider("Seller's past handling time (days)", 0.0, 10.0,
                         float(round(o["seller_hist_handling_days"], 1)) if pd.notna(o["seller_hist_handling_days"]) else 2.5,
                         0.1)
    w4, w5 = st.columns(2)
    dist = w4.slider("Seller-to-customer distance (km)", 0, 3500, int(o["distance_km"]), 50)
    peak = w5.toggle("Black Friday to Christmas period", bool(o["is_peak_season"]))

    new = base.copy()
    new["promised_days"] = pdays
    if pd.notna(o["state_delivery_days_30d"]):
        new["promise_slack_days"] = pdays - o["state_delivery_days_30d"]
    new["seller_hist_late_rate"] = late_rate
    new["seller_hist_handling_days"] = handling
    new["distance_km"] = dist
    new["is_peak_season"] = int(peak)
    p1, _g1 = explain_row(new)
    r1, r2, r3 = st.columns(3)
    r1.metric("Original risk", f"{p0:.2f}")
    r2.metric("New risk", f"{p1:.2f}", f"{p1 - p0:+.2f}", delta_color="inverse")
    r3.metric("Daily review list", "Flagged" if p1 >= thr else "Not flagged", f"cut-off {thr:.2f}", delta_color="off", delta_arrow="off")
    if p0 >= thr > p1:
        st.success(f"With these changes the order would drop off the review list (risk {p1:.2f} < {thr:.2f}).")
    st.markdown('<div class="note">The what-if shows how the <i>model</i> responds. It does not prove that '
                "changing a fact would change the real delivery. A longer promise lowers the score, but it also "
                "gives the customer a later date: the charter (ER5) forbids quietly padding promises for remote "
                "customers.</div>", unsafe_allow_html=True)


def page_fairness():
    st.title("Ethical compliance: is the model fair across groups?")
    st.markdown('<div class="note">Being flagged <b>helps</b> an order: it gets early attention. So the main '
                "fairness question is: <b>are late orders caught equally well in every group?</b> "
                "(equal opportunity, goal G5: gap of 10 points or less). Metrics are computed live with "
                "Fairlearn's MetricFrame on the test period.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 3])
    attr_label = c1.radio("Group by", ["Customer region", "Seller size"], horizontal=True)
    rule = c2.radio("Flag rule", ["Standard (top 25% each day)", "Fairness-adjusted thresholds"], horizontal=True)
    attr, order = (("customer_region", REGIONS) if attr_label == "Customer region"
                   else ("seller_size", SELLER_SIZES))
    t_std = group_table(orders, attr, "flagged", order)
    t_fair = group_table(orders, attr, "flagged_fair", order)
    t = t_std if rule.startswith("Standard") else t_fair
    judged = t[t["late_orders"] >= MIN_LATE]
    rec = judged["Late orders caught (recall)"]
    gap = rec.max() - rec.min()
    sel = t["Share of orders flagged"]
    fpr = t["On-time orders flagged (false alarms)"]

    m = st.columns(4)
    m[0].metric("Recall gap (judged groups)", f"{100 * gap:.1f} pts", "gate: 10 pts or less", delta_color="off", delta_arrow="off")
    m[1].metric("G5 gate", "Pass" if gap <= G5_GATE else "Fail")
    m[2].metric("Flag-rate ratio (disparate impact)", f"{sel.min() / sel.max():.2f}", "1.00 = equal", delta_color="off", delta_arrow="off")
    eo = max(rec.max() - rec.min(), fpr[judged.index].max() - fpr[judged.index].min())
    m[3].metric("Equalized-odds difference (judged)", f"{eo:.2f}", "0 = equal", delta_color="off", delta_arrow="off")

    fig = go.Figure()
    for lab, tt, colr in (("Standard", t_std, TEAL), ("Fairness-adjusted", t_fair, ORANGE)):
        fig.add_bar(x=[f"{g}<br>({int(n)} late{'' if n >= MIN_LATE else ', not judged'})"
                       for g, n in zip(tt.index, tt["late_orders"])],
                    y=tt["Late orders caught (recall)"], name=lab, marker_color=colr,
                    hovertemplate="%{x}<br>" + lab + ": %{y:.0%} of late orders caught<extra></extra>")
    st.markdown("**Late orders caught, by group**")
    fig.add_hline(y=0.70, line_dash="dot", line_color=SLATE, annotation_text="G1 target 70%",
                  annotation_position="top left")
    bar_layout(fig, 380, yaxis_range=[0, 1.05], yaxis_title="Late orders caught")
    st.plotly_chart(fig, width="stretch")

    show = t.copy()
    for col in ["Share of orders flagged", "Late orders caught (recall)", "On-time orders flagged (false alarms)"]:
        show[col] = show[col].map(lambda v: pct(v, 1))
    show["Judged for G5?"] = np.where(t["late_orders"] >= MIN_LATE, "Yes", "Too few late orders")
    show = show.rename(columns={"orders": "Orders", "late_orders": "Late orders"}).rename_axis(attr_label)
    st.dataframe(show, width="stretch")
    small = t.index[t["late_orders"] < MIN_LATE].tolist()
    if small:
        st.caption(f"Not judged: {', '.join(small)} — fewer than {MIN_LATE} late orders, so the result could be chance.")

    st.subheader("What was found, and what was tried")
    st.markdown(
        "- **Found:** the standard model flags 78% of Northeast orders but 18% of Southeast orders. It learned that "
        "long routes and the Northeast meant lateness (14% of Northeast orders were late across 2017–18), but by "
        "mid-2018 Northeast orders were late *less* often than Southeast ones (5.4% vs 6.2%).\n"
        "- **Tried 1 – remove location columns:** the gap barely moved (67 → 62 points), because distance carries "
        "the same signal.\n"
        "- **Tried 2 – fairness-adjusted thresholds** (Fairlearn ThresholdOptimizer, a separate cut-off per region "
        "set on earlier data): the gap fell to 25 points and the share of late orders caught *rose* to 51%.\n"
        "- **Still open:** 25 points is above the 10-point gate, so the Review Board gate stays closed.")
    ex = pd.read_csv(DATA / "fairness_region_unaware.csv")
    with st.expander("Proxy test detail (model without location columns)"):
        st.dataframe(ex, hide_index=True, width="stretch")


def page_transparency():
    st.title("Transparency statement")
    a, b = st.columns(2)
    with a:
        st.subheader("What the model can do")
        st.markdown("- Score every order on the day it is placed, using only facts known at that moment\n"
                    "- Rank a day's orders so the team can review the riskiest 25% (about 50 a day)\n"
                    "- Catch about 2 in 5 late orders (about 1 in 2 with fairness-adjusted thresholds), "
                    "against 1 in 4 for the best simple rule\n"
                    "- Give the top three reasons behind every score in plain words")
    with b:
        st.subheader("What it cannot do")
        st.markdown("- Catch most late orders: 6 in 10 are still missed\n"
                    "- Be precise: about 11 in 12 flagged orders arrive on time anyway\n"
                    "- See courier events, weather or strikes: none are in the data\n"
                    "- Stay accurate without retraining: performance fell each month after training\n"
                    "- Be used to judge or penalise sellers or customers")
    st.subheader("Commitments (Module 1 Ethical AI Charter)")
    st.markdown("1. **Fairness** – no customer gets worse service because of where they live; release is blocked "
                "until the regional gap is 10 points or less.\n"
                "2. **Transparency** – every score shows its top three reasons.\n"
                "3. **Accountability** – the tool only advises; a named person makes each decision.\n"
                "4. **Privacy** – no names, customer or seller IDs, or postcodes are shown; location is state and "
                "region only (LGPD).\n"
                "5. **Support, not punishment** – a flag means help for the order, never a penalty for the seller.")
    st.subheader("About the data and model")
    tm = meta["test_metrics"]
    st.markdown(f"- Model: {meta['model_name']} v{meta['version']} ({meta['algorithm']}), MLflow registry\n"
                f"- Trained on 57,050 orders (Jan 2017–Feb 2018); tested on 18,603 later orders (Jun–Aug 2018)\n"
                f"- Test ROC-AUC {tm.get('roc_auc', 0.7126):.2f}: the model ranks a random late order above a "
                "random on-time order about 7 times in 10\n"
                "- Data: Olist Brazilian e-commerce public dataset (2016–2018), CC BY-NC-SA 4.0\n"
                "- Code, model card and fairness report: "
                "[github.com/gaintawo/delivery-risk-intelligence](https://github.com/gaintawo/delivery-risk-intelligence)")
