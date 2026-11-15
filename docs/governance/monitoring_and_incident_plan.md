# Implementation, Monitoring, Incident Response and Decommissioning Plan

Delivery Risk Intelligence · builds on the Module 4 Model Card monitoring hooks and the Module 5 dashboard.

## 1. Deployment plan

| Phase | What happens | Exit criterion | Owner |
| --- | --- | --- | --- |
| 0 Shadow (now) | API, dashboard and monitoring live; scores stored, never acted on | Gate 2: recall ≥ 55%, regional gap ≤ 10 points, Review Board approval | Model owner |
| 1 Pilot (8 weeks) | One operations team acts on flags for half of the days, with the other half as control (randomised by day) | Measured effect on late rate (G2) and 1–2★ reviews (G4); no fairness or privacy incident | Operations lead |
| 2 Rollout | All teams; seller-facing explanations and appeal route live | Quarterly Review Board (gate 3) | Head of Logistics |

Architecture in production: pipeline (Prefect, nightly) → model-ready table → scoring API (FastAPI container) → daily review list and dashboard; explanation store and Prometheus metrics from the API; monitoring job weekly; retraining job monthly. Locally the whole stack runs with `docker compose up --build`.

## 2. Performance tracking KPIs

| KPI | Definition | Target / alert | Frequency | Source |
| --- | --- | --- | --- | --- |
| Late orders caught (G1) | Recall within the daily 25% capacity, rolling 4 weeks | Target 70%; alert < 45% | Weekly (3-week label lag) | `monitor.py` |
| Ranking quality | ROC-AUC, rolling 4 weeks | Retrain < 0.70 | Weekly | `monitor.py` |
| Late delivery rate (G2) | Share of orders after promise | ≤ 5% | Monthly | Pipeline |
| Promise gap (G3) | Mean promised minus actual days | 20% smaller | Monthly | Pipeline |
| Poor reviews on flagged orders (G4) | Share of 1–2★ reviews | 25% lower than control | Pilot | Pipeline |
| Regional recall gap (G5) | Max minus min recall, regions with 30+ late orders | ≤ 10 points | Weekly | `monitor.py`, Fairlearn |
| Scoring latency | 95th percentile | < 0.5 s | Live | Prometheus/Grafana |
| Staff override rate (ER6) | Flags dismissed without action | Review if > 50% | Weekly | Operations log |

## 3. Continuous fairness monitoring

* **Weekly, label-free:** share of orders flagged per region, compared with the reference period (Prometheus alert `DRIRegionalFlagGap` live; `monitor.py` weekly). Detects bias drift on the day of scoring.
* **Weekly, outcome-based (3-week lag):** recall and false-alarm rate by region and seller size with Fairlearn `MetricFrame`, rolling 4 weeks, judged only where a group has 30+ late orders.
* **Monthly:** full fairness report regenerated with every retrained candidate; a candidate that widens the regional gap by more than 5 points fails its promotion gate.
* **Replay result (Jun–Aug 2018):** the standard model's regional recall gap stayed above 10 points in every judged window (16–73 points); fairness-adjusted thresholds kept it at 10–29 points, reaching 10 points only in the last window. 11 fairness alerts were raised.

## 4. Model drift and bias drift detection

| Signal | Method | Threshold | Replay finding |
| --- | --- | --- | --- |
| Prediction drift | PSI of the risk score vs the validation period | > 0.20 | Fired from week 1 (PSI 1.8): scores fell as late deliveries calmed, so the serving threshold flagged only 6–13% of orders: recalibrate |
| Data drift (order inputs) | PSI per input; Evidently AI reports per month | > 0.20 | Promised days drifted (PSI up to 1.2): Olist shortened promises in August 2018 |
| Market context | PSI of recent delivery-time features | Reported, not alerted | Large by design: these features track delivery "weather" |
| Concept drift | Rolling ROC-AUC and recall once labels arrive | AUC < 0.70; recall < 45% | AUC alert from 23 July, recall alert from 30 July |
| Bias drift | Regional flag-rate gap vs reference; rolling recall gap | +0.15 vs reference; > 0.10 | Flag-rate gap rose to 0.81 in early July |

Key lesson: label-free drift signals warned **seven weeks before** the outcome-based performance alerts.

## 5. Retraining

Monthly job (`src/models/retrain.py`, GitHub Actions `retrain.yml`): retrain on the latest 13 labelled months, compare with the current model on the most recent labelled month, and register a candidate only if it passes all gates (recall, ROC-AUC, fairness). Promotion to champion is a Review Board decision. **Replay finding:** every candidate passed its gates, but retraining did not stop the August decline (28% recall vs 32% for the untouched model). Retraining is necessary upkeep, not a cure. Recovering performance needs data the model does not yet have, such as courier tracking.

## 6. Incident response plan (ethical and technical failures)

| Severity | Examples | First response | Who decides | Time limit |
| --- | --- | --- | --- | --- |
| 1 Critical | Personal data exposed; flags used to penalise sellers; discriminatory outcome reported publicly | Stop scoring (kill switch: suspend the API service on Render, or `docker compose stop api`; the review list falls back to the manual rule); preserve logs | DPO + Review Board chair | Contain within 4 hours; ANPD notice within 3 working days if personal data |
| 2 High | Regional recall gap > 20 points in live use; override rate > 70%; model serving wrong version | Switch to fairness-adjusted thresholds or roll back to the previous registry version; notify Review Board | Model owner + Review Board | 2 working days |
| 3 Medium | Drift or performance alert; latency above 0.5 s | Investigate; recalibrate threshold or retrain | Model owner | Next weekly review |

Steps for every incident: **detect** (monitoring alert, staff report, seller or customer complaint) → **triage** severity → **contain** → **investigate** using the explanation store and audit log → **remediate** → **communicate** to affected people in plain language → **post-incident review** within 10 working days, published to the Review Board and recorded in the RAID log.

## 7. Decommissioning plan

Retire the model when any of these holds: it fails its fairness gate for 3 consecutive months after mitigation; recall stays below 45% after two retraining cycles; the business process changes (for example courier tracking makes purchase-time scoring obsolete); or the Review Board orders it.

1. **Decide and announce:** Review Board decision recorded; users and the seller representative told 30 days ahead (immediately for Severity 1).
2. **Fall back:** the daily review list reverts to the documented manual rule (seller's past late rate), so no team is left without a process.
3. **Switch off:** stop the API and dashboard deployments; archive container images; set the MLflow registry version to "Archived" with the reason.
4. **Keep what audit needs:** model card, fairness reports, monitoring history and explanation store kept read-only for 24 months, then deleted. Training data deleted on its own 24-month schedule.
5. **Learn:** publish a short retirement note (what worked, what harmed, what to do differently) to the repository and the Review Board.
