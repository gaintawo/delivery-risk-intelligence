# Integration audit: reconciling Modules 1–5

Before writing the Final Project report, each module was checked against the others. Every contradiction found is listed here with how it was resolved. "Fixed" means the code or documents were changed; "Explained" means the difference is intended and is now stated in the report.

| # | Where | Contradiction | Resolution | Type |
| --- | --- | --- | --- | --- |
| 1 | M1 vision vs M3 data | Late rate "about 7–8%" | Measured 8.1% of delivered orders (2017–Aug 2018), but only 5.5% in the test months. G2 (≤ 5%) is therefore judged against a control group in a pilot, not against the historical baseline | Explained |
| 2 | M1 risk score inputs vs M4 drivers | M1 expected seller speed, distance, weight and size to drive risk | SHAP shows the promise length against recent delivery times drives risk; weight and size barely matter | Explained (insight) |
| 3 | M1/M2 stack vs M3 | Google Cloud Storage, PostgreSQL and dbt planned | Public data needs no cloud bucket; SQLite by default, PostgreSQL via `DATABASE_URL`; transformations in pandas plus a SQL view, not dbt | Explained; architecture doc §7 |
| 4 | M1 versioning vs M3 | DVC planned for data versions | Each file's SHA-256 fingerprint is logged in `lineage.json` and the audit log, which gives the same traceability for a fixed public dataset | Explained |
| 5 | M1 "Going live" | Google Cloud Run | Render free tier runs the same container (`api/Dockerfile`, `render.yaml`); images also published to GitHub Container Registry | Explained |
| 6 | M1 FR4 vs M2–M5 | Recommendation engine planned, `src/actions/` left empty | Built: rule-based, support-only actions in `src/actions/recommend.py`, returned by the API | **Fixed** |
| 7 | M1 FR3 vs M4 API | "Expected days late" required; regressor trained in M4 but not served | API now returns `expected_delivery_days` and `expected_days_late` | **Fixed** |
| 8 | M1/M2 two scoring points | Score at purchase and again at courier hand-over | Only purchase-time scoring built; the 3 hand-over (H) features exist in the pipeline for a second-stage model | Explained; next step |
| 9 | M3 vs M4 row counts | 96,470 labelled orders (M3 view) vs 96,203 modelled (M4) | M4 drops 267 launch-period orders from 2016 | Explained |
| 10 | M3 vs M4 quality checks | "35/35 passed" (M3) vs "22/22" (M4 report) | 35 checks in 3 suites; 22 are in the model-ready suite that acts as the hard gate | Explained |
| 11 | M3 governance vs M5 dashboard | M3: row-level table is restricted (k = 1). M5 published a row-level test extract with exact dates and review scores | New "public de-identified extract" class: purchase week instead of date, review scores removed, no keys; governance doc updated | **Fixed** |
| 12 | M3 bias check vs M4 fairness | M3 flagged North, Northeast and Central-West as under-represented | Carried into M4 (North has 9 late test orders: not judged) and into monitoring (30 late orders minimum) | Linked |
| 13 | M1 fairness objective (small sellers) vs M4 | "Small sellers not flagged more often" | Small sellers are flagged less, but their late orders are caught least (34%); new sellers get most false alarms (31%). Monitored by seller size | Explained |
| 14 | M1 risk register vs M4/Final | Concept drift not in the register | Added R03 drift as measured risk: ROC-AUC fell to 0.65 within three months; monitoring and retraining added | **Fixed** |
| 15 | M4 API vs M5 deck | API flags at a fixed 0.50 cut-off; batch list uses top 25% per day | Both kept: batch rule for the daily list, fixed threshold for single orders. Monitoring showed the fixed cut-off flagged 6–38% of orders per week, so it is recalibrated monthly | Explained; monitored |
| 16 | M4 report vs M5 deck | Example "missed" order differs (score 0.60 vs 0.40) | M5 uses a typical miss; the M4 example was a near-miss | Explained |
| 17 | Brief vs build | LIME requested alongside SHAP | For a linear model, exact SHAP values are available, so LIME's local approximation adds nothing; SHAP plus DiCE counterfactuals used | Explained |
| 18 | M1 NFR vs build | Single score in under 0.5 s | Measured median 0.06 s, 95th percentile 0.09 s; alert in Prometheus | Verified |
| 19 | M1 dashboard | "Streamlit or Power BI" | Streamlit only; CSV export of the public extract serves Power BI if needed | Explained |
| 20 | M2 board vs status | Sprint 5 issues open | Board and backlog updated for the Final sprint | **Fixed** |

Result: no remaining contradiction between what the report claims, what the code does and what the dashboard shows. All figures in the report are produced by the code in this repository.
