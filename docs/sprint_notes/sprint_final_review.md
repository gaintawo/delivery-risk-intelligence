# Final sprint review (Final Project, November 2026)

| Item | Result | Evidence |
| --- | --- | --- |
| DRI-25 Review Board gate 2 | Held; **not approved** (G1 recall 38% vs 70%; G5 gap 25–67 points vs 10) | Model card; fairness report |
| DRI-27 Ranked actions | Built, ranked by cost (not yet by measured saving: needs a pilot) | `src/actions/recommend.py` |
| DRI-28 Dashboard | Live on Streamlit Community Cloud; 6 pages incl. monitoring | README live links |
| DRI-30 API and Docker image | Single order 0.06 s median (target < 0.5 s) | `api/Dockerfile`, CI Compose test |
| DRI-31 Deploy with batch and latency tests | Batch: a full day (about 207 orders) scored and explained in 0.002 s; the 18,603-order test period in 0.21 s. Live deployment on Render | `render.yaml` |
| DRI-32 Release v1.0.0 | Tag `v1.0.0` links code, model registry version 3 and data fingerprints | CHANGELOG |
| New: monitoring, retraining, explanation store, Compose, CD, governance | Done | README "Monitoring, retraining and governance" |

**Not done, carried forward:** DRI-13 EDA notebook, DRI-14 baseline KPI report (baselines now in the Final report), DRI-26 seller clustering, DRI-29 persona review session, DRI-19 recall target (not met by any model), DRI-15 DPO sign-off record.

**Retrospective.** What worked: time-aware validation and honest gates stopped an unfair model reaching users. What did not: purchase-time data alone cannot reach the 70% recall target; retraining did not fix the August decline. Next: courier tracking data and the hand-over scoring point.
