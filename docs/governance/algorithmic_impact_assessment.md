# Algorithmic Impact Assessment (AIA)

Delivery Risk Intelligence late-delivery classifier, MLflow registry version 3 · assessed November 2026 · format adapted from the Government of Canada (2023) Algorithmic Impact Assessment tool and the NIST AI RMF (2023) Map function.

## 1. System description

| Item | Answer |
| --- | --- |
| Purpose | Estimate, at purchase, the chance an order arrives after its promised date, so staff can act early |
| Decision supported | Which orders a delivery team reviews each day (top 25%, about 50 a day) and which low-cost action to take |
| Level of automation | Advisory only. No automatic action on customers or sellers. Currently shadow mode: scores are not used at all |
| Data | Olist public e-commerce data 2016–2018 (99,441 orders), pseudonymised; no names, IDs, postcodes or coordinates in the model |
| Model | Logistic regression on 32 purchase-time inputs; exact SHAP explanations; optional fairness thresholds per region |
| Reversibility | Fully reversible: a flag can be ignored; no record is changed by the model |

## 2. Who is affected

| Group | How they are affected | Direction |
| --- | --- | --- |
| Customers | Flagged orders may get an earlier update or a corrected date; unflagged late orders get no help | Benefit if flagged; harm if missed |
| Customers in the North and Northeast | The model over-flags them (78% of Northeast orders); they may receive more updates, and could face padded dates if ER5 controls fail | Mixed |
| Customers in the Southeast | Late orders caught least often (30%) though 8 in 10 late orders are here | Harm through omission |
| Sellers, especially new and small | New sellers get the most false alarms (31%); a flag could be misread as poor performance | Risk of unfair reputation harm |
| Staff | Extra review work; most flags (11 in 12) are false alarms; risk of over-trust (ER6) | Workload and judgement |

## 3. Impact scoring

Scale 1 (low) to 4 (very high). Score = likelihood × severity.

| Risk (Module 1 register) | Likelihood | Severity | Score | Mitigation | Residual |
| --- | --- | --- | --- | --- | --- |
| ER1 Regional bias | 4 (measured) | 3 | 12 | Fairness thresholds; release gate; weekly monitoring | **High**: gap 25 points after mitigation |
| ER2 Hidden proxy (distance) | 4 (measured) | 2 | 8 | Proxy test recorded; thresholds act on outcomes, not inputs | Medium |
| ER3 Unfair to small or new sellers | 3 | 3 | 9 | Support-only actions; appeal via stored explanation; seller-size monitoring | Medium |
| ER4 Re-identification | 1 | 3 | 3 | Pseudonymisation, deletion, suppression, public extract coarsened | Low |
| ER5 Misleading dates | 2 | 3 | 6 | No region-specific padding; expected delivery days published | Low–medium |
| ER6 Blind trust | 3 | 2 | 6 | Reasons shown; override log; low precision stated openly | Medium |
| ER7 Legal breach | 1 | 4 | 4 | Compliance report; DPO sign-off per release | Low |
| R03 Concept drift (new) | 4 (measured) | 2 | 8 | Weekly drift monitoring; monthly retraining with gates | Medium: retraining alone did not recover August |

## 4. Impact level and conditions

Overall level: **moderate impact** (advisory, reversible, no access to essential services), but with a **measured, unresolved fairness harm**. Conditions before any use beyond shadow mode:

1. Regional recall gap of 10 points or less, judged on regions with 30 or more late orders.
2. Recall of 55% or more within the daily capacity.
3. Seller-facing wording approved by the seller representative; appeal route live.
4. Override logging switched on and reviewed weekly.
5. DPO confirms the compliance report for the deployment country.

## 5. Consultation and review

Reviewed by the Responsible AI Review Board at gate 2 (outcome: not approved). To be repeated at every major model version, at any change of deployment country (for example a HESU pilot in Tanzania), and at least once a year.
