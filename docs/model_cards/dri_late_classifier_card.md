# Model Card: DRI Late-Delivery Classifier

Format after Mitchell et al. (2019). BAN6800 Module 4 · Gain Tawodzera · 13 November 2026

| | |
| --- | --- |
| **Model** | `dri-late-delivery-classifier`, MLflow Model Registry **version 3**, alias `champion` |
| **Algorithm** | Logistic regression (L2, C = 0.0042, class_weight = balanced) in a scikit-learn pipeline |
| **Files** | `models/dri_late_classifier.joblib` (pipeline), `models/dri_late_classifier.json` (metadata), `models/dri_fairness_postprocessor.joblib` (fairness thresholds) |
| **Serving** | FastAPI `POST /predict` (`api/main.py`) |
| **Status** | **Shadow mode only.** Fails the G1 recall target and the G5 fairness gate (below). Not approved to drive actions. |

## 1. Intended use

- **Purpose:** at the moment an Olist order is placed, estimate the probability that it will arrive after the promised date, so the delivery operations team can act early (remind the seller, update the date, message the customer). Business goals G1–G5, Module 1 Vision Document.
- **Decision rule:** each day, the 25% of that day's orders with the highest risk are flagged for review (about 50 orders a day at this dataset's volume). For single-order scoring, the API compares the score with the 75th percentile of the last 30 days' scores (0.5003), recalibrated monthly.
- **Users:** delivery operations (daily list), seller support, customer service. Explanations are shown with every score.
- **Out of scope:** automatic action without human review; judging or penalising sellers; use outside Brazil (for example, HESU in Tanzania) without retraining on local data.

## 2. Data

- **Source:** Olist public dataset (2016–2018), processed by the Module 3 pipeline (`data/processed/orders_model_ready.parquet`, Great Expectations gate passed: 22/22).
- **Unit:** one delivered order. **Target:** `is_late` = delivered after the promised date.
- **Features:** 32 inputs known at purchase (26 numeric, 6 categorical). This includes 5 market-context features added in Module 4: recent state and national late rates, recent state delivery time, promise slack and seller workload. All use only deliveries completed before the purchase.
- **Time-aware split:**

| Set | Purchases | Orders | Late rate |
| --- | --- | --- | --- |
| Train | Jan 2017 – Feb 2018 | 57,050 | 7.7% |
| Validation | Mar – May 2018 | 20,550 | 11.7% |
| Test (held out) | Jun – Aug 2018 | 18,603 | 5.5% |

## 3. Performance (test period, top 25% of each day flagged)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Majority class (baseline) | 0.945 | 0.000 | 0.000 | 0.000 | 0.500 | 0.055 |
| Seller late rate (heuristic) | 0.721 | 0.059 | 0.273 | 0.097 | 0.497 | 0.055 |
| **Logistic regression (champion)** | 0.735 | 0.082 | **0.376** | 0.134 | 0.713 | 0.108 |
| Random Forest | 0.738 | 0.088 | 0.406 | 0.145 | 0.710 | 0.095 |
| XGBoost (challenger) | 0.745 | 0.103 | 0.472 | 0.169 | 0.772 | 0.133 |
| Champion + fairness thresholds | – | 0.123 | 0.511 | – | – | – |

The champion was chosen on validation recall and confirmed by a rolling-origin backtest over three windows before the test period (mean recall 0.476, vs 0.458 for XGBoost and 0.449 for Random Forest). XGBoost did better on the test period. That was seen after the choice, so it is recorded as a challenger, not adopted (see the report, section 4).

**Against the success criteria:**

| Goal | Target | Result | Met? |
| --- | --- | --- | --- |
| G1: late orders flagged in advance | ≥ 70% recall within capacity | 37.6% (51.1% with fairness thresholds) | No |
| Beats baselines | PR-AUC above heuristic | 0.108 vs 0.055 (2× base rate) | Yes |
| G5: recall gap between regions | ≤ 10 points | 66.8 points (24.7 after mitigation) | No |

## 4. Fairness

Protected attributes: the data holds no age, gender or race. The groups at risk named in the Module 1 charter are **customer region** (a proxy for income and remoteness, ER1/ER2) and **seller size** (ER3). Full detail: `docs/fairness_report.md`.

- The model **over-flags the Northeast** (flags 78% of its orders; recall 97%; false-positive rate 77%) and **under-serves the Southeast** (flags 18%; recall 30%). The Southeast has 812 of the 1,019 late test orders.
- Removing state and region columns does not fix this (gap 62 points), because distance and recent state delivery time carry the same information.
- Fairlearn's ThresholdOptimizer (recall parity by region, fitted on validation) narrows the gap to 25 points and **raises overall recall to 51%** while flagging fewer orders (22.7%).

## 5. Explainability

- **Global (SHAP):** the top drivers are days promised, recent delivery time in the customer's state, promise slack, distance, peak season and the seller's past handling time.
- **Local:** every `/predict` response includes the top three reasons in plain words (exact linear SHAP).
- **Counterfactual:** for flagged orders, a median of 5 extra promise days brings the risk below the threshold (70% resolved within 7 days; 7 days needed in the North and Northeast).

## 6. Robustness (sensitivity analysis)

- ±2 promised days changes 4–5% of decisions; ±10% in distance, freight, price or weight changes under 1%.
- 5% Gaussian noise on all numeric inputs lowers ROC-AUC from 0.713 to 0.690.
- Removing seller history does not lower ROC-AUC (0.719), so that history adds little.
- ROC-AUC falls each month after training: June 0.761, July 0.692, August 0.654. The model needs regular retraining.

## 7. Limitations and risks (Module 1 risk register)

- Recall is far below the G1 target: purchase-time facts explain only part of lateness.
- Concept drift (R03) is visible within three months.
- Data from 2016–2018 may not reflect current conditions.
- The North has only 9 late test orders, too few to judge its fairness.
- Precision is low (8–12%): most flagged orders arrive on time, so actions must be cheap and non-punitive.
- The regional pattern risks padding promises in the Northeast (ER5).

## 8. Monitoring and acceptance criteria for release

| Signal | Frequency | Alert |
| --- | --- | --- |
| Recall within capacity (once labels arrive) | Weekly | Below 45% |
| Recall gap between regions (≥ 30 late orders) | Monthly | Above 10 points |
| ROC-AUC on the last 30 days | Monthly | Below 0.70 → retrain |
| Score drift (PSI) and threshold recalibration | Monthly | PSI above 0.2 |
| Flag rate per day | Daily | Outside 20–30% |
| Override rate by staff (ER6) | Weekly | Above 50% |

**Release beyond shadow mode requires:** recall ≥ 55% within capacity, a regional recall gap ≤ 10 points with every judged region holding at least 30 late orders, and Responsible AI Review Board sign-off (gate 2).
