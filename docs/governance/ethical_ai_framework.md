# Ethical AI Framework: Delivery Risk Intelligence

BAN6800 Final Project · Gain Tawodzera · November 2026

This framework brings together the ethics work of Modules 1–5. It sets out the principles, who is accountable for them, the controls that put them into practice, and the evidence that each control works. Supporting documents in this folder:

| Document | Purpose |
| --- | --- |
| [Algorithmic Impact Assessment](algorithmic_impact_assessment.md) | Who could be affected, how, and how much |
| [Regulatory Compliance Report](regulatory_compliance.md) | LGPD, GDPR, CCPA/CPRA, HIPAA, Tanzania and Kenya |
| [Public Trust Statement](public_trust_statement.md) | Plain-language promise to customers and sellers |
| [Monitoring, Incident Response and Decommissioning Plan](monitoring_and_incident_plan.md) | KPIs, drift and fairness monitoring, incidents, retirement |
| [Integration audit](../integration_audit.md) | Contradictions found across Modules 1–5 and how they were resolved |

## 1. Principles (Ethical AI Charter, Module 1, after NIST AI RMF 1.0)

| # | Principle | Commitment |
| --- | --- | --- |
| 1 | Fairness | No customer gets worse service because of where they live. |
| 2 | Transparency | Every prediction shows, in plain words, the top three reasons behind it. |
| 3 | Accountability | The tool only recommends. A named person makes each final decision. |
| 4 | Privacy | Only anonymous data is used. No one tries to identify real customers or sellers. |
| 5 | Support, not punishment | Seller scores help sellers improve; they are never used to penalise or remove a seller without human review. |

## 2. From principle to control to evidence

| Principle | Control (where it lives) | Evidence it works | Status |
| --- | --- | --- | --- |
| Fairness | Recall-parity gate of 10 points across regions with 30+ late orders (model card, release criteria); Fairlearn ThresholdOptimizer; weekly fairness monitoring (`src/monitoring/monitor.py`) | `docs/fairness_report.md`; monitoring site; dashboard "Ethical compliance" page | **Not met**: gap 67 points, 25 with mitigation; release blocked |
| Fairness (ER2 proxy) | Model retrained without location columns to test for proxies | Gap only fell 67 → 62: distance is a proxy; recorded | Known limitation |
| Transparency | Exact linear SHAP; top three reasons in every API response; dashboard drivers and explorer pages | `tests/test_api.py`; `notebooks/02_shap_analysis.ipynb` | In place |
| Transparency (audit) | Explanation store: every score saved with model version, input hash, reasons and actions (`api/store.py`, `GET /explanations/{id}`) | `tests/test_api.py::test_prediction_is_stored_for_audit` | In place |
| Accountability | Shadow mode; actions are suggestions; retraining registers a "candidate" but only the Review Board moves the "champion" alias (`src/models/retrain.py`) | MLflow registry aliases and tags | In place |
| Privacy | Salted SHA-256 keys, postcodes and coordinates deleted, allow-list of columns, k ≥ 5 suppression for shared aggregates, audit log of every data access (Module 3) | Great Expectations privacy gate (blocks on a leaked postcode); `docs/evidence/audit_log_sample.jsonl` | In place |
| Privacy (public extract) | Dashboard copy: no keys, state and region only, purchase week not date, no review scores | `dashboard/data/README.md` | In place |
| Support, not punishment | Action engine offers only support actions to sellers (`src/actions/recommend.py`); sellers can challenge a flag using its stored explanation | `tests/test_api.py::test_flagged_order_gets_support_actions_not_penalties` | Designed; to be checked in use |
| Honest dates (ER5) | No region-specific promise padding; what-if warns about it; expected delivery days reported for every order | Dashboard what-if note; API `expected_days_late` | Designed |

## 3. Governance structure (ongoing oversight)

```
Responsible AI Review Board (chair: Data Protection Officer)
  members: Head of Logistics · seller representative · customer representative · independent analyst
  ├── Gate 1: before building (data fair and lawful?)          passed, Sprint 3
  ├── Gate 2: before user testing (fair and explainable?)      NOT passed: G1 and G5 fail
  ├── Gate 3: quarterly after launch                           not reached
  └── can stop any release, order a rollback, or retire the model
Model owner (lead analyst)          day-to-day monitoring, retraining, incident triage
Data owner (data engineer)          pipeline, quality gates, privacy controls, retention
Operations lead                     how staff use flags; override log (ER6)
Data Protection Officer             LGPD compliance, data-subject requests, breach notification
```

| Activity | Responsible | Accountable | Consulted | Informed |
| --- | --- | --- | --- | --- |
| Weekly monitoring review | Model owner | Head of Logistics | Operations lead | Review Board |
| Promote a retrained model | Model owner | Review Board | DPO | Users |
| Fairness incident | Model owner | Review Board chair | Seller and customer reps | Head of Logistics |
| Privacy incident | Data owner | DPO | Review Board | ANPD (regulator), affected people |
| Retire the model | Model owner | Review Board | DPO, Operations lead | All users and sellers |

Meeting rhythm: weekly monitoring review (30 minutes); monthly retraining decision; quarterly Review Board; ad hoc within 2 working days of a Severity 1 or 2 incident.

## 4. Lifecycle controls

| Stage | Control | Owner |
| --- | --- | --- |
| Data | Licence check, minimisation, pseudonymisation, quality and privacy gates, bias report on every run | Data owner |
| Build | Leakage check (only facts known at purchase), time-aware validation, backtest, fairness tests, model card | Model owner |
| Release | Review Board gate, documented release criteria (recall ≥ 55%, regional gap ≤ 10 points) | Review Board |
| Operate | Monitoring, explanation store, override log, monthly retraining with gates | Model owner, Operations lead |
| Retire | Decommissioning plan (archive, notify, delete on schedule) | Review Board |

## 5. Current position

The model is **in shadow mode**. It meets the transparency, accountability and privacy controls. It does not meet the fairness gate (G5) or the detection target (G1), so gate 2 is closed. This framework is working as intended: it stopped a model that is unfair across regions from being used, and it made the reason visible.
