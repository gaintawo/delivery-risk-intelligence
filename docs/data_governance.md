# Data Governance Framework

Delivery Risk Intelligence (DRI) · Module 3 · Puts into practice the Module 1 Ethical AI Charter and the Module 2 Data Privacy Plan and RACI.

## 1. Data classification

| Class | Datasets | Who may access |
| --- | --- | --- |
| **Restricted – raw** | `data/raw/*` (original Olist files) | Data engineer only; read-only; never committed to Git |
| **Restricted – model-ready** | `data/processed/orders_model_ready.parquet`, warehouse table `orders_model_ready`, view `v_training_delivered` | Data engineer, analysts (Module 4 modelling), Responsible AI Review Board for audits |
| **Internal – aggregated** | `data/processed/dashboard_feed.csv` (small cells suppressed) | Delivery operations, seller support, Head of Logistics |
| **Internal – evidence** | `docs/evidence/*`, `logs/audit.log` | Project team, Data Protection Officer, Review Board |

## 2. Access control matrix

| Role (Module 2 RACI) | Raw | Model-ready | Dashboard feed | Audit log | Change pipeline code |
| --- | --- | --- | --- | --- | --- |
| Data engineer / analyst (me) | Read | Read / write | Read / write | Read | Yes, via pull request into `develop` |
| Data Protection Officer | — | Read (audit only) | Read | Read | Approves privacy changes |
| Responsible AI Review Board | — | Read (audit only) | Read | Read | Gate sign-off (Sprints 3 and 4) |
| Delivery operations, seller support | — | — | Read | — | — |
| Head of Logistics (sponsor) | — | — | Read | Summary only | — |

**How it is enforced in this project:**
- Raw and processed data are git-ignored, so they never reach the public repository.
- Secrets (`DRI_HASH_SALT`, `DATABASE_URL`) live in `.env`, which is also git-ignored, and in GitHub Actions secrets.
- In production, PostgreSQL roles mirror the matrix: `dri_engineer` (all), `dri_analyst` (model-ready, read), `dri_dashboard` (feed view only).
- The container runs as a non-root user.

## 3. Retention

| Data | Kept for | Then |
| --- | --- | --- |
| Raw source files | Life of the project (public data, licence CC BY-NC-SA 4.0) | Deleted at project close |
| Model-ready table and warehouse | 24 months from the last order date (Module 1 governance principle) | Rebuilt or deleted |
| Dashboard feed | 24 months | Deleted |
| Privacy audit log | 24 months, append-only | Archived, then deleted |
| Great Expectations results and Data Docs | Last 12 runs | Older runs deleted |

## 4. Privacy audit logging

Every pipeline step calls `src/pipeline/audit.py`, which appends one JSON line to `logs/audit.log` with:
`ts` (UTC), `run_id`, `actor`, `action` (download, read, transform, pii_pseudonymise, pii_remove, validate, bias_check, write), `dataset`, `rows`, `cols`, `purpose`, the file path and its **SHA-256 fingerprint**, plus details such as which columns were removed. A sample from a real run is in `docs/evidence/audit_log_sample.jsonl`.

## 5. Change control and quality gates

1. Every change goes through a pull request into `develop`; CI must pass (lint, 21 unit tests, a container build and a full pipeline run).
2. The Great Expectations model-ready suite is a **hard gate**: if any of its 22 checks fails, the flow stops before anything is loaded.
3. The bias report is regenerated on every run and reviewed at the Review Board gates (end of Sprints 3 and 4).

## 6. Regulatory basis

- LGPD (Brasil, 2018): purpose limitation, data minimisation and security, applied to the Brazilian source data.
- Tanzania Personal Data Protection Act, 2022: the same controls are designed in for the future HESU phase.
- NIST AI RMF 1.0 (NIST, 2023): the bias checks and gates map to its *Measure* and *Manage* functions.
