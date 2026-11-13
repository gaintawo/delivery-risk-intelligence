# Changelog

All notable changes to Delivery Risk Intelligence (DRI) are documented in this file.
Versions follow semantic versioning (Module 1, Section 6).

## [0.5.0] - November 2026 (Module 5, Sprint 5)

### Added
- Streamlit stakeholder dashboard (`dashboard/app.py`, `dashboard/views.py`): overview with goal scorecard, global drivers, order explorer with live what-if, ethical compliance page using Fairlearn MetricFrame, transparency statement. Deployable to Streamlit Community Cloud (`dashboard/requirements.txt`, `.streamlit/config.toml`).
- De-identified, scored test-period export for the dashboard (`src/models/export_dashboard.py`, `dashboard/data/`).
- API demo script (`api/demo.py`) and its output; dashboard smoke tests (`tests/test_dashboard.py`).

### Changed
- `src/models/explain.py` imports SHAP only for tree models, so the dashboard and API need no SHAP install.

## [0.4.0] - November 2026 (Module 4, Sprint 4)

### Added
- Model training with MLflow tracking: majority-class and heuristic baselines, logistic regression, Random Forest and XGBoost, tuned by randomized search with time-series cross-validation (`src/models/train.py`). 116 runs logged.
- Rolling-origin backtest for champion selection (`src/models/backtest.py`).
- Champion registered as `dri-late-delivery-classifier` v3 (alias `champion`); serialized with joblib in `models/`.
- SHAP explanations (`src/models/explain.py`, `notebooks/02_shap_analysis.ipynb`), DiCE counterfactuals, sensitivity analysis and Fairlearn fairness metrics with ThresholdOptimizer mitigation (`src/models/analyze.py`, `src/fairness/model_fairness.py`).
- FastAPI service with `/predict` and `/health` (`api/main.py`) and 7 API tests.
- Model card, fairness metrics report, delivery-days regressor.
- Five market-context features known at purchase (`src/features/build.py`).

### Changed
- Decision rule: top 25% of each day's orders, replacing a fixed probability threshold that flagged only 7% of test orders (model v2).

## [0.3.0] - November 2026 (Module 3, Sprints 2–3)

### Added
- Data pipeline as a Prefect flow (`src/pipeline/flow.py`) with 9 stages: acquire, ingest, clean, integrate, transform, anonymise, validate, bias check, load.
- Great Expectations: 3 suites, 35 expectations; the model-ready suite is a hard gate (`src/pipeline/validate.py`, `gx/`).
- Representation-bias checks with Fairlearn (`src/fairness/bias.py`).
- Pseudonymisation and PII removal (`src/pipeline/anonymize.py`); privacy audit log (`src/pipeline/audit.py`).
- Leakage-safe features with point-in-time seller history (`src/features/build.py`).
- 21 unit tests using Faker-generated data; `Dockerfile`, `requirements-pipeline.txt` with exact versions.
- CI now builds the container and runs the full pipeline on the Olist data.
- `docs/pipeline.md`, `docs/data_anonymization_plan.md`, `docs/data_governance.md`, `docs/evidence/`.

### Changed
- Data dictionary: added the model-ready output schema (section 9).

## [0.2.0] - November 2026 (Module 2, Sprint 1 – Project Setup)

### Added
- Full Module 1 directory structure: `notebooks/`, `src/` (pipeline, features, models, fairness, actions), `api/`, `dashboard/`, `tests/`.
- `.gitignore` (keeps data and secrets out of Git) and `.env.example`.
- GitHub Actions CI running `ruff` and `pytest` on every push and pull request.
- Sprint backlog of 32 tickets (`docs/backlog.csv`), each linked to one of the 11 Project board issues.
- Technical RAID log as a spreadsheet (`docs/raid_log.xlsx`) alongside `docs/RAID_LOG.md`.
- Sprint 1 review and retrospective (`docs/sprint_notes/`).

### Changed
- Data dictionary extended to all nine Olist tables, with an availability tag per column to prevent leakage.
- RAID log expanded to 24 entries with owners, sprints and likelihood/impact scores.
- Architecture document now maps each component to its repository folder and includes MLflow.
- `requirements.txt` given minimum versions and the missing storage, versioning and testing packages.

### Fixed
- Renamed `data/proceed/` to `data/processed/` and `docs/model_cards/READM.md` to `README.md`.

## [0.1.0] - November 2026 (Module 1)

### Added
- Initial GitHub repository.
- Project vision established in Module 1.
- Initial repository structure planned.
- Olist dataset selected for delivery risk analysis.
