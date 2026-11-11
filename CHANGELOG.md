# Changelog

All notable changes to Delivery Risk Intelligence (DRI) are documented in this file.
Versions follow semantic versioning (Module 1, Section 6).

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
