# Changelog

All notable changes to Delivery Risk Intelligence (DRI) are documented in this file.
Versions follow semantic versioning (Module 1, Section 6).

## [0.2.0] - November 2026 (Module 2, Sprint S0)

### Added
- Full Module 1 directory structure: `notebooks/`, `src/` (pipeline, features, models, fairness, actions), `api/`, `dashboard/`, `tests/`.
- `.gitignore` (keeps data and secrets out of Git) and `.env.example`.
- GitHub Actions CI running `ruff` and `pytest` on every push and pull request.
- Sprint backlog of 32 tickets (`docs/backlog.csv`) and board setup script (`.github/setup_board.py`).
- Technical RAID log as a spreadsheet (`docs/raid_log.xlsx`) alongside `docs/RAID_LOG.md`.
- Sprint S0 review and retrospective (`docs/sprint_notes/`).

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
