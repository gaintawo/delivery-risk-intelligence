# Delivery Risk Intelligence (DRI)

A predictive analytics project for identifying and preventing late e-commerce deliveries using the Olist dataset.

BAN6800 Business Analytics Capstone · Nexford University · Gain Tawodzera

| | |
| --- | --- |
| Vertical | Logistics, last-mile delivery |
| Dataset | [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0) |
| Status | Sprint 4 (Module 4): model trained, explained and fairness-tested; **shadow mode only** |

## Repository structure (from the Module 1 Vision Document)

```
delivery-risk-intelligence/
├── README.md            What the project is and how to run it
├── CHANGELOG.md         Changes in each version
├── docs/                Vision, data dictionary, RAID log, architecture, backlog
│   ├── model_cards/     One summary sheet per model version
│   └── sprint_notes/    Sprint reviews and retrospectives
├── data/                Tracked with DVC, not Git
│   ├── raw/             Original Olist files, never edited
│   └── processed/       Cleaned data ready for modelling
├── notebooks/           Step-by-step analysis (01_explore.ipynb, ...)
├── src/
│   ├── pipeline/        Loading and cleaning data
│   ├── features/        Building model inputs
│   ├── models/          Training and testing
│   ├── fairness/        Bias checks
│   └── actions/         Recommendation logic
├── api/                 Model serving (FastAPI)
├── dashboard/           Streamlit dashboard
├── tests/               Automatic checks
└── requirements.txt     Software the project needs
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your own values; .env is never committed
pytest -q
```

## Run the data pipeline (Module 3)

```bash
pip install -r requirements-pipeline.txt
python -m src.pipeline.flow     # downloads the Olist data if missing, then runs all 9 stages
pytest -q                       # 21 unit tests
```

Or in Docker: `docker build -t dri-pipeline . && docker run --rm -v "$PWD/data:/app/data" dri-pipeline`.
GitHub Actions builds the container and runs the full pipeline on every push.

| Document | What it covers |
| --- | --- |
| [Pipeline](docs/pipeline.md) | Stages, data-quality decisions, outputs, lineage |
| [Data anonymization plan](docs/data_anonymization_plan.md) | PII inventory and how each field is removed or pseudonymised |
| [Data governance framework](docs/data_governance.md) | Access control, retention, audit logging, change control |
| [Evidence](docs/evidence/) | Prefect run graph, Great Expectations results, bias report, audit log sample |

## Model and API (Module 4)

```bash
pip install -r requirements-pipeline.txt -r requirements-model.txt
python -m src.models.train      # tune 3 algorithms + baselines, log to MLflow, register champion
python -m src.models.backtest   # rolling-origin check of the champion choice
python -m src.models.analyze    # SHAP, fairness (Fairlearn), counterfactuals (DiCE), sensitivity
uvicorn api.main:app            # POST /predict, docs at http://127.0.0.1:8000/docs
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

| Document | What it covers |
| --- | --- |
| [Model card](docs/model_cards/dri_late_classifier_card.md) | Intended use, performance, fairness, limits, monitoring |
| [Fairness metrics report](docs/fairness_report.md) | Demographic parity, equalized odds, disparate impact, mitigation |
| [SHAP notebook](notebooks/02_shap_analysis.ipynb) | Global, local and counterfactual explanations |
| [Model evidence](docs/evidence/model/) | Figures, metrics, backtest, MLflow run export and screenshots |

## Planning documents (Module 2)

- [Technical architecture](docs/technical_architecture.md)
- [Data dictionary](docs/data_dictionary.md)
- [RAID log](docs/RAID_LOG.md) ([spreadsheet](docs/raid_log.xlsx))
- [Sprint backlog](docs/backlog.csv): 32 detailed tickets, each linked to one of the 11 issues on the GitHub Project board

## Branching

Git Flow: `main` (released), `develop` (integration), `feature/DRI-xx` (one branch per ticket).
Pull requests merge into `develop` only when CI passes.

## Licence and credit

Data © Olist, used under CC BY-NC-SA 4.0 for non-commercial coursework.
