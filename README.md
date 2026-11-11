# Delivery Risk Intelligence (DRI)

A predictive analytics project for identifying and preventing late e-commerce deliveries using the Olist dataset.

BAN6800 Business Analytics Capstone · Nexford University · Gain Tawodzera

| | |
| --- | --- |
| Vertical | Logistics, last-mile delivery |
| Dataset | [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0) |
| Status | Sprint 1 – Project Setup complete (Module 2) |

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
