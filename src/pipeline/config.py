"""Central settings for the DRI data pipeline.

Paths and secrets come from environment variables (see .env.example) so the
same code runs on a laptop, in Docker and in CI without edits.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(os.getenv("DRI_ROOT", Path(__file__).resolve().parents[2]))
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
LOG_DIR = ROOT / "logs"
EVIDENCE_DIR = ROOT / "docs" / "evidence"
GX_DIR = ROOT / "gx"

# Warehouse target. Module 1 names PostgreSQL; a local SQLite file is the
# zero-setup default so the pipeline runs anywhere.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROCESSED_DIR / 'dri_warehouse.db'}")

# Salt for pseudonymising identifiers. Must be set outside Git in real use.
HASH_SALT = os.getenv("DRI_HASH_SALT", "dri-dev-salt-change-me")

# Who is running the pipeline, for the privacy audit log.
ACTOR = os.getenv("DRI_ACTOR") or os.getenv("USER") or "unknown"

# Source of the public Olist files (Olist's own GitHub copy of the Kaggle data).
SOURCE_URL = os.getenv(
    "OLIST_SOURCE_URL",
    "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets",
)

# Brazil's 27 federative units mapped to the five IBGE regions.
STATE_TO_REGION = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North", "RO": "North", "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast", "PB": "Northeast",
    "PE": "Northeast", "PI": "Northeast", "RN": "Northeast", "SE": "Northeast",
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West", "MS": "Central-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South",
}

# Share of Brazil's population by region, 2022 Census (IBGE, 2023).
# Used as the reference for the representation-bias check.
POPULATION_SHARE_2022 = {
    "Southeast": 0.418,
    "Northeast": 0.269,
    "South": 0.147,
    "North": 0.085,
    "Central-West": 0.080,
}

# A group is flagged as under-represented when its share of orders is below
# this fraction of its share of the population (the "four-fifths" rule).
REPRESENTATION_THRESHOLD = 0.8

# Brazil's bounding box, used to drop impossible coordinates.
BRAZIL_LAT = (-34.0, 5.5)
BRAZIL_LNG = (-74.5, -34.0)


def ensure_dirs() -> None:
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, LOG_DIR, EVIDENCE_DIR):
        d.mkdir(parents=True, exist_ok=True)
