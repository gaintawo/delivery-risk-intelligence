"""DRI ETL pipeline - Prefect flow (the orchestration DAG).

    sources -> ingest -> clean -> integrate -> transform -> anonymise
            -> validate (hard gate) -> bias check -> load -> lineage

Run locally:   python -m src.pipeline.flow
Run in Docker: docker run --rm -v "$PWD/data:/app/data" dri-pipeline
Each task retries once on failure; every task writes to the privacy audit log.
"""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from src.fairness.bias import run_bias_checks
from src.features.build import build_features
from src.pipeline import anonymize, clean, config, ingest, integrate, load, validate
from src.pipeline.audit import RUN_ID, audit

TASK = {"retries": 1, "retry_delay_seconds": 5, "cache_policy": NO_CACHE}
LINEAGE: list[dict] = []


def _stage(name: str, inputs: list[str], output: str, n_rows: int, n_cols: int, t0: float, **notes):
    LINEAGE.append({"stage": name, "inputs": inputs, "output": output, "rows_out": n_rows,
                    "cols_out": n_cols, "seconds": round(time.time() - t0, 2), **notes})


@task(name="1 Acquire sources", **TASK)
def acquire() -> list[str]:
    t0 = time.time()
    fetched = ingest.download_missing()
    _stage("acquire", [config.SOURCE_URL], "data/raw/*.csv", len(ingest.TABLES), 0, t0,
           downloaded=fetched)
    return fetched


@task(name="2 Ingest", **TASK)
def ingest_task() -> dict[str, pd.DataFrame]:
    t0 = time.time()
    raw = ingest.load_all()
    _stage("ingest", [s["file"] for s in ingest.TABLES.values()], "9 typed DataFrames",
           sum(len(d) for d in raw.values()), 0, t0,
           tables={k: len(v) for k, v in raw.items()})
    return raw


@task(name="3 Clean", **TASK)
def clean_task(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    t0 = time.time()
    cleaned, reports = clean.clean_all(raw)
    for name, rep in reports.items():
        audit("transform", name, purpose="Clean table", rows=rep.get("rows_out"), report=rep)
    _stage("clean", list(raw), "8 clean tables", sum(len(d) for d in cleaned.values()), 0, t0,
           reports=reports)
    return cleaned


@task(name="4 Integrate", **TASK)
def integrate_task(cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    t0 = time.time()
    df, rep = integrate.build_order_table(cleaned)
    audit("transform", "orders_integrated", purpose="Join tables to one row per order",
          rows=len(df), cols=df.shape[1], report=rep)
    _stage("integrate", list(cleaned), "orders_integrated", len(df), df.shape[1], t0, report=rep)
    return df


@task(name="5 Transform (features)", **TASK)
def transform_task(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    df = build_features(df)
    audit("transform", "orders_features", purpose="Build labels and leakage-safe features",
          rows=len(df), cols=df.shape[1])
    _stage("transform", ["orders_integrated"], "orders_features", len(df), df.shape[1], t0,
           late_rate=round(float(df["is_late"].mean()), 4), delivered=int(df["is_delivered"].sum()))
    return df


@task(name="6 Anonymise", **TASK)
def anonymise_task(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    out, rep = anonymize.anonymise(df)
    k = anonymize.k_anonymity(out, ["customer_state", "purchase_month"])
    _stage("anonymise", ["orders_features"], "orders_model_ready", len(out), out.shape[1], t0,
           columns_removed=rep["columns_removed"], k_anonymity_state_month=k)
    return out


@task(name="7 Validate (Great Expectations)", **TASK)
def validate_task(raw: dict[str, pd.DataFrame], df: pd.DataFrame) -> dict:
    t0 = time.time()
    summary = validate.run_validation(raw["orders"], raw["items"], df)
    _stage("validate", ["raw_orders", "raw_order_items", "orders_model_ready"], "gx data docs",
           len(df), df.shape[1], t0, results=summary)
    if not summary["orders_model_ready"]["success"]:
        raise ValueError(f"Model-ready data failed validation: {summary['orders_model_ready']['failed']}")
    return summary


@task(name="8 Bias check (Fairlearn)", **TASK)
def bias_task(df: pd.DataFrame) -> dict:
    t0 = time.time()
    res = run_bias_checks(df)
    _stage("bias_check", ["orders_model_ready"], "docs/evidence/bias_*.csv", len(df), df.shape[1],
           t0, flags=res["flags"])
    return res["flags"]


@task(name="9 Load outputs", **TASK)
def load_task(df: pd.DataFrame) -> dict:
    t0 = time.time()
    res = load.write_outputs(df)
    _stage("load", ["orders_model_ready"],
           "parquet + warehouse table/view + dashboard_feed.csv", len(df), df.shape[1], t0, **res)
    return res


@flow(name="dri-etl", log_prints=True)
def dri_etl() -> dict:
    log = get_run_logger()
    config.ensure_dirs()
    LINEAGE.clear()
    audit("run_start", "pipeline", purpose="Start DRI ETL run")
    acquire()
    raw = ingest_task()
    cleaned = clean_task(raw)
    integrated = integrate_task(cleaned)
    features = transform_task(integrated)
    ready = anonymise_task(features)
    checks = validate_task(raw, ready)
    flags = bias_task(ready)
    outputs = load_task(ready)

    lineage = {"run_id": RUN_ID, "finished": datetime.now(UTC).isoformat(timespec="seconds"),
               "stages": LINEAGE}
    (config.EVIDENCE_DIR / "lineage.json").write_text(json.dumps(lineage, indent=2, default=str))
    audit("run_end", "pipeline", purpose="Finish DRI ETL run", rows=outputs["rows"])
    log.info("Validation: %s", {k: v for k, v in checks.items() if k != "data_docs"})
    log.info("Bias flags: %s", flags)
    log.info("Outputs: %s", outputs)
    return {"run_id": RUN_ID, "validation": checks, "bias": flags, "outputs": outputs}


if __name__ == "__main__":
    dri_etl()
