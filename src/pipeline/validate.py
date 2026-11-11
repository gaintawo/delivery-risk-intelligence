"""Stage 7: data validation with Great Expectations.

Three suites run in one checkpoint:
  raw_orders_suite          - is the source orders file what we expect?
  raw_order_items_suite     - are prices, freight and keys sane?
  orders_model_ready_suite  - is the final table safe to train on, and free of PII?

Raw suites document source problems; the model-ready suite is a hard gate:
if it fails, the pipeline stops and nothing is loaded.
Results and HTML Data Docs are written under gx/.
"""
from __future__ import annotations

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd
from great_expectations.expectations.row_conditions import Column

from . import config
from .anonymize import OUTPUT_COLUMNS
from .audit import audit

REGIONS = list(config.POPULATION_SHARE_2022)
STATES = list(config.STATE_TO_REGION)
ORDER_STATUSES = ["delivered", "shipped", "canceled", "unavailable", "invoiced",
                  "processing", "created", "approved"]
PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card"]
DELIVERED = {"row_condition": Column("is_delivered") == 1}


def raw_orders_expectations() -> list:
    return [
        gxe.ExpectTableRowCountToBeBetween(min_value=5000),
        gxe.ExpectColumnValuesToBeUnique(column="order_id"),
        gxe.ExpectColumnValuesToNotBeNull(column="order_id"),
        gxe.ExpectColumnValuesToNotBeNull(column="order_purchase_timestamp"),
        gxe.ExpectColumnValuesToNotBeNull(column="order_estimated_delivery_date"),
        gxe.ExpectColumnValuesToBeInSet(column="order_status", value_set=ORDER_STATUSES),
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="order_approved_at", column_B="order_purchase_timestamp", or_equal=True,
            ignore_row_if="either_value_is_missing", mostly=0.99),
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="order_delivered_customer_date", column_B="order_delivered_carrier_date",
            or_equal=True, ignore_row_if="either_value_is_missing", mostly=0.99),
    ]


def raw_items_expectations() -> list:
    return [
        gxe.ExpectTableRowCountToBeBetween(min_value=5000),
        gxe.ExpectCompoundColumnsToBeUnique(column_list=["order_id", "order_item_id"]),
        gxe.ExpectColumnValuesToNotBeNull(column="seller_id"),
        gxe.ExpectColumnValuesToBeBetween(column="price", min_value=0.01),
        gxe.ExpectColumnValuesToBeBetween(column="freight_value", min_value=0),
    ]


def model_ready_expectations() -> list:
    return [
        # Shape and privacy gate: exactly the allow-listed columns, no identifiers.
        gxe.ExpectTableRowCountToBeBetween(min_value=5000),
        gxe.ExpectTableColumnsToMatchSet(column_set=OUTPUT_COLUMNS, exact_match=True),
        gxe.ExpectColumnValuesToMatchRegex(column="order_key", regex=r"^[0-9a-f]{16}$"),
        gxe.ExpectColumnValuesToBeUnique(column="order_key"),
        # Completeness
        gxe.ExpectColumnValuesToNotBeNull(column="order_purchase_timestamp"),
        gxe.ExpectColumnValuesToNotBeNull(column="customer_region"),
        gxe.ExpectColumnValuesToNotBeNull(column="promised_days"),
        gxe.ExpectColumnValuesToNotBeNull(column="is_late", **DELIVERED),
        gxe.ExpectColumnValuesToNotBeNull(column="distance_km", mostly=0.99),
        # Valid categories
        gxe.ExpectColumnValuesToBeInSet(column="is_late", value_set=[0, 1]),
        gxe.ExpectColumnValuesToBeInSet(column="customer_region", value_set=REGIONS),
        gxe.ExpectColumnValuesToBeInSet(column="customer_state", value_set=STATES),
        gxe.ExpectColumnValuesToBeInSet(column="payment_type", value_set=PAYMENT_TYPES),
        gxe.ExpectColumnValuesToBeInSet(column="seller_size", value_set=["new", "small", "large"]),
        # Plausible ranges
        gxe.ExpectColumnValuesToBeBetween(column="promised_days", min_value=0, max_value=200),
        gxe.ExpectColumnValuesToBeBetween(column="distance_km", min_value=0, max_value=4500),
        gxe.ExpectColumnValuesToBeBetween(column="price_total", min_value=0.01),
        gxe.ExpectColumnValuesToBeBetween(column="seller_hist_late_rate", min_value=0, max_value=1),
        gxe.ExpectColumnValuesToBeBetween(column="review_score", min_value=1, max_value=5),
        gxe.ExpectColumnValuesToBeBetween(column="handling_days", min_value=0, max_value=60, mostly=0.99),
        gxe.ExpectColumnValuesToBeBetween(column="approval_lag_h", min_value=0, max_value=720, mostly=0.99),
        # Drift guard: the late rate should stay near the 7-8% baseline.
        gxe.ExpectColumnMeanToBeBetween(column="is_late", min_value=0.04, max_value=0.15),
    ]


def _context():
    context = gx.get_context(mode="file", project_root_dir=str(config.ROOT))
    context.variables.progress_bars = {"globally": False}
    return context


def _validation(context, ds, name: str, expectations: list):
    try:
        asset = ds.get_asset(name)
    except LookupError:
        asset = ds.add_dataframe_asset(name=name)
    try:
        bd = asset.get_batch_definition("whole_table")
    except (LookupError, KeyError):
        bd = asset.add_batch_definition_whole_dataframe("whole_table")
    suite = context.suites.add_or_update(gx.ExpectationSuite(name=f"{name}_suite", expectations=expectations))
    return context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name=f"{name}_validation", data=bd, suite=suite))


def run_validation(raw_orders: pd.DataFrame, raw_items: pd.DataFrame,
                   model_ready: pd.DataFrame) -> dict:
    context = _context()
    ds = context.data_sources.add_or_update_pandas(name="dri_pandas")
    frames = {"raw_orders": (raw_orders, raw_orders_expectations()),
              "raw_order_items": (raw_items, raw_items_expectations()),
              "orders_model_ready": (model_ready, model_ready_expectations())}
    summary = {}
    for name, (df, exps) in frames.items():
        vd = _validation(context, ds, name, exps)
        cp = context.checkpoints.add_or_update(gx.Checkpoint(
            name=f"{name}_checkpoint", validation_definitions=[vd],
            actions=[gx.checkpoint.UpdateDataDocsAction(name="update_data_docs")]))
        result = cp.run(batch_parameters={"dataframe": df})
        vr = next(iter(result.run_results.values()))
        stats = vr.statistics
        summary[name] = {
            "success": bool(vr.success),
            "evaluated": int(stats["evaluated_expectations"]),
            "passed": int(stats["successful_expectations"]),
            "failed": [r.expectation_config.type for r in vr.results if not r.success],
        }
        audit("validate", name, purpose="Great Expectations data quality check", rows=len(df),
              **summary[name])
    summary["data_docs"] = "gx/uncommitted/data_docs/local_site/index.html"
    return summary
