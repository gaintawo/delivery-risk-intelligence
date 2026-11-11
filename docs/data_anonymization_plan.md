# Data Anonymization Plan

Delivery Risk Intelligence (DRI) · Module 3 (Sprint 3 – Data Preparation) · Builds on the Module 2 Data Privacy Plan and Module 1 ethical risk ER4.

The Olist data is already partly anonymised by Olist (hashed IDs, 5-digit postcode prefixes, company names replaced in reviews). DRI still treats every identifier and location field as personal data under Brazil's LGPD (Brasil, 2018). It also applies the same rules it would need for HESU data under Tanzania's Personal Data Protection Act, 2022.

## 1. PII inventory and action

| Field | Table | Type of risk | Action | Where in code |
| --- | --- | --- | --- | --- |
| `review_comment_title`, `review_comment_message` | reviews | Free text; may contain names | **Deleted at ingestion**, never processed | `clean.DROP_AT_INGEST` |
| `customer_city`, `seller_city`, `geolocation_city` | customers, sellers, geolocation | Quasi-identifier; redundant with state | **Deleted at ingestion** | `clean.DROP_AT_INGEST` |
| `order_id`, `seller_id` | orders, items | Linkable back to the public source | **Pseudonymised**: salted SHA-256, 16 hex characters | `anonymize.pseudonymise` |
| `customer_unique_id` | customers | Identifies one person across orders | **Pseudonymised**, kept only to stop one customer appearing in both training and test sets | `anonymize.pseudonymise` |
| `customer_id` | orders, customers | Per-order customer key | **Deleted** after joining | `anonymize.DELETE_AFTER_USE` |
| `customer_zip_code_prefix`, `seller_zip_code_prefix` | customers, sellers | Location quasi-identifier | Used to derive `distance_km`, then **deleted** | `anonymize.DELETE_AFTER_USE` |
| Latitude and longitude | geolocation | Precise location | Collapsed to one median point per prefix, used for distance, then **deleted** | `clean.clean_geolocation`, `anonymize` |
| Delivery, approval and carrier timestamps | orders | Behavioural detail | Turned into durations and labels, then **deleted**; only the purchase timestamp is kept (needed for time-based splits) | `anonymize.OUTPUT_COLUMNS` |

## 2. Techniques

1. **Data minimisation.** The output is an *allow-list* (`anonymize.OUTPUT_COLUMNS`). Any column not on it is dropped, so a new source column can never leak through by accident.
2. **Pseudonymisation.** Keys are `sha256(salt | value)` cut to 16 characters. The salt comes from the `DRI_HASH_SALT` environment variable and is never committed. Without the salt the keys cannot be reversed or re-linked to the public dataset.
3. **Generalisation.** Location is kept only as state and region.
4. **Small-cell suppression.** The shareable dashboard feed aggregates orders by region × month × category and blanks every cell with fewer than 5 orders (`load.dashboard_feed`).
5. **Synthetic test data.** Unit tests use records generated with Faker (`tests/synthetic.py`), so no test touches a real person's data.

## 3. Verification (automated)

| Check | Tool | Result on the real data (run of 11 Nov 2026) |
| --- | --- | --- |
| Output columns exactly match the allow-list | Great Expectations `expect_table_columns_to_match_set` | Pass |
| Keys are 16-character hex strings, not raw IDs | Great Expectations `expect_column_values_to_match_regex` | Pass |
| A leaked postcode column fails the gate | pytest `test_validation_gate_rejects_a_leaked_pii_column` | Pass |
| k-anonymity on state × purchase month | `anonymize.k_anonymity` | k = 2; 45 orders sit in groups smaller than 5 |
| Re-identification risk on state × month × category | `anonymize.k_anonymity` | k = 1, so the row-level table is **restricted** (see the governance framework); only the suppressed dashboard feed may be shared more widely |

## 4. Residual risk

The row-level table still holds state, purchase time and product category. Together, these can single out a small number of orders. This is accepted for model training because access is limited to the analytics role and every read is audit-logged. It is not acceptable for wider sharing, which is why the dashboard feed exists.
