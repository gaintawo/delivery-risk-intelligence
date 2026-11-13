# Delivery Risk Intelligence (DRI)
## Preliminary Data Dictionary

Version 0.2 · Module 2 (Sprint 1 – Project Setup) · Source: Olist (2018), *Brazilian E-Commerce Public Dataset by Olist*, Kaggle, CC BY-NC-SA 4.0.

## 1. Overview

This document describes the key fields from the Olist Brazilian E-Commerce Public Dataset that are relevant to the Delivery Risk Intelligence project.

The dataset contains historical Brazilian e-commerce transactions from 2016–2018. It is used for academic research and does not contain HESU Investment Limited's operational records.

## 2. Availability tags (enforces FR2: no leakage)

| Tag | Meaning | May be a model feature? |
| --- | --- | --- |
| **P** | Known at purchase (stage-1 scoring) | Yes, at both stages |
| **H** | Known by carrier hand-over (stage-2 scoring) | Only in the hand-over model |
| **O** | Outcome, known only after delivery | **Never.** Used for labels and evaluation only |
| **X** | Excluded from modelling (privacy or irrelevance) | No |

`tests/` will contain a check that fails the build if any **O** or **X** column reaches `src/features/`.

## 3. Source tables

### 3.1 `olist_orders_dataset.csv` (about 99,441 rows) · grain: one order · key: `order_id`

| Column | Type | Tag | Description |
| --- | --- | --- | --- |
| order_id | string (hash) | P | Unique order identifier |
| customer_id | string (hash) | P | Links to customers; one per order |
| order_status | category | O | delivered, shipped, canceled, etc. Final status leaks the outcome |
| order_purchase_timestamp | datetime | P | When the order was placed |
| order_approved_at | datetime | H | When payment was approved (lags for boleto) |
| order_delivered_carrier_date | datetime | H | When the seller handed the parcel to the carrier |
| order_delivered_customer_date | datetime | O | Actual delivery date |
| order_estimated_delivery_date | date | P | Date promised to the customer |

### 3.2 `olist_order_items_dataset.csv` (about 112,650 rows) · grain: one item in an order · key: `order_id` + `order_item_id`

| Column | Type | Tag | Description |
| --- | --- | --- | --- |
| order_id | string | P | Parent order |
| order_item_id | integer | P | Item sequence within the order |
| product_id | string | P | Links to products |
| seller_id | string | P | Links to sellers |
| shipping_limit_date | datetime | P | Deadline for the seller to hand the item to the carrier |
| price | decimal (BRL) | P | Item price |
| freight_value | decimal (BRL) | P | Delivery fee; proxy for route length and courier |

### 3.3 `olist_order_payments_dataset.csv` (about 103,886 rows) · grain: one payment · key: `order_id` + `payment_sequential`

| Column | Type | Tag | Description |
| --- | --- | --- | --- |
| order_id | string | P | Parent order |
| payment_sequential | integer | P | Payment number when several methods are used |
| payment_type | category | P | credit_card, boleto, voucher, debit_card |
| payment_installments | integer | P | Number of instalments |
| payment_value | decimal (BRL) | P | Amount paid |

### 3.4 `olist_order_reviews_dataset.csv` (about 99,224 rows) · grain: one review · key: `review_id`

| Column | Type | Tag | Description |
| --- | --- | --- | --- |
| review_id | string | O | Review identifier |
| order_id | string | O | Reviewed order |
| review_score | integer 1-5 | O | Star rating; used to measure G4 only |
| review_comment_title | text (Portuguese) | X | May contain personal names (ER4) |
| review_comment_message | text (Portuguese) | X | May contain personal names (ER4) |
| review_creation_date | datetime | O | When the review survey was sent |
| review_answer_timestamp | datetime | O | When the customer answered |

### 3.5 `olist_customers_dataset.csv` (about 99,441 rows) · key: `customer_id`

| Column | Type | Tag | Description |
| --- | --- | --- | --- |
| customer_id | string | P | Order-level customer key |
| customer_unique_id | string | P | Same person across orders; join key only, never displayed |
| customer_zip_code_prefix | string (5 digits) | P | Quasi-identifier; used only to derive distance and region |
| customer_city | string | X | Redundant with prefix; dropped (data minimisation) |
| customer_state | category (27 UF codes) | P | Mapped to Brazil's five regions for fairness checks |

### 3.6 `olist_sellers_dataset.csv` (about 3,095 rows) · key: `seller_id`

| Column | Type | Tag | Description |
| --- | --- | --- | --- |
| seller_id | string | P | Seller key |
| seller_zip_code_prefix | string | P | Used to derive distance |
| seller_city | string | X | Dropped (data minimisation) |
| seller_state | category | P | Interstate-route flag |

### 3.7 `olist_products_dataset.csv` (about 32,951 rows) · key: `product_id`

| Column | Type | Tag | Description |
| --- | --- | --- | --- |
| product_id | string | P | Product key |
| product_category_name | category (Portuguese) | P | Translated with table 9 |
| product_name_lenght | integer | X | Listing text length (column name misspelt in source); not relevant |
| product_description_lenght | integer | X | Not relevant |
| product_photos_qty | integer | X | Not relevant |
| product_weight_g | integer | P | Weight in grams |
| product_length_cm | integer | P | Package length |
| product_height_cm | integer | P | Package height |
| product_width_cm | integer | P | Package width |

### 3.8 `olist_geolocation_dataset.csv` (about 1,000,163 rows) · grain: one coordinate · several rows per prefix

| Column | Type | Tag | Description |
| --- | --- | --- | --- |
| geolocation_zip_code_prefix | string | P | Join key to customer and seller prefixes |
| geolocation_lat | float | P | Latitude; aggregated to one median point per prefix |
| geolocation_lng | float | P | Longitude; aggregated to one median point per prefix |
| geolocation_city | string | X | Dropped |
| geolocation_state | category | X | Dropped (state comes from customers/sellers) |

### 3.9 `product_category_name_translation.csv` (about 71 rows)

| Column | Type | Tag | Description |
| --- | --- | --- | --- |
| product_category_name | string | P | Portuguese category |
| product_category_name_english | string | P | English category |

## 4. Derived analytics fields

These fields will be created during data preparation in `src/features/`.

### 4.1 Labels and outputs (never model inputs)

| Field | Description | Tag |
| --- | --- | --- |
| actual_delivery_days | Days between purchase and actual delivery | O |
| delivery_delay_days | Actual delivery date minus estimated delivery date, in days | O |
| is_late | 1 if actual delivery occurs after the estimated date; otherwise 0 (classifier target) | O |
| days_late | delivery_delay_days floored at 0 (regressor target, G3) | O |
| delivery_risk_score | Predicted probability of late delivery, ranging from 0 to 1 | Output |
| predicted_days_late | Estimated number of days an order may be late | Output |

### 4.2 Model features

| Field | Formula | Tag | Purpose |
| --- | --- | --- | --- |
| promised_days | estimated_delivery_date - purchase_timestamp | P | Promise-gap parity (ER5) |
| distance_km | haversine(seller prefix point, customer prefix point) | P | Geography cause |
| interstate | seller_state != customer_state | P | Geography cause |
| region | customer_state mapped to N, NE, CW, SE, S | P | Fairness grouping (G5); not a model input by default |
| volume_cm3 | length x height x width | P | Product cause |
| n_items, n_sellers | counts per order | P | Multi-seller basket cause |
| seller_hist_late_rate | seller's late rate on orders delivered *before* this purchase | P | Seller cause; computed point-in-time to avoid leakage |
| seller_hist_handling_days | seller's median approval-to-carrier days on earlier orders | P | Seller cause |
| seller_size | small / large by prior order volume | P | Fairness grouping (ER3) |
| purchase_dow, purchase_month, is_peak_week | calendar parts of purchase timestamp | P | Calendar cause |
| is_boleto | payment_type == boleto | P | Payment cause |
| approval_lag_h | approved_at - purchase_timestamp | H | Payment cause, stage 2 only |
| handling_days | delivered_carrier_date - approved_at | H | Seller cause, stage 2 only |
| handover_vs_limit_days | delivered_carrier_date - shipping_limit_date | H | Missed hand-over deadline, stage 2 only |
| state_late_rate_30d | late share of deliveries to the customer's state completed in the 30 days before purchase (min. 20) | P | Market context (Module 4); captures late-delivery waves |
| state_delivery_days_30d | mean delivery days in the customer's state, same window | P | Market context (Module 4) |
| promise_slack_days | promised_days - state_delivery_days_30d | P | How generous the promise is vs recent reality (Module 4) |
| market_late_rate_14d | late share of all deliveries completed in Brazil in the 14 days before purchase (min. 20) | P | National disruption signal, e.g. May 2018 strike (Module 4) |
| seller_orders_7d | orders the seller received in the 7 days before purchase | P | Seller workload (Module 4) |

## 5. Row counts

Row counts are the published dataset sizes. They will be confirmed, and data types corrected, when the data is loaded in Sprint 2 – Data Understanding (Module 3).

## 6. Data Quality Considerations

The following checks are planned:

- Identify missing delivery dates.
- Identify duplicate records and validate join keys.
- Check for invalid chronological sequences.
- Review cancelled and unavailable orders separately.
- Validate derived delivery durations.
- Prevent future delivery information from entering prediction-time features.

## 7. Data Governance

The project will follow data minimisation and responsible analytics principles.

Customer identifiers will be used only where necessary for joining or validating records. Model evaluation will consider regional fairness and potential seller-related bias.

## 8. Implementation Status

This is a preliminary data dictionary based on the published Olist dataset schema. Field availability, data types, missing values and derived calculations will be verified during data exploration in Module 3.

## 9. Model-ready output (Modules 3–4)

`data/processed/orders_model_ready.parquet`: 99,441 rows (one per order), 46 columns, produced by `src/pipeline/flow.py`. Module 4 added the five market-context features.
Tags as in section 2; **Key** = salted pseudonym; **QA** = data-quality flag. *Missing* = share of rows without a value (run of 13 Nov 2026).

| Column | Type | Tag | Missing |
| --- | --- | --- | --- |
| `order_key` | string | Key | 0.0% |
| `customer_key` | string | Key | 0.0% |
| `seller_key` | string | Key | 0.8% |
| `order_purchase_timestamp` | datetime64[us] | P | 0.0% |
| `order_status` | string | O | 0.0% |
| `ts_anomaly` | int8 | QA | 0.0% |
| `is_late` | Int8 | O | 3.0% |
| `days_late` | float64 | O | 3.0% |
| `delivery_days` | float64 | O | 3.0% |
| `review_score` | float64 | O | 0.8% |
| `is_delivered` | int8 | O | 0.0% |
| `promised_days` | float64 | P | 0.0% |
| `purchase_dow` | int8 | P | 0.0% |
| `purchase_month` | int8 | P | 0.0% |
| `purchase_hour` | int8 | P | 0.0% |
| `is_peak_season` | int8 | P | 0.0% |
| `distance_km` | float64 | P | 0.8% |
| `interstate` | Int8 | P | 0.8% |
| `customer_region` | str | P | 0.0% |
| `customer_state` | string | P | 0.0% |
| `seller_state` | string | P | 0.8% |
| `n_items` | float64 | P | 0.8% |
| `n_sellers` | float64 | P | 0.8% |
| `price_total` | float64 | P | 0.8% |
| `freight_total` | float64 | P | 0.8% |
| `freight_ratio` | float64 | P | 0.8% |
| `weight_total_g` | float64 | P | 0.8% |
| `volume_total_cm3` | float64 | P | 0.8% |
| `any_size_imputed` | float64 | P | 0.8% |
| `primary_category` | string | P | 0.8% |
| `payment_type` | string | P | 0.0% |
| `is_boleto` | int8 | P | 0.0% |
| `payment_installments` | float64 | P | 0.0% |
| `n_payment_methods` | float64 | P | 0.0% |
| `seller_hist_orders` | float64 | P | 0.8% |
| `seller_hist_late_rate` | float64 | P | 6.4% |
| `seller_hist_handling_days` | float64 | P | 6.4% |
| `seller_size` | string | P | 0.8% |
| `state_late_rate_30d` | float64 | P | 3.0% |
| `state_delivery_days_30d` | float64 | P | 3.0% |
| `promise_slack_days` | float64 | P | 3.0% |
| `market_late_rate_14d` | float64 | P | 0.5% |
| `seller_orders_7d` | float64 | P | 0.8% |
| `approval_lag_h` | float64 | H | 0.2% |
| `handling_days` | float64 | H | 3.1% |
| `handover_vs_limit_days` | float64 | H | 1.8% |
