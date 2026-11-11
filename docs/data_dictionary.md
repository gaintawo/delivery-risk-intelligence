
# Delivery Risk Intelligence (DRI)
## Preliminary Data Dictionary

### 1. Overview

This document describes the key fields from the Olist Brazilian E-Commerce Public Dataset that are relevant to the Delivery Risk Intelligence project.

The dataset contains historical Brazilian e-commerce transactions from 2016–2018. It is used for academic research and does not contain HESU Investment Limited's operational records.

### 2. Orders Dataset

Source file: olist_orders_dataset.csv

| Field | Description |
|---|---|
| order_id | Unique identifier for each order |
| customer_id | Customer identifier linked to the order |
| order_status | Current or final order status |
| order_purchase_timestamp | Date and time the order was placed |
| order_approved_at | Date and time payment was approved |
| order_delivered_carrier_date | Date the order was handed to the carrier |
| order_delivered_customer_date | Actual customer delivery date |
| order_estimated_delivery_date | Estimated delivery date provided to the customer |

### 3. Customers Dataset

Source file: olist_customers_dataset.csv

| Field | Description |
|---|---|
| customer_id | Customer identifier used in orders |
| customer_unique_id | Identifier linking purchases by the same customer |
| customer_zip_code_prefix | Customer postal code prefix |
| customer_city | Customer city |
| customer_state | Customer state |

### 4. Order Items Dataset

Source file: olist_order_items_dataset.csv

| Field | Description |
|---|---|
| order_id | Identifier linking an item to an order |
| order_item_id | Item sequence number within an order |
| product_id | Product identifier |
| seller_id | Seller identifier |
| shipping_limit_date | Deadline for seller shipment |
| price | Product selling price |
| freight_value | Shipping charge |

### 5. Derived Analytics Fields

These fields will be created during data preparation.

| Field | Description |
|---|---|
| actual_delivery_days | Days between purchase and actual delivery |
| promised_delivery_days | Days between purchase and estimated delivery |
| delivery_delay_days | Actual delivery date minus estimated delivery date, in days |
| is_late | 1 if actual delivery occurs after the estimated date; otherwise 0 |
| delivery_risk_score | Predicted probability of late delivery, ranging from 0 to 1 |
| predicted_days_late | Estimated number of days an order may be late |

### 6. Data Quality Considerations

The following checks are planned:

- Identify missing delivery dates.
- Identify duplicate records and validate join keys.
- Check for invalid chronological sequences.
- Review cancelled and unavailable orders separately.
- Validate derived delivery durations.
- Prevent future delivery information from entering prediction-time features.

### 7. Data Governance

The project will follow data minimisation and responsible analytics principles.

Customer identifiers will be used only where necessary for joining or validating records. Model evaluation will consider regional fairness and potential seller-related bias.

### 8. Implementation Status

This is a preliminary data dictionary based on the published Olist dataset schema. Field availability, data types, missing values and derived calculations will be verified during data exploration in Module 3.
