# Processed Data

Written by the pipeline; not committed to Git (restricted, see `docs/data_governance.md`).

- `orders_model_ready.parquet` - one pseudonymised row per order, for Module 4 modelling
- `dri_warehouse.db` - default SQLite warehouse (`orders_model_ready` table, `v_training_delivered` view); set `DATABASE_URL` to use PostgreSQL
- `dashboard_feed.csv` - region x month x category aggregates with small cells suppressed
