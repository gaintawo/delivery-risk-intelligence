# Dashboard data

Built by `python -m src.models.export_dashboard` from the Module 3 pipeline output and the Module 4 champion model.

| File | Contents |
| --- | --- |
| `test_scored.csv.gz` | 18,603 delivered orders from the held-out test period (June–August 2018): the 32 model inputs, outcome, risk score, standard and fairness-adjusted flags, three marked example orders |
| `test_shap.csv.gz` | Exact linear SHAP values per order and input (same row order) |
| other files | Copies of Module 4 evidence (global SHAP, sensitivity, backtest, fairness summaries) |

**Privacy (Module 2 privacy plan, ER4):** no order, customer or seller keys, names, postcodes or review text. Location appears only as state and region.
**Licence:** derived from the Olist Brazilian e-commerce public dataset, CC BY-NC-SA 4.0. Non-commercial use, credit Olist, share alike.
