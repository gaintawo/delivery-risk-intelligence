# Models

| File | What it is |
| --- | --- |
| `dri_late_classifier.joblib` | Champion pipeline (preprocessing + logistic regression). MLflow registry: `dri-late-delivery-classifier` **version 3**, alias `champion` |
| `dri_late_classifier.json` | Metadata: features, decision rule, serving threshold, test metrics, training means for SHAP |
| `dri_fairness_postprocessor.joblib` | Fairlearn ThresholdOptimizer: region-specific thresholds (recall parity), used for `flagged_fairness_adjusted` |
| `dri_delivery_days_regressor.joblib` | XGBoost regressor for delivery days (goal G3, promise gap) |

Rebuild: `python -m src.pipeline.flow && python -m src.models.train && python -m src.models.backtest && python -m src.models.analyze`.
Status: **shadow mode only**. See `docs/model_cards/dri_late_classifier_card.md`.
