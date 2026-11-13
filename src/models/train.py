"""Train, tune, compare and register the late-delivery models (Module 4).

    python -m src.models.train

Every search candidate, every model and every artifact is logged to MLflow
(tracking store: mlflow.db, artifacts: mlruns/). The best model by validation
PR-AUC is registered as "dri-late-delivery-classifier" with alias "champion",
and saved to models/ with joblib for the FastAPI service.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.models import evaluate as ev
from src.models.data import (
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    REGRESSION_TARGET,
    SPLITS,
    TARGET,
    TIME_COL,
    check_no_leakage,
    load_modelling_frame,
    split,
)
from src.pipeline import config

SEED = 42
EXPERIMENT = "dri-late-delivery"
MODEL_NAME = "dri-late-delivery-classifier"
MODEL_DIR = config.ROOT / "models"
FIG_DIR = config.ROOT / "docs" / "evidence" / "model"
SKEWED = ["distance_km", "price_total", "freight_total", "freight_ratio", "weight_total_g",
          "volume_total_cm3", "seller_hist_orders", "seller_hist_handling_days", "seller_orders_7d"]


def tracking_setup() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(config.ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def preprocessor(numeric=NUMERIC, categorical=CATEGORICAL) -> ColumnTransformer:
    """Encoding and scaling, fitted on training data only (inside the pipeline)."""
    skewed = [c for c in numeric if c in SKEWED]
    plain = [c for c in numeric if c not in SKEWED]
    log_num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scale", StandardScaler())])
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200, sparse_output=False)
    return ColumnTransformer([("log", log_num, skewed), ("num", num, plain), ("cat", cat, categorical)],
                             verbose_feature_names_out=False)


def candidates(pos_weight: float) -> dict:
    return {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs"),
            {"model__C": loguniform(1e-3, 10)}, 8),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
            {"model__n_estimators": randint(150, 351), "model__max_depth": randint(6, 17),
             "model__min_samples_leaf": randint(5, 60), "model__max_features": uniform(0.15, 0.45)}, 8),
        "XGBoost": (
            XGBClassifier(objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
                          scale_pos_weight=pos_weight, n_jobs=2, random_state=SEED),
            {"model__n_estimators": randint(200, 601), "model__max_depth": randint(3, 8),
             "model__learning_rate": loguniform(0.02, 0.2), "model__subsample": uniform(0.6, 0.4),
             "model__colsample_bytree": uniform(0.5, 0.5), "model__min_child_weight": randint(1, 20),
             "model__reg_lambda": loguniform(0.5, 20)}, 12),
    }


def heuristic_scores(train: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """Naive baseline: a seller's past late rate (new sellers get the training average)."""
    return df["seller_hist_late_rate"].fillna(train[TARGET].mean()).to_numpy()


def log_figure(path: Path) -> None:
    mlflow.log_artifact(str(path), artifact_path="figures")


def main() -> dict:
    t0 = time.time()
    tracking_setup()
    check_no_leakage(FEATURES)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_modelling_frame()
    s = split(df)
    tr, va, te = s["train"], s["validation"], s["test"]
    Xtr, ytr = tr[FEATURES], tr[TARGET].to_numpy()
    Xva, yva = va[FEATURES], va[TARGET].to_numpy()
    Xte, yte = te[FEATURES], te[TARGET].to_numpy()
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())
    data_tags = {f"rows_{k}": len(v) for k, v in s.items()}
    data_tags.update({f"late_rate_{k}": round(float(v[TARGET].mean()), 4) for k, v in s.items()})

    results, scores_va, scores_te, fitted = {}, {}, {}, {}

    # ---------------- baselines ----------------
    with mlflow.start_run(run_name="baseline-majority-class"):
        dummy = Pipeline([("prep", preprocessor()), ("model", DummyClassifier(strategy="prior"))]).fit(Xtr, ytr)
        sva, ste = dummy.predict_proba(Xva)[:, 1], dummy.predict_proba(Xte)[:, 1]
        mva = ev.classification_metrics(yva, sva, 0.5)
        mte = ev.classification_metrics(yte, ste, 0.5)
        mlflow.set_tags({"model_family": "baseline", **data_tags})
        mlflow.log_params({"strategy": "prior", "threshold": 0.5})
        mlflow.log_metrics({f"val_{k}": v for k, v in mva.items()} | {f"test_{k}": v for k, v in mte.items()})
        results["Majority class"] = {"val": mva, "test": mte, "threshold": 0.5}
        scores_va["Majority class"], scores_te["Majority class"] = sva, ste

    with mlflow.start_run(run_name="baseline-seller-late-rate"):
        sva, ste = heuristic_scores(tr, va), heuristic_scores(tr, te)
        thr = ev.capacity_threshold(sva)
        mva = ev.classification_metrics(yva, sva, pred=ev.daily_capacity_flags(sva, va[TIME_COL]))
        mte = ev.classification_metrics(yte, ste, pred=ev.daily_capacity_flags(ste, te[TIME_COL]))
        mlflow.set_tags({"model_family": "baseline", **data_tags})
        mlflow.log_params({"rule": "seller_hist_late_rate, new sellers = train mean",
                           "decision": "top 25% of each day's orders"})
        mlflow.log_metrics({f"val_{k}": v for k, v in mva.items()} | {f"test_{k}": v for k, v in mte.items()})
        results["Heuristic (seller late rate)"] = {"val": mva, "test": mte, "threshold": thr}
        scores_va["Heuristic (seller late rate)"], scores_te["Heuristic (seller late rate)"] = sva, ste

    # ---------------- tuned models ----------------
    cv = TimeSeriesSplit(n_splits=4)
    for name, (est, space, n_iter) in candidates(pos_weight).items():
        with mlflow.start_run(run_name=name.lower().replace(" ", "-")) as parent:
            pipe = Pipeline([("prep", preprocessor()), ("model", est)])
            search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=cv, scoring="average_precision",
                                        random_state=SEED, n_jobs=1, refit=True, return_train_score=True)
            t1 = time.time()
            search.fit(Xtr, ytr)
            fit_s = time.time() - t1
            cvres = pd.DataFrame(search.cv_results_)
            for i, row in cvres.iterrows():  # one child run per search candidate
                with mlflow.start_run(run_name=f"{name} candidate {i + 1}", nested=True):
                    mlflow.log_params({k.replace("param_model__", ""): row[k] for k in cvres if k.startswith("param_")})
                    mlflow.log_metrics({"cv_pr_auc_mean": row["mean_test_score"],
                                        "cv_pr_auc_std": row["std_test_score"],
                                        "cv_train_pr_auc_mean": row["mean_train_score"]})
            best = search.best_estimator_
            sva, ste = best.predict_proba(Xva)[:, 1], best.predict_proba(Xte)[:, 1]
            thr = ev.capacity_threshold(ste[te[TIME_COL] >= te[TIME_COL].max() - pd.Timedelta(days=30)])
            mva = ev.classification_metrics(yva, sva, pred=ev.daily_capacity_flags(sva, va[TIME_COL]))
            mte = ev.classification_metrics(yte, ste, pred=ev.daily_capacity_flags(ste, te[TIME_COL]))
            mlflow.set_tags({"model_family": name, "split": "time-aware", "cv": "TimeSeriesSplit(4)",
                             "tuning": f"RandomizedSearchCV n_iter={n_iter}, scoring=average_precision",
                             **data_tags})
            mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
            mlflow.log_params({"decision": "top 25% of each day's orders", "serving_threshold_last30d": round(thr, 5),
                               "n_features_in": len(FEATURES),
                               "class_weighting": "scale_pos_weight" if name == "XGBoost" else "balanced"})
            mlflow.log_metrics({"cv_best_pr_auc": search.best_score_, "fit_seconds": fit_s}
                               | {f"val_{k}": v for k, v in mva.items()} | {f"test_{k}": v for k, v in mte.items()})
            cvres.to_csv(FIG_DIR / f"cv_results_{name.lower().replace(' ', '_')}.csv", index=False)
            mlflow.log_artifact(str(FIG_DIR / f"cv_results_{name.lower().replace(' ', '_')}.csv"), "cv")
            sig = infer_signature(Xtr.head(200), best.predict_proba(Xtr.head(200))[:, 1])
            mlflow.sklearn.log_model(best, name="model", signature=sig, input_example=Xtr.head(3),
                                     serialization_format="cloudpickle")
            results[name] = {"val": mva, "test": mte, "threshold": thr, "best_params": search.best_params_,
                             "cv_best_pr_auc": round(float(search.best_score_), 4), "run_id": parent.info.run_id}
            scores_va[name], scores_te[name], fitted[name] = sva, ste, best
            print(f"{name}: cv AP {search.best_score_:.4f} | val AP {mva['pr_auc']} | test AP {mte['pr_auc']} "
                  f"| test recall@25% {mte['recall']} | {fit_s:.0f}s")

    # ---------------- champion ----------------
    # Champion = best recall within daily capacity on VALIDATION (the business goal G1);
    # validation PR-AUC breaks ties. The test set is never used to choose.
    champ = max(fitted, key=lambda n: (results[n]["val"]["recall"], results[n]["val"]["pr_auc"]))
    model, thr = fitted[champ], results[champ]["threshold"]
    ev.plot_roc_pr(yte, {k: scores_te[k] for k in ["XGBoost", "Random Forest", "Logistic Regression",
                                                    "Heuristic (seller late rate)"]}, FIG_DIR / "roc_pr_test.png")
    cm = ev.plot_confusion(yte, ev.daily_capacity_flags(scores_te[champ], te[TIME_COL]),
                           FIG_DIR / "confusion_matrix_test.png", f"{champ}: test period, top 25% of each day")
    with mlflow.start_run(run_id=results[champ]["run_id"]):
        mlflow.set_tags({"champion": "true"})
        log_figure(FIG_DIR / "roc_pr_test.png")
        log_figure(FIG_DIR / "confusion_matrix_test.png")
        mv = mlflow.register_model(f"runs:/{results[champ]['run_id']}/model", MODEL_NAME)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    client.set_model_version_tag(MODEL_NAME, mv.version, "stage", "Staging")
    client.set_model_version_tag(MODEL_NAME, mv.version, "validation_status", "approved-for-pilot")
    client.update_model_version(MODEL_NAME, mv.version,
                                description=f"{champ}. Decision: top 25% of each day's orders; serving threshold "
                                            f"{thr:.4f} = 75th percentile of the last 30 days' scores, "
                                            "recalibrated monthly. Trained on purchases Jan 2017-Feb 2018.")

    meta = {"model_name": MODEL_NAME, "version": int(mv.version), "algorithm": champ,
            "threshold": round(thr, 6), "capacity": ev.CAPACITY,
            "decision_rule": "top 25% of each day's orders by risk; threshold = 75th percentile of the "
                             "last 30 days' scores, recalibrated monthly", "features": FEATURES,
            "numeric": NUMERIC, "categorical": CATEGORICAL, "splits": SPLITS,
            "train_late_rate": round(float(ytr.mean()), 4), "mlflow_run_id": results[champ]["run_id"],
            "test_metrics": results[champ]["test"]}
    if hasattr(model.named_steps["model"], "coef_"):  # linear SHAP needs the training means
        meta["encoded_mean"] = np.asarray(model.named_steps["prep"].transform(Xtr), float).mean(axis=0).round(8).tolist()
    joblib.dump(model, MODEL_DIR / "dri_late_classifier.joblib", compress=3)
    (MODEL_DIR / "dri_late_classifier.json").write_text(json.dumps(meta, indent=2, default=str))
    (FIG_DIR / "confusion_matrix_test.json").write_text(json.dumps(cm.tolist()))

    # ---------------- delivery-days regressor (G3, promise gap) ----------------
    rtr = tr.dropna(subset=[REGRESSION_TARGET])
    rte = te.dropna(subset=[REGRESSION_TARGET])
    with mlflow.start_run(run_name="regressor-delivery-days"):
        reg = Pipeline([("prep", preprocessor()),
                        ("model", XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                                               colsample_bytree=0.8, objective="reg:absoluteerror",
                                               n_jobs=2, random_state=SEED))]).fit(rtr[FEATURES], rtr[REGRESSION_TARGET])
        pred = reg.predict(rte[FEATURES])
        base = np.full(len(rte), rtr[REGRESSION_TARGET].median())
        m_model = ev.regression_metrics(rte[REGRESSION_TARGET], pred)
        m_base = ev.regression_metrics(rte[REGRESSION_TARGET], base)
        ev.plot_residuals(rte[REGRESSION_TARGET].to_numpy(), pred, FIG_DIR / "residuals_regressor.png")
        mlflow.set_tags({"model_family": "XGBoost regressor", "target": REGRESSION_TARGET, **data_tags})
        mlflow.log_params({"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05, "objective": "reg:absoluteerror"})
        mlflow.log_metrics({f"test_{k}": v for k, v in m_model.items()} | {f"baseline_{k}": v for k, v in m_base.items()})
        log_figure(FIG_DIR / "residuals_regressor.png")
        mlflow.sklearn.log_model(reg, name="model", serialization_format="cloudpickle")
    joblib.dump(reg, MODEL_DIR / "dri_delivery_days_regressor.joblib", compress=3)
    results["regressor"] = {"model": m_model, "baseline_median": m_base}

    summary = {"champion": champ, "threshold": thr, "registered_version": int(mv.version),
               "results": results, "minutes": round((time.time() - t0) / 60, 1)}
    (FIG_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, default=str))
    return summary


if __name__ == "__main__":
    main()
