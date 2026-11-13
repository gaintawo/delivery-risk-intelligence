"""Metrics and plots for the late-delivery models.

Decision policy (Module 1, G1): the operations team reviews a daily list.
Each day, the 25% of that day's orders with the highest risk scores are
flagged (~50 orders a day at this dataset's volume). Ranking within the day
uses no outcome information. A fixed probability threshold was tried first
(MLflow model v2): it flagged only 7% of test orders, because scores fall in
calm periods, so it was replaced by this daily-capacity rule.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

CAPACITY = 0.25  # share of orders the team can act on

# Validated palette (dataviz validator, light mode): teal / orange / slate / grey.
TEAL, ORANGE, SLATE, GREY, INK, MUTED = "#00927F", "#D9691A", "#5B6B7A", "#94A3B8", "#1B2430", "#5B6B7A"
SERIES = {"XGBoost": TEAL, "Random Forest": ORANGE, "Logistic Regression": SLATE,
          "Heuristic (seller late rate)": GREY, "Majority class": "#C4CDD5"}


def capacity_threshold(scores: np.ndarray, capacity: float = CAPACITY) -> float:
    return float(np.quantile(scores, 1 - capacity))


def daily_capacity_flags(score, dates, capacity: float = CAPACITY) -> np.ndarray:
    """Flag the top `capacity` share of each day's orders by risk score."""
    s = pd.Series(np.asarray(score, dtype=float))
    day = pd.Series(pd.to_datetime(np.asarray(dates))).dt.normalize()
    cut = s.groupby(day.values).transform(lambda x: x.quantile(1 - capacity))
    return (s >= cut).to_numpy().astype(int)


def classification_metrics(y, score, threshold=None, pred=None) -> dict:
    if pred is None:
        pred = (score >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "flag_rate": float(pred.mean()),
        "brier": brier_score_loss(y, np.clip(score, 0, 1)),
    }
    if len(np.unique(score)) > 1:
        out["roc_auc"] = roc_auc_score(y, score)
        out["pr_auc"] = average_precision_score(y, score)
    else:  # constant scores (majority-class baseline)
        out["roc_auc"] = 0.5
        out["pr_auc"] = float(np.mean(y))
    return {k: round(float(v), 4) for k, v in out.items()}


def regression_metrics(y, pred) -> dict:
    return {"mae": round(float(mean_absolute_error(y, pred)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 3),
            "r2": round(float(r2_score(y, pred)), 4)}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C4CDD5")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color="#E6EBEF", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_roc_pr(y, scores: dict[str, np.ndarray], path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    for name, s in scores.items():
        c = SERIES.get(name, SLATE)
        lw = 2.2 if name == "XGBoost" else 1.4
        if len(np.unique(s)) > 1:
            fpr, tpr, _ = roc_curve(y, s)
            axes[0].plot(fpr, tpr, color=c, lw=lw, label=f"{name} (AUC {roc_auc_score(y, s):.3f})")
            p, r, _ = precision_recall_curve(y, s)
            axes[1].plot(r, p, color=c, lw=lw, label=f"{name} (AP {average_precision_score(y, s):.3f})")
    axes[0].plot([0, 1], [0, 1], ls="--", color="#C4CDD5", lw=1, label="Random (AUC 0.500)")
    axes[1].axhline(np.mean(y), ls="--", color="#C4CDD5", lw=1, label=f"Base rate ({np.mean(y):.3f})")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    for ax in axes:
        _style(ax)
        ax.title.set_color(INK)
        ax.title.set_fontsize(11)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right" if ax is axes[0] else "upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(y, pred, path, title):
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=200)
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("t", ["#EEF3F2", TEAL]))
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            dark = v > cm.max() * 0.55
            ax.text(j, i - 0.08, f"{v:,}", ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if dark else INK)
            ax.text(j, i + 0.2, labels[i][j], ha="center", va="center", fontsize=8,
                    color="white" if dark else MUTED)
    ax.set_xticks([0, 1], ["Predicted on time", "Predicted late"])
    ax.set_yticks([0, 1], ["Actually on time", "Actually late"])
    ax.tick_params(colors=INK, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return cm


def plot_residuals(y, pred, path):
    res = y - pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200)
    axes[0].scatter(pred, res, s=3, alpha=0.25, color=TEAL, linewidths=0)
    axes[0].axhline(0, color=INK, lw=1)
    axes[0].set(xlabel="Predicted delivery days", ylabel="Residual (actual - predicted, days)",
                title="Residuals vs predicted")
    axes[1].hist(res.clip(-30, 40), bins=70, color=TEAL, alpha=0.85)
    axes[1].axvline(0, color=INK, lw=1)
    axes[1].set(xlabel="Residual (days, clipped to -30..40)", ylabel="Orders", title="Residual distribution")
    for ax in axes:
        _style(ax)
        ax.title.set_color(INK)
        ax.title.set_fontsize(11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
