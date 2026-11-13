"""SHAP explanations for the champion model, in business language.

The model sees ~90 encoded columns (one-hot categories, missing-value flags).
People think in the 27 original features, so SHAP values are summed back to
the feature they came from before they are shown (Module 2 Explainability
Framework: local reasons for operations, global summary for the Review Board).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLAIN = {
    "promised_days": "Days promised to the customer",
    "purchase_dow": "Day of week of purchase", "purchase_month": "Month of purchase",
    "purchase_hour": "Hour of purchase", "is_peak_season": "Black Friday to Christmas period",
    "distance_km": "Seller-to-customer distance", "interstate": "Seller and customer in different states",
    "n_items": "Items in the order", "n_sellers": "Sellers in the order",
    "price_total": "Order value", "freight_total": "Freight charged", "freight_ratio": "Freight as share of price",
    "weight_total_g": "Parcel weight", "volume_total_cm3": "Parcel volume",
    "any_size_imputed": "Product size was missing", "is_boleto": "Paid by boleto (bank slip)",
    "payment_installments": "Payment instalments", "n_payment_methods": "Payment methods used",
    "seller_hist_orders": "Seller's past delivered orders", "seller_hist_late_rate": "Seller's past late rate",
    "seller_hist_handling_days": "Seller's past handling time",
    "state_late_rate_30d": "Late rate in customer's state, last 30 days",
    "state_delivery_days_30d": "Delivery time in customer's state, last 30 days",
    "promise_slack_days": "Promise slack vs recent delivery time",
    "market_late_rate_14d": "Late rate across Brazil, last 14 days",
    "seller_orders_7d": "Seller's orders in the last 7 days",
    "customer_state": "Customer's state", "seller_state": "Seller's state",
    "customer_region": "Customer's region", "primary_category": "Product category",
    "payment_type": "Payment type", "seller_size": "Seller size (history)",
}


def original_feature(encoded: str, originals: list[str]) -> str:
    """Map an encoded column name back to the feature it came from."""
    name = encoded.replace("missingindicator_", "")
    for f in sorted(originals, key=len, reverse=True):  # longest first: seller_state before seller_
        if name == f or name.startswith(f + "_"):
            return f
    return name


class Explainer:
    """SHAP for the champion pipeline.

    Tree models (Random Forest, XGBoost) use TreeExplainer. A logistic
    regression uses exact linear SHAP: contribution = coefficient x (value -
    training mean) on the scaled, encoded features, which is what
    shap.LinearExplainer computes with an independent masker. The training
    means are stored next to the model so the API needs no training data.
    """

    def __init__(self, pipeline, background: pd.DataFrame | None, features: list[str],
                 mean_encoded: np.ndarray | None = None):
        self.pipe, self.features = pipeline, features
        self.prep, self.model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
        self.encoded_names = list(self.prep.get_feature_names_out())
        self.groups = [original_feature(n, features) for n in self.encoded_names]
        self.linear = hasattr(self.model, "coef_")
        if self.linear:
            if mean_encoded is None:
                mean_encoded = self.prep.transform(background[features]).mean(axis=0)
            self.mean_z = np.asarray(mean_encoded, dtype=float)
            self.coef = self.model.coef_.ravel()
            self._base = float(self.model.intercept_[0] + self.coef @ self.mean_z)
        else:
            import shap  # only tree models need the shap library

            self.tree = shap.TreeExplainer(self.model, feature_perturbation="tree_path_dependent")
            ev = self.tree.expected_value
            self._base = float(ev[1] if np.ndim(ev) else ev)

    def encoded_values(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Z = np.asarray(self.prep.transform(X[self.features]), dtype=float)
        if self.linear:
            return Z, (Z - self.mean_z) * self.coef
        sv = self.tree.shap_values(Z)
        if isinstance(sv, list):          # older shap: one array per class
            sv = sv[1]
        if sv.ndim == 3:                  # (n, features, classes)
            sv = sv[:, :, 1]
        return Z, sv

    def grouped(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values (log-odds) summed per original feature: rows = orders."""
        _, sv = self.encoded_values(X)
        df = pd.DataFrame(sv, columns=self.encoded_names)
        return df.T.groupby(self.groups).sum().T[self.features]

    def base_value(self) -> float:
        return self._base

    def reasons(self, row: pd.DataFrame, k: int = 3) -> list[dict]:
        """Top-k reasons that push this order towards 'late', in plain words."""
        g = self.grouped(row).iloc[0].sort_values(ascending=False)
        out = []
        for f, v in g.head(k).items():
            if v <= 0:
                break
            val = row.iloc[0][f]
            out.append({"feature": f, "reason": PLAIN.get(f, f),
                        "value": None if pd.isna(val) else (round(float(val), 2) if isinstance(val, (int, float, np.number))
                                                            else val),
                        "impact_log_odds": round(float(v), 3)})
        return out
