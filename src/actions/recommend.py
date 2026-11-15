"""Suggested actions for a flagged order (Module 1, FR4: the prescriptive layer).

Simple, auditable rules that turn the top SHAP reasons into low-cost actions.
They follow the Ethical AI Charter:

* advice only: a named person decides (Accountability);
* seller actions are support, never penalties (ER3, "Support, not punishment");
* no action pads the promised date for one region only (ER5): a date update is
  suggested from the order's own expected delivery time, for any customer.

Actions are ranked by cost: cheapest first. Rules are deliberately transparent
rather than learned, because their effect has not yet been measured in a trial
(goals G2 and G4).
"""
from __future__ import annotations

ACTIONS = {
    "update_date": {"action": "Update the delivery date shown to the customer",
                    "who": "Customer service", "cost": 1},
    "message_customer": {"action": "Send the customer an early, honest delivery update",
                         "who": "Customer service", "cost": 1},
    "remind_seller": {"action": "Remind the seller to hand the parcel to the courier today",
                      "who": "Seller support", "cost": 2},
    "support_seller": {"action": "Offer the seller packing and hand-over support (not a penalty)",
                       "who": "Seller support", "cost": 3},
    "watch_route": {"action": "Check the courier's progress on this long route at hand-over",
                    "who": "Delivery operations", "cost": 3},
}

# Which reasons point to which actions
RULES = {
    "promised_days": ["update_date", "message_customer"],
    "promise_slack_days": ["update_date", "message_customer"],
    "state_delivery_days_30d": ["update_date", "message_customer"],
    "state_late_rate_30d": ["message_customer"],
    "market_late_rate_14d": ["message_customer"],
    "is_peak_season": ["message_customer", "remind_seller"],
    "seller_hist_late_rate": ["remind_seller", "support_seller"],
    "seller_hist_handling_days": ["remind_seller", "support_seller"],
    "seller_orders_7d": ["remind_seller"],
    "distance_km": ["watch_route"],
    "interstate": ["watch_route"],
    "customer_state": ["watch_route"],
}


def suggest(reasons: list[dict], flagged: bool, expected_days_late: float | None = None, k: int = 3) -> list[dict]:
    """Ranked suggested actions for one order. Empty if the order is not flagged."""
    if not flagged:
        return []
    keys: list[str] = []
    if expected_days_late is not None and expected_days_late > 0:
        keys.append("update_date")
    for r in reasons:
        for a in RULES.get(r.get("feature", ""), []):
            if a not in keys:
                keys.append(a)
    if not keys:
        keys = ["message_customer"]
    first_seen = {a: i for i, a in enumerate(keys)}
    ranked = sorted(keys, key=lambda a: (ACTIONS[a]["cost"], first_seen[a]))[:k]
    return [{"id": a, **ACTIONS[a]} for a in ranked]
