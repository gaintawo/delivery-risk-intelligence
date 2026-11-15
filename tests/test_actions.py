from src.actions.recommend import suggest


def test_not_flagged_gets_no_actions():
    assert suggest([{"feature": "promised_days"}], flagged=False) == []


def test_seller_reasons_lead_to_support_and_are_ranked_by_cost():
    acts = suggest([{"feature": "seller_hist_late_rate"}, {"feature": "distance_km"}], flagged=True)
    ids = [a["id"] for a in acts]
    assert ids[0] == "remind_seller" and "support_seller" in ids
    assert [a["cost"] for a in acts] == sorted(a["cost"] for a in acts)


def test_expected_lateness_triggers_date_update():
    acts = suggest([{"feature": "distance_km"}], flagged=True, expected_days_late=2.0)
    assert acts[0]["id"] == "update_date"
