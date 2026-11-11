"""Synthetic Olist-shaped tables for unit tests, generated with Faker.

Small, fast and fully made-up: tests never touch the real dataset or real people.
"""
from __future__ import annotations

import random
from datetime import timedelta

import pandas as pd
from faker import Faker

STATES = ["SP", "RJ", "MG", "BA", "PE", "AM", "PR", "RS", "GO", "PA"]


def make_raw(n_orders: int = 60, seed: int = 7) -> dict[str, pd.DataFrame]:
    fake = Faker("pt_BR")
    Faker.seed(seed)
    random.seed(seed)

    sellers = pd.DataFrame({
        "seller_id": [fake.md5() for _ in range(5)],
        "seller_zip_code_prefix": [f"{random.randint(1000, 99999)}" for _ in range(5)],
        "seller_city": [fake.city() for _ in range(5)],
        "seller_state": [random.choice(STATES) for _ in range(5)],
    })
    products = pd.DataFrame({
        "product_id": [fake.md5() for _ in range(8)],
        "product_category_name": ["beleza_saude", "esporte_lazer", None, "beleza_saude",
                                  "moveis_decoracao", "esporte_lazer", "beleza_saude", None],
        "product_name_lenght": [40] * 8, "product_description_lenght": [300] * 8,
        "product_photos_qty": [1] * 8,
        "product_weight_g": [500, 1200, None, 300, 0, 800, 650, 2000],
        "product_length_cm": [20, 30, 25, 15, 40, 30, 20, 50],
        "product_height_cm": [10, 10, 10, 5, 30, 10, 10, 20],
        "product_width_cm": [15, 20, 20, 10, 40, 20, 15, 30],
    })
    translation = pd.DataFrame({
        "product_category_name": ["beleza_saude", "esporte_lazer", "moveis_decoracao"],
        "product_category_name_english": ["health_beauty", "sports_leisure", "furniture_decor"],
    })

    orders, customers, items, payments, reviews = [], [], [], [], []
    base = pd.Timestamp("2017-03-01 10:00")
    for i in range(n_orders):
        oid, cid, uid = fake.md5(), fake.md5(), fake.md5()
        purchase = base + timedelta(days=i, hours=random.randint(0, 10))
        approved = purchase + timedelta(hours=random.randint(0, 30))
        carrier = approved + timedelta(days=random.randint(1, 4))
        delivered = carrier + timedelta(days=random.randint(2, 20))
        estimated = (purchase + timedelta(days=15)).normalize()
        orders.append({"order_id": oid, "customer_id": cid, "order_status": "delivered",
                       "order_purchase_timestamp": purchase, "order_approved_at": approved,
                       "order_delivered_carrier_date": carrier,
                       "order_delivered_customer_date": delivered,
                       "order_estimated_delivery_date": estimated})
        customers.append({"customer_id": cid, "customer_unique_id": uid,
                          "customer_zip_code_prefix": f"{random.randint(1000, 99999)}",
                          "customer_city": fake.city(), "customer_state": random.choice(STATES)})
        seller = sellers.sample(1, random_state=i).iloc[0]["seller_id"]
        product = products.sample(1, random_state=i).iloc[0]["product_id"]
        items.append({"order_id": oid, "order_item_id": 1, "product_id": product, "seller_id": seller,
                      "shipping_limit_date": approved + timedelta(days=3),
                      "price": round(random.uniform(20, 400), 2),
                      "freight_value": round(random.uniform(5, 60), 2)})
        payments.append({"order_id": oid, "payment_sequential": 1,
                         "payment_type": random.choice(["credit_card", "boleto", "voucher"]),
                         "payment_installments": random.randint(1, 6),
                         "payment_value": round(random.uniform(30, 450), 2)})
        reviews.append({"review_id": fake.md5(), "order_id": oid, "review_score": random.randint(1, 5),
                        "review_comment_title": fake.first_name(),
                        "review_comment_message": fake.sentence(),
                        "review_creation_date": delivered, "review_answer_timestamp": delivered})

    geo_rows = []
    for z in list({c["customer_zip_code_prefix"] for c in customers} | set(sellers["seller_zip_code_prefix"])):
        for _ in range(2):
            geo_rows.append({"geolocation_zip_code_prefix": z,
                             "geolocation_lat": random.uniform(-30, -3),
                             "geolocation_lng": random.uniform(-60, -36),
                             "geolocation_city": fake.city(), "geolocation_state": random.choice(STATES)})
    geo_rows.append({"geolocation_zip_code_prefix": "99999", "geolocation_lat": 40.0,
                     "geolocation_lng": -3.0, "geolocation_city": "Madrid", "geolocation_state": "SP"})

    raw = {"orders": pd.DataFrame(orders), "items": pd.DataFrame(items),
           "payments": pd.DataFrame(payments), "reviews": pd.DataFrame(reviews),
           "customers": pd.DataFrame(customers), "sellers": sellers, "products": products,
           "geolocation": pd.DataFrame(geo_rows), "category_translation": translation}
    for df in raw.values():
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype("string")
    return raw
