"""Generate the small ShopWave tables used for by-hand examples.

Run: python data/make_small_tables.py

The numbers are fixed (seeded) so the lessons' hand calculations always
match the files. Re-running produces identical CSVs.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
rng = np.random.default_rng(42)


def customers_small() -> pd.DataFrame:
    """20 customers. Used in Modules 02-04 for hand calculations.

    Columns:
      customer_id        - C001 ... C020
      visits_last_month  - website visits in the last 30 days
      items_last_month   - items bought in the last 30 days
      days_since_last    - days since the last order
      country            - UK / DE / FR
      spend_next_month   - target for regression (what they spent next month)
      churned            - target for classification (1 = no order in next 90 days)
    """
    n = 20
    visits = rng.integers(1, 15, n)
    items = np.clip((visits * 0.6 + rng.normal(0, 1.2, n)).round(), 0, None).astype(int)
    days_since = rng.integers(1, 60, n)
    country = rng.choice(["UK", "DE", "FR"], n, p=[0.5, 0.3, 0.2])
    spend = 12 * visits + 9 * items - 0.8 * days_since + rng.normal(0, 12, n) + 40
    spend = spend.clip(0).round(2)
    churn_prob = 1 / (1 + np.exp(-(0.08 * days_since - 0.35 * visits - 0.5)))
    churned = (rng.random(n) < churn_prob).astype(int)
    return pd.DataFrame({
        "customer_id": [f"C{i:03d}" for i in range(1, n + 1)],
        "visits_last_month": visits,
        "items_last_month": items,
        "days_since_last": days_since,
        "country": country,
        "spend_next_month": spend,
        "churned": churned,
    })


def visits_spend_5() -> pd.DataFrame:
    """The 5-row table used for linear regression by hand (lesson 02.2).

    Chosen so the arithmetic is clean: x-mean = 6, y-mean = 100.
    """
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "visits_last_month": [2, 4, 6, 8, 10],
        "spend_next_month": [50, 80, 90, 130, 150],
    })


if __name__ == "__main__":
    customers_small().to_csv(HERE / "shopwave_customers_small.csv", index=False)
    visits_spend_5().to_csv(HERE / "shopwave_visits_spend_5.csv", index=False)
    print("Wrote shopwave_customers_small.csv and shopwave_visits_spend_5.csv")
