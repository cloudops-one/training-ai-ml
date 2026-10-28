"""The ShopWave churn model, packaged for Module 10.

Everything a deployed model needs lives here, so the training script, the batch
scorer, the API and the web app all use exactly the same definitions:

    from churn_model import build_customer_table, train_and_save, load_bundle, score

The model is the one chosen in lesson 05.10: scaled logistic regression on five
features, decision threshold chosen from voucher costs (£5 voucher, £60 lost customer).
"""
from __future__ import annotations

import datetime as _dt
import os
import sys
from pathlib import Path

FEATURES = ["orders", "spend", "products", "tenure_days", "days_since_last"]
VOUCHER_COST, LOST_CUSTOMER_COST = 5, 60
CUTOFF = "2011-09-01"          # features from orders before this date, churn = no order in the next 91 days
HORIZON_DAYS = 91


def repo_root() -> Path:
    """The folder that contains data/download.py, searched upwards from here."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "data" / "download.py").exists():
            return parent
    raise FileNotFoundError("Cannot find the course folder (data/download.py).")


def _default_model_path() -> Path:
    """CHURN_MODEL_PATH if set (e.g. inside a Docker image), else <course>/models/."""
    if os.environ.get("CHURN_MODEL_PATH"):
        return Path(os.environ["CHURN_MODEL_PATH"])
    try:
        return repo_root() / "models" / "churn_bundle.joblib"
    except FileNotFoundError:
        return Path("models") / "churn_bundle.joblib"


MODEL_PATH = _default_model_path()


def load_sales(both_years: bool = False):
    """Cleaned ShopWave order lines with a Revenue column (the 01.5 cleaning)."""
    import pandas as pd
    sys.path.insert(0, str(repo_root() / "data"))
    from download import load_retail

    raw = load_retail("Year 2010-2011")
    if both_years:   # the sheets overlap from 1 to 9 December 2010: count those days once
        older = load_retail("Year 2009-2010")
        raw = pd.concat([older[older["InvoiceDate"] < "2010-12-01"], raw], ignore_index=True)
    sales = raw[~raw["Invoice"].astype(str).str.startswith("C")]
    sales = sales[(sales["Quantity"] > 0) & (sales["Price"] > 0)].dropna(subset=["Customer ID"])
    sales = sales.drop_duplicates()
    return sales.assign(Revenue=sales["Quantity"] * sales["Price"])


def build_customer_table(sales, cutoff=CUTOFF, lookback_days=None, horizon_days=HORIZON_DAYS):
    """One row per customer who ordered before `cutoff`.

    Features use orders before the cutoff (optionally only the last `lookback_days`).
    `churned` = 1 if the customer placed no order in the `horizon_days` after it.
    `country` is the customer's most common country; it is NOT a model feature.
    """
    import pandas as pd
    cutoff = pd.Timestamp(cutoff)
    start = cutoff - pd.Timedelta(days=lookback_days) if lookback_days else sales["InvoiceDate"].min()
    past = sales[(sales["InvoiceDate"] >= start) & (sales["InvoiceDate"] < cutoff)]
    future = sales[(sales["InvoiceDate"] >= cutoff)
                   & (sales["InvoiceDate"] < cutoff + pd.Timedelta(days=horizon_days))]
    cust = past.groupby("Customer ID").agg(
        orders=("Invoice", "nunique"), spend=("Revenue", "sum"), products=("StockCode", "nunique"),
        first_order=("InvoiceDate", "min"), last_order=("InvoiceDate", "max"),
        country=("Country", lambda s: s.mode().iloc[0]))
    cust["tenure_days"] = (cutoff - cust["first_order"]).dt.days
    cust["days_since_last"] = (cutoff - cust["last_order"]).dt.days
    came_back = future.groupby("Customer ID")["Revenue"].sum().reindex(cust.index).fillna(0) > 0
    cust["churned"] = (~came_back).astype(int)
    return cust[FEATURES + ["country", "churned"]]


def expected_cost(y, p, threshold, c_voucher=VOUCHER_COST, c_lost=LOST_CUSTOMER_COST):
    """£ cost of sending a voucher to everyone with p >= threshold (03.7)."""
    import numpy as np
    y, flag = np.asarray(y), np.asarray(p) >= threshold
    return c_voucher * flag.sum() + c_lost * ((y == 1) & ~flag).sum()


def train_and_save(path=MODEL_PATH, verbose=True) -> dict:
    """Train the 05.10 model on the standard split and save a bundle. Returns the bundle."""
    import joblib, numpy as np, sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cust = build_customer_table(load_sales())
    X_train, X_test, y_train, y_test = train_test_split(
        cust[FEATURES], cust["churned"], test_size=0.25, random_state=42, stratify=cust["churned"])
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_cv = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    threshold = float(grid[np.argmin([expected_cost(y_train, p_cv, t) for t in grid])])
    pipeline.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    bundle = {
        "pipeline": pipeline, "features": FEATURES, "threshold": threshold,
        "metrics": {"cv_auc": round(roc_auc_score(y_train, p_cv), 3), "test_auc": round(test_auc, 3)},
        "data": f"{len(cust):,} customers, orders before {CUTOFF}, churn = no order in next {HORIZON_DAYS} days",
        "trained": _dt.date.today().isoformat(), "sklearn_version": sklearn.__version__,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    if verbose:
        print(f"Saved {Path(path).name}: test AUC {test_auc:.3f}, threshold {threshold}")
    return bundle


def load_bundle(path=MODEL_PATH, train_if_missing=True) -> dict:
    """Load the saved bundle, training it first if the file does not exist yet."""
    import joblib, sklearn, warnings
    if not Path(path).exists():
        if not train_if_missing:
            raise FileNotFoundError(f"{path} not found. Run train_churn.py first.")
        return train_and_save(path)
    bundle = joblib.load(path)
    if bundle["sklearn_version"] != sklearn.__version__:
        warnings.warn(f"Model saved with scikit-learn {bundle['sklearn_version']}, "
                      f"running {sklearn.__version__}. Retrain if predictions look wrong.")
    return bundle


def psi(expected, actual, bins=10) -> float:
    """Population Stability Index between a reference sample and a new one (lesson 10.6).

    Bins are deciles of the reference sample; empty bins are floored at 0.01%.
    Rule of thumb: < 0.1 stable, 0.1-0.25 watch, > 0.25 investigate.
    """
    import numpy as np
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(((a - e) * np.log(a / e)).sum())


def score(bundle: dict, customers) -> "pd.DataFrame":
    """Churn probability and voucher decision for a DataFrame of customers."""
    import pandas as pd
    missing = [f for f in bundle["features"] if f not in customers.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    p = bundle["pipeline"].predict_proba(customers[bundle["features"]])[:, 1]
    return pd.DataFrame({"churn_probability": p.round(4),
                         "voucher": p >= bundle["threshold"]}, index=customers.index)
