"""ShopWave churn API, production version: the 10.2 API plus a prediction log.

    python -m uvicorn api:app            # from this folder, Windows / macOS / Linux

Every prediction is appended to models/prediction_log.csv (or PREDICTION_LOG if set),
which the monitoring report reads. Interactive docs: http://127.0.0.1:8000/docs
"""
import csv
import datetime
import os
import sys
import threading
from pathlib import Path


def find_lib() -> Path:
    for folder in Path(__file__).resolve().parents:
        if (folder / "lib" / "churn_model.py").exists():
            return folder / "lib"
    sys.exit("Cannot find lib/churn_model.py. Run this from inside the course folder.")


sys.path.insert(0, str(find_lib()))
import pandas as pd  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel, Field, model_validator  # noqa: E402
import churn_model as cm  # noqa: E402

bundle = cm.load_bundle()
LOG_PATH = Path(os.environ.get("PREDICTION_LOG", cm.MODEL_PATH.parent / "prediction_log.csv"))
LOG_COLUMNS = ["timestamp", "model_trained", *bundle["features"], "churn_probability", "voucher"]
_log_lock = threading.Lock()


class Customer(BaseModel):
    orders: int = Field(ge=1, le=10_000)
    spend: float = Field(ge=0, le=10_000_000)
    products: int = Field(ge=1, le=100_000)
    tenure_days: int = Field(ge=0, le=3_650)
    days_since_last: int = Field(ge=0, le=3_650)

    @model_validator(mode="after")
    def last_order_not_before_first(self):
        if self.days_since_last > self.tenure_days:
            raise ValueError("days_since_last cannot be larger than tenure_days")
        return self


class Prediction(BaseModel):
    churn_probability: float
    voucher: bool
    threshold: float


app = FastAPI(title="ShopWave churn API", version="1.1")


def _log(rows: pd.DataFrame, probabilities, decisions) -> None:
    """Append one line per prediction; the monitoring report compares these with training data."""
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with _log_lock:
        new_file = not LOG_PATH.exists()
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(LOG_COLUMNS)
            for values, p, v in zip(rows.itertuples(index=False), probabilities, decisions):
                writer.writerow([now, bundle["trained"], *values, round(float(p), 4), bool(v)])


def _predict(customers: list[Customer]) -> list[Prediction]:
    rows = pd.DataFrame([c.model_dump() for c in customers])[bundle["features"]]
    p = bundle["pipeline"].predict_proba(rows)[:, 1]
    decisions = p >= bundle["threshold"]
    _log(rows, p, decisions)
    return [Prediction(churn_probability=round(float(x), 4), voucher=bool(v), threshold=bundle["threshold"])
            for x, v in zip(p, decisions)]


@app.get("/health")
def health():
    return {"status": "ok", "model_trained": bundle["trained"], "test_auc": bundle["metrics"]["test_auc"],
            "threshold": bundle["threshold"], "mlflow_run_id": bundle.get("mlflow_run_id")}


@app.get("/model")
def model_card():
    """What the model is for, what it was trained on, and how well it did."""
    return {"purpose": "Rank customers by churn risk to target win-back vouchers. Not for any other decision.",
            "data": bundle["data"], "features": bundle["features"], "metrics": bundle["metrics"],
            "threshold": bundle["threshold"], "trained": bundle["trained"]}


@app.post("/predict")
def predict(customer: Customer) -> Prediction:
    return _predict([customer])[0]


@app.post("/predict/batch")
def predict_batch(customers: list[Customer]) -> list[Prediction]:
    return _predict(customers)
