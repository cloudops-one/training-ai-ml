"""ShopWave churn API: the saved churn model behind a web address.

Start it from this folder, with the course environment active:

    python -m uvicorn api:app --reload            # Windows, macOS and Linux

Then open http://127.0.0.1:8000/docs for an interactive page that lets you try every endpoint.
"""
import sys
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
from churn_model import load_bundle  # noqa: E402

bundle = load_bundle()          # loaded once, when the server starts


class Customer(BaseModel):
    """One customer's history up to today. Field limits reject impossible values."""
    orders: int = Field(ge=1, le=10_000, description="Number of orders placed", examples=[2])
    spend: float = Field(ge=0, le=10_000_000, description="Total spent, in £", examples=[310.0])
    products: int = Field(ge=1, le=100_000, description="Number of different products bought", examples=[14])
    tenure_days: int = Field(ge=0, le=3_650, description="Days since the first order", examples=[150])
    days_since_last: int = Field(ge=0, le=3_650, description="Days since the latest order", examples=[95])

    @model_validator(mode="after")
    def last_order_not_before_first(self):
        if self.days_since_last > self.tenure_days:
            raise ValueError("days_since_last cannot be larger than tenure_days")
        return self


class Prediction(BaseModel):
    churn_probability: float
    voucher: bool
    threshold: float


app = FastAPI(title="ShopWave churn API", version="1.0",
              description="Churn probability and voucher decision for ShopWave customers.")


@app.get("/health")
def health():
    """Is the service up, and which model is it serving?"""
    return {"status": "ok", "model_trained": bundle["trained"], "test_auc": bundle["metrics"]["test_auc"],
            "threshold": bundle["threshold"], "features": bundle["features"]}


def _predict(customers: list[Customer]) -> list[Prediction]:
    rows = pd.DataFrame([c.model_dump() for c in customers])[bundle["features"]]
    probabilities = bundle["pipeline"].predict_proba(rows)[:, 1]
    return [Prediction(churn_probability=round(float(p), 4), voucher=bool(p >= bundle["threshold"]),
                       threshold=bundle["threshold"]) for p in probabilities]


@app.post("/predict")
def predict(customer: Customer) -> Prediction:
    """Score one customer."""
    return _predict([customer])[0]


@app.post("/predict/batch")
def predict_batch(customers: list[Customer]) -> list[Prediction]:
    """Score many customers in one request (much faster than one request each)."""
    return _predict(customers)
