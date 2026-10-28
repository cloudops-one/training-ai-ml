"""Retrain the final model from the processed table and save it.

Run from the project root:  python src/train.py

This is the script version of notebooks/03_models.ipynb: once you have chosen
a model in the notebook, copy the winning pipeline here so anyone can rebuild
it with one command.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "data" / "processed" / "table.parquet"
MODELS = ROOT / "models"

FEATURES = ["TODO_feature_1", "TODO_feature_2"]   # TODO: your feature columns
TARGET = "TODO_target"                            # TODO: your target column
RANDOM_STATE = 42


def build_model():
    # TODO: replace with the pipeline you chose in notebooks/03_models.ipynb
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def main():
    table = pd.read_parquet(TABLE)
    # TODO: if your split is by time, split on a date column instead
    X_train, X_test, y_train, y_test = train_test_split(
        table[FEATURES], table[TARGET], test_size=0.25,
        random_state=RANDOM_STATE, stratify=table[TARGET])

    model = build_model().fit(X_train, y_train)
    score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])   # TODO: your metric
    print(f"test score: {score:.3f}")

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "test_score": score},
                MODELS / "model.joblib")
    print("saved", MODELS / "model.joblib")


if __name__ == "__main__":
    main()
