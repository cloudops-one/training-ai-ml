"""Train the production churn model and record the run in MLflow.

    python train.py

Writes models/churn_bundle.joblib (used by api.py and app.py) and logs a run, with
its parameters, metrics and the bundle file, to mlruns/ at the course root.
Browse runs with:  python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
(run that from the course root).
"""
import datetime
import logging
import os
import sys
from pathlib import Path


def find_lib() -> Path:
    for folder in Path(__file__).resolve().parents:
        if (folder / "lib" / "churn_model.py").exists():
            return folder / "lib"
    sys.exit("Cannot find lib/churn_model.py. Run this from inside the course folder.")


sys.path.insert(0, str(find_lib()))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import joblib  # noqa: E402
import mlflow  # noqa: E402
import churn_model as cm  # noqa: E402

EXPERIMENT = "shopwave-churn-production"


def main() -> str:
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    root = cm.repo_root()
    (root / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlruns' / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(root / "mlruns" / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=f"production-{datetime.date.today()}") as run:
        bundle = cm.train_and_save(verbose=False)
        mlflow.log_params({"model": "logistic_regression", "features": ",".join(bundle["features"]),
                           "cutoff": cm.CUTOFF, "horizon_days": cm.HORIZON_DAYS, "split_seed": 42,
                           "voucher_cost": cm.VOUCHER_COST, "lost_customer_cost": cm.LOST_CUSTOMER_COST,
                           "threshold": bundle["threshold"], "sklearn_version": bundle["sklearn_version"]})
        mlflow.log_metrics(bundle["metrics"])
        bundle["mlflow_run_id"] = run.info.run_id        # the file now says which run made it
        joblib.dump(bundle, cm.MODEL_PATH)
        mlflow.log_artifact(str(cm.MODEL_PATH))
    print(f"Trained and saved {cm.MODEL_PATH.name}: test AUC {bundle['metrics']['test_auc']}, "
          f"threshold {bundle['threshold']}, MLflow run {run.info.run_id[:8]}")
    return run.info.run_id


if __name__ == "__main__":
    main()
