"""Train the ShopWave churn model and save it as a bundle in <course>/models/.

Run from anywhere inside the course folder, with the course environment active:

    python train_churn.py

The steps (data, features, pipeline, threshold, bundle) are in lib/churn_model.py,
and lesson 10.1 walks through them one by one.
"""
import sys
from pathlib import Path


def find_lib() -> Path:
    for folder in Path(__file__).resolve().parents:
        if (folder / "lib" / "churn_model.py").exists():
            return folder / "lib"
    sys.exit("Cannot find lib/churn_model.py. Run this from inside the course folder.")


sys.path.insert(0, str(find_lib()))
from churn_model import MODEL_PATH, train_and_save  # noqa: E402

if __name__ == "__main__":
    bundle = train_and_save()
    print(f"Bundle written to {MODEL_PATH}")
    print(f"  data      : {bundle['data']}")
    print(f"  metrics   : {bundle['metrics']}")
    print(f"  threshold : {bundle['threshold']}  (cheapest on cross-validated training predictions)")
    print(f"  versions  : scikit-learn {bundle['sklearn_version']}, trained {bundle['trained']}")
