"""Score a CSV of customers with the saved churn model (batch scoring).

    python score_customers.py customers.csv                 # writes customers_scored.csv
    python score_customers.py customers.csv -o scores.csv

The CSV needs the columns orders, spend, products, tenure_days, days_since_last.
Any other columns (customer_id, for example) are copied to the output unchanged.
Rows with missing or negative values are reported and skipped, not guessed.
"""
import argparse
import sys
from pathlib import Path


def find_lib() -> Path:
    for folder in Path(__file__).resolve().parents:
        if (folder / "lib" / "churn_model.py").exists():
            return folder / "lib"
    sys.exit("Cannot find lib/churn_model.py. Run this from inside the course folder.")


sys.path.insert(0, str(find_lib()))
import pandas as pd  # noqa: E402
from churn_model import load_bundle, score  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score customers with the ShopWave churn model.")
    parser.add_argument("input", help="CSV file with one row per customer")
    parser.add_argument("-o", "--output", help="where to write the scores (default: <input>_scored.csv)")
    args = parser.parse_args(argv)

    bundle = load_bundle(train_if_missing=False)
    customers = pd.read_csv(args.input)
    missing = [f for f in bundle["features"] if f not in customers.columns]
    if missing:
        print(f"ERROR: {args.input} has no column(s) {missing}", file=sys.stderr)
        return 1

    values = customers[bundle["features"]]
    bad = values.isna().any(axis=1) | (values < 0).any(axis=1)
    if bad.any():
        print(f"Skipping {bad.sum()} row(s) with missing or negative values: rows {list(bad[bad].index[:10])}")
    good = customers[~bad]
    scored = good.join(score(bundle, good))

    output = Path(args.output) if args.output else Path(args.input).with_name(Path(args.input).stem + "_scored.csv")
    scored.to_csv(output, index=False)
    print(f"Scored {len(scored):,} customers with the model trained {bundle['trained']} "
          f"(threshold {bundle['threshold']}): {int(scored['voucher'].sum()):,} get a voucher.")
    print(f"Written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
