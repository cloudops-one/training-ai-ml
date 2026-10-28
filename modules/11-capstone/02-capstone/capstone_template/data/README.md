# Data

**Do not commit data files.** They are re-created by `src/download_data.py`.

## Source

- Name: TODO
- URL: TODO
- Licence: TODO (must allow use in a public repository)
- Citation: TODO (if the source asks for one)

## Files

| File | Rows | Columns | Date range | Notes |
|---|---|---|---|---|
| `raw/TODO` | | | | as downloaded |
| `processed/table.parquet` | | | | the modelling table built in `notebooks/02_baseline.ipynb` |

## Data dictionary

| Column | Type | Meaning | Known at prediction time? |
|---|---|---|---|
| TODO | | | yes / no |

The last column is the leakage check from lesson 04.2. Any "no" must not be a feature.

## Cleaning decisions

- TODO: each rule you applied (rows removed, values fixed) and how many rows it affected.
