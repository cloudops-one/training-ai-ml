# Project title: what it predicts, for whom

> Copy this whole `capstone_template/` folder into a new repository of your own, then replace every line that starts with ">" or "TODO". Delete this note when you are done.

**One-sentence summary.** TODO: *"A model that ranks [units] by [target] so that [user] can [decision]; it beats [baseline] by [result] on [held-out data]."*

## The problem

TODO: two or three sentences. Who has the problem, what they do today, why it matters (in money, time or risk if you can).

## Result

| | Metric | Value |
|---|---|---|
| Baseline: TODO | TODO | TODO |
| **Chosen model: TODO** | TODO | **TODO** |

TODO: one sentence on what the improvement means in the user's terms ("32 more high-value customers reached for the same budget").

## Data

- Source and licence: TODO (link)
- Size and date range: TODO
- How to get it: `python src/download_data.py` (TODO: write this script; never commit the data itself)
- What one row of the modelling table means: TODO

## Approach

1. TODO: split strategy and why (time-based? stratified?)
2. TODO: baseline(s)
3. TODO: models compared, with cross-validation scores
4. TODO: how the final model was chosen
5. TODO: how it was explained

## Limitations

- TODO: what the model cannot do, where it fails (from your error analysis), what data you wish you had.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/download_data.py
jupyter lab                        # then run notebooks/01_eda, 02_baseline, 03_models in order
python src/train.py                # retrains and saves models/model.joblib
```

## Repository layout

```
data/            README only; raw data is downloaded, not committed
notebooks/       01_eda, 02_baseline, 03_models
src/             download_data.py, train.py (and app.py if you deploy a demo)
models/          saved models (git-ignored)
model_card.md    what the model is for, how it was tested, its limits
```

## Next steps

TODO: two or three things you would do with another week.
