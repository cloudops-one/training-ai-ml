# Model card: TODO model name

> Fill in every section. A reviewer should be able to decide whether to trust and use your model from this page alone. See the [worked example](../example/model_card.md) for a completed card.

## Model details

- **Type:** TODO (algorithm, library, preprocessing steps)
- **File:** `models/TODO.joblib`
- **Trained on:** TODO (data and date range)
- **Version and date:** TODO

## Intended use

- **Use:** TODO (who uses it, for which decision)
- **Not for:** TODO (decisions it must not be used for)
- **Output:** TODO (a probability? a ranking? a number in which units?)

## Data

- Source and licence: TODO
- Features: TODO (table with meaning; confirm each is known at prediction time)
- Target: TODO (exact definition and time window)

## Evaluation

| Metric | Cross-validation | Test | Baseline |
|---|---|---|---|
| TODO | | | |

TODO: how the split was made and why.

## Limitations and risks

- TODO: where it fails (from error analysis)
- TODO: drift, seasonality, cold start
- TODO: fairness: does performance differ across groups? Which groups did you check?
- TODO: privacy: does the model or data contain personal information?

## Monitoring

- TODO: what you measure, how often, and what triggers a retrain or a rollback.
