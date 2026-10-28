# Model card: ShopWave high-value customer ranker

## Model details

- **Type:** logistic regression (scikit-learn) on nine features, each log-transformed (`log1p`) and standardised, in a single `Pipeline`.
- **File:** `models/high_value_logistic_auc0.815.joblib` (a dictionary holding the pipeline, the feature list, the £500 threshold and the training cut-off).
- **Trained on:** customers as of 1 June 2011, target from June to August 2011.
- **Owner:** the ShopWave data person (that is you, in this course).

## Intended use

- **Use:** rank registered customers once a quarter; send the premium catalogue to the top N.
- **Not for:** credit, pricing, or any decision about an individual customer beyond whether they receive a catalogue. Not for guest checkouts (no history).
- **Output:** a score between 0 and 1. Treat it as a **ranking**, not a calibrated probability (see Limitations).

## Data

Online Retail II, UCI Machine Learning Repository (CC BY 4.0), sheet "Year 2010-2011". Cleaned by removing cancelled invoices, returns, zero-price lines, rows without a customer ID and exact duplicates.

| Feature | Meaning |
|---|---|
| `orders` | distinct invoices before the cut-off |
| `spend` | total revenue before the cut-off |
| `products` | distinct stock codes bought |
| `tenure_days` | days since the first order |
| `days_since_last` | days since the most recent order |
| `avg_order` | spend ÷ orders |
| `spend_90d`, `orders_90d` | spend and orders in the 90 days before the cut-off |
| `is_uk` | 1 if the customer's country is the United Kingdom |

**Target:** `high_value = 1` if the customer spends £500 or more in the three months after the cut-off.

## Evaluation

Time-based split: trained at cut-off 1 June 2011 (2,718 customers, 24.4% positive), tested at 1 September 2011 (3,317 customers, 32.0% positive).

| Metric | Cross-validation (train) | Test |
|---|---|---|
| ROC-AUC | 0.848 ± 0.027 | 0.815 |
| Average precision | | 0.716 |
| Precision in top 500 | | 0.810 (405 of 500) |
| Current rule, precision in top 500 | | 0.746 (373 of 500) |

## Limitations and risks

- **Seasonal shift.** The test quarter (September to November) has a higher positive rate than the training quarter. Scores rank well but underestimate probabilities in autumn.
- **Single backtest.** Results come from one test quarter.
- **Cold start.** Customers with one order and no recent activity are ranked low even when they later spend a lot; 656 high-value customers were missed in the test quarter.
- **Country is nearly irrelevant** to the model (permutation importance 0.000), so the model is not treating customers differently by country in practice. This should be rechecked after every retrain.

## Monitoring

- Quarterly: precision in the top N against actual spend; alert if it drops below the recent-spend rule.
- Each scoring run: compare feature distributions with the training table.
- Retrain quarterly on the most recent complete quarter; replace only if it beats the current model on the same holdout.
