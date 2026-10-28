# Worked example: who gets the premium catalogue?

A complete, compact capstone on ShopWave's real transactions, written to show what a finished project looks like. Compare it with yours *after* you have drafted your own; reading it first will make your project a copy of this one.

- Notebook: [example.ipynb](example.ipynb) (runs in under a minute after the data is cached)
- Model card: [model_card.md](model_card.md)

## The problem

Marketing posts a premium catalogue to 500 customers each quarter, at about £3 each. Today they send it to whoever spent the most in the last 90 days. They want it to go to the customers who will spend **£500 or more in the next three months**.

## Data

Online Retail II (UCI, CC BY 4.0), the 2010–2011 sheet: 541,910 transaction lines, 392,693 after removing cancellations, returns, free lines, guest checkouts and duplicates.

One row per registered customer at a cut-off date. Nine features use only orders before the cut-off: orders, spend, distinct products, tenure, days since last order, average order value, spend and orders in the last 90 days, UK or not. The target uses only the three months after it.

| Table | Cut-off | Customers | High-value rate |
|---|---|---|---|
| Training | 1 June 2011 | 2,718 | 24.4% |
| Test | 1 September 2011 | 3,317 | 32.0% |

The split is by time, so the test quarter is genuinely in the model's future.

## Approach

1. Baselines: "nobody is high-value" (68.0% accuracy, useless) and marketing's current rule, ranking by spend in the last 90 days.
2. Two models compared with 5-fold cross-validation on the training table only: logistic regression on log-transformed, scaled features, and LightGBM.
3. They tied (CV AUC 0.848 ± 0.027 and 0.847 ± 0.026), so the simpler logistic regression was chosen before the test set was opened.
4. The test set was used once.

## Results

| On the test quarter | Test AUC | Average precision | Precision in top 500 |
|---|---|---|---|
| **Logistic regression (chosen)** | **0.815** | **0.716** | **0.810** |
| LightGBM | 0.808 | 0.709 | 0.818 |
| Current rule: rank by recent spend | 0.744 | 0.630 | 0.746 |

**For the same 500 catalogues (£1,500), the model reaches 405 high-value customers instead of 373: 32 more.**

## What drives the prediction

How many orders a customer has placed and how large their average order is. Recent spend adds little once those are known.

## Where it fails

- **95 of the 500 catalogues** go to customers who look excellent on paper and then spend almost nothing next quarter (median £3).
- **656 high-value customers are missed.** They have fewer orders (median 2) and lower recent spend (median £289) than the customers the model picks. About a third bought nothing in the previous 90 days, and about a third had ordered only once. Catching them needs new information, such as browsing or email engagement, not a different algorithm.

## Limitations

- The high-value rate in the autumn test quarter is higher than in the summer training quarter. Use the model to **rank** customers; its probabilities are too low for autumn.
- One test quarter only. A second backtest (for example cut-off 1 March, scored on March to May) would make the result more convincing.
- Registered customers only: guest checkouts cannot receive a catalogue anyway.

## What I would do next

1. Backtest on two more quarters.
2. Recalibrate probabilities on the same quarter of the previous year (the 2009–2010 sheet).
3. Ask marketing whether email engagement data exists.
4. Deploy as a quarterly batch job that writes a ranked list, monitored as described in the notebook.
