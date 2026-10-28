# 02.P · Mini-project: predict customer spend

⏱ about 3 hours · Prerequisites: all of Module 02 · Notebook: [project.ipynb](project.ipynb) · Solution: [solution.ipynb](solution.ipynb)

---

## The brief

Marketing is back. They have the budget for 2,000 premium catalogues in September and want them to go to the customers who will spend the most in the following quarter. They do not want a chart; they want a ranked list and a one-paragraph answer to *"how much should we trust it?"*

You have the 2010–2011 transactions and the time-split recipe from lesson 01.9: features from everything a customer did before **1 September 2011**, target = what they spent from **1 September to 30 November 2011**. Build the best regression model you can with the tools of this module, prove it is better than the obvious baselines on customers it has never seen, and say which model to ship and why.

## Tasks

1. **Build the customer table.** Clean the transactions (`clean_retail` from 01.5), then one row per customer with at least: `orders`, `spend`, `products`, `avg_order`, `tenure_days`, `days_since_last`, and the target `spend_next_quarter`. State the number of customers and the share with a target of zero.
2. **Hold out a test set** of 20% of customers *before* looking at any model (`train_test_split`, `random_state=42`). Everything below is fitted on the 80% and scored on the 20%.
3. **Baselines.** Predict the mean; predict the median. Report MAE, RMSE and R² on the test set. These are the bar.
4. **One feature.** A line on `spend` alone. Did it beat both baselines on all three metrics?
5. **All features.** Multiple linear regression on the six features. Print the coefficients and explain, in one sentence each, two of them, including any whose sign surprises you.
6. **A curve.** Polynomial degree 2 on `spend`. Does bending help on the test set?
7. **A tree.** `DecisionTreeRegressor` with `max_depth` in {2, 4, 8, None}. Tabulate train R² and test R² side by side and name the depth where overfitting starts.
8. **Leaderboard.** One table, every model, test MAE / RMSE / R². One sentence on which metric marketing should care about and why.
9. **Residual check.** For the winning model, plot residuals vs predictions. Who does it get most wrong? (Look up the three biggest misses in the customer table.)
10. **The ranked list.** Score every customer with the winning model, sort, and show the top 20. How many of them are in the test set? (If none are, say what that means for how much you can trust the ranking.)
11. **Recommendation.** A markdown cell addressed to the head of marketing: which model, its typical error in pounds, one caveat about wholesalers, and one thing you would do next (Module 04 will give you the tools).

## Rubric (self-assess, 20 points)

| Criterion | Points |
|---|---|
| Customer table is built with features strictly before the cut-off and target strictly after; no leaky columns | 4 |
| Test set split once, before modelling, and used for every reported score | 3 |
| Both baselines reported and every model compared against them | 3 |
| Leaderboard complete and correct; train vs test R² shown for trees | 3 |
| Coefficients interpreted correctly (including the "holding others fixed" trap) | 2 |
| Residual plot made and the worst misses identified and explained | 2 |
| Recommendation is specific, numeric, honest about limits, and readable by a non-technical person | 3 |

## What "good" looks like

A reader should finish your notebook knowing: the baseline MAE, the winning model's MAE, roughly how much better that is, which customers the model cannot handle, and what you would try next. If the winning model is the simplest one, say so; that is a finding, not a failure.

## Stretch

- Fit the models on `log1p(spend_next_quarter)` and convert predictions back with `expm1`. Does MAE on the original scale improve? (Watch for absurd predictions on the biggest customers; clip them and explain why.)
- Drop customers with past spend above the 99th percentile from the *training* set only. Does the model get better or worse for everyone else?
