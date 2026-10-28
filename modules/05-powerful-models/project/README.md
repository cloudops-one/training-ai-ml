# 05.P · Mini-project: a tuned, explained churn model

⏱ about 2 hours · Prerequisites: all of Module 05 · Notebooks: [project.ipynb](project.ipynb) (your work), [solution.ipynb](solution.ipynb) (check afterwards)

---

## The brief

The CEO has approved a monthly win-back campaign. Each month, ShopWave will send a £5 voucher to customers the model flags as likely to churn. A churned customer costs about £60 in lost margin.

Before the campaign launches, the CEO wants three things:

1. **The best model you can honestly build**, compared against the logistic-regression baseline from Module 03 (test AUC 0.741).
2. **An explanation** of what drives churn overall, and of one individual customer's score, in language the retention team understands.
3. **A deployable file and a one-page summary**: which model, why, who gets a voucher, what it costs, and what the limits are.

Use the churn table from Module 03: 3,317 customers, features computed from orders before 1 September 2011, target `churned` = no order from 1 September to 30 November 2011. The setup cell in the notebook builds it and splits it 75/25 (2,487 training, 830 test).

## Tasks

1. **Baseline.** Scaled logistic regression: 5-fold CV AUC (keep the per-fold scores) and test AUC.
2. **Tune a boosted model.** LightGBM with `RandomizedSearchCV` (30 iterations) over learning rate, leaves, minimum leaf size, row and column sampling. Training data only.
3. **Is the difference real?** Per-fold differences against the baseline on the same folds; mean and std. Then open the test set once.
4. **Threshold by voucher cost.** Choose each model's cheapest threshold on cross-validated training predictions; report test costs against "voucher to everyone" and "voucher to nobody".
5. **Explain globally.** SHAP beeswarm and mean |SHAP|; two sentences on what drives churn.
6. **Explain one customer.** A customer near the 75th percentile of risk: waterfall and a three-line plain-English explanation.
7. **Choose, save, document.** One-standard-error rule; save a versioned bundle to `models/`; reload and score a new customer; write the one-page CEO summary.

## Rules

- The test set is opened only in tasks 3, 4 and 7, after every choice has been made on training data.
- Every number in your summary must come from an executed cell.
- If the boosted model does not beat the baseline by more than the noise, say so. That is a valid, valuable finding.

## Rubric (20 points)

| Area | Points | Full marks when… |
|---|---|---|
| Baseline and tuning | 4 | Both models evaluated on identical CV folds; search on training data only; sensible ranges |
| Honest comparison | 4 | Fold-by-fold differences reported; conclusion matches their size relative to the noise |
| Threshold and cost | 4 | Threshold chosen on CV training predictions; test costs compared with both trivial policies; break-even (£5/£60) recognised |
| Explanation | 4 | Beeswarm read correctly; one customer explained in plain English; no causal over-claims |
| Deployable output | 4 | Bundle contains pipeline, features, threshold, metrics, date and versions; reload works; CEO summary is accurate and readable |

## Stretch

- Add XGBoost with `max_depth=1` (05.4 found it tied the baseline at CV 0.741). Does it change your choice?
- Recompute the threshold and costs for a £20 voucher. How many customers get one now, and how much more does the model matter?
- Engineer one new feature (for example, average days between orders) and see whether it moves the CV AUC by more than the tuning did.

## What the solution found

So you can check your direction, not copy numbers:

- Logistic regression: CV AUC 0.740 ± 0.029, test 0.741.
- Tuned LightGBM: CV AUC 0.743; it beats the baseline on all five folds, but by only +0.003 on average; test 0.742.
- Cheapest thresholds 0.07 (logistic) and 0.09 (LightGBM), close to the break-even 5/60 = 0.083. Test costs £3,945 and £3,905, against £4,150 for vouchers to everyone and £21,720 for nobody.
- Main drivers: few orders, a long gap since the last order, low spend; tenure barely matters.
- Chosen model: logistic regression (tied by the one-standard-error rule; simpler to explain and maintain). Choosing LightGBM for its consistent small edge is also defensible if you say why.
