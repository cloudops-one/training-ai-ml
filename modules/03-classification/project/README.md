# 03.P · Mini-project: Titanic survival, then ShopWave churn

⏱ about 3.5 hours · Prerequisites: all of Module 03 · Notebooks: [project.ipynb](project.ipynb) (your work), [solution.ipynb](solution.ipynb) (check afterwards)

---

## The brief

Two parts. Part A is a warm-up on a dataset everyone in ML knows; Part B is the real job.

### Part A: Titanic (1 hour)

The Titanic passenger list is the "hello world" of classification: 891 passengers, did they survive? It is small, has a few missing values, mixes numbers and categories, and has a strong, interpretable signal. It is also the dataset behind Kaggle's long-running [Titanic competition](https://www.kaggle.com/c/titanic), which is where most people's first public ML notebook lives. Here we load it with `seaborn.load_dataset("titanic")`, no account needed.

Tasks:

1. Load, inspect, and handle missing `age` (median) and `embarked` (most common).
2. Encode `sex` and `embarked` as numbers; choose features: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`.
3. Stratified 75/25 split.
4. Baseline ("nobody survives" and "all women survive"), then KNN, decision tree, logistic regression.
5. Report confusion matrix, precision, recall, F1 and ROC-AUC for each on the test set, in one table.
6. One sentence on what the best model learned (look at the tree's first split or the logistic coefficients).

### Part B: ShopWave churn (2.5 hours)

Marketing will send a £5 retention voucher to customers predicted to churn. A churned customer costs ShopWave about £60. Build the model that decides who gets a voucher.

Use the customer table from 01.9: features computed before 1 September 2011, target `churned` = no purchase from 1 September to 30 November 2011.

Tasks:

1. Build the table; print churn rate and the majority baseline accuracy.
2. Stratified 75/25 split. (Module 04 will add cross-validation; for now, one split.)
3. Four candidates: baseline (recency rule `days_since_last > 90`), KNN (scaled), decision tree (`max_depth=4`), logistic regression (scaled).
4. One comparison table: accuracy, precision, recall, F1, ROC-AUC on the test set.
5. For the best model by AUC: plot ROC and precision–recall curves.
6. Choose the voucher threshold by minimising cost = £5 × flagged + £60 × missed. Report the chosen threshold, the cost, and the costs of three alternatives: default 0.5, voucher to everyone, voucher to nobody.
7. Write **five insights** for marketing in plain English (no metric names without a translation).

## Deliverable

A completed `project.ipynb` that runs top to bottom, ending with the comparison tables, the chosen threshold, and the five insights in a markdown cell.

## Rubric (20 points)

| Criterion | Points |
|---|---|
| Data prepared correctly: missing values handled, categories encoded, stratified split, no target leakage | 4 |
| Baselines reported before any model, in both parts | 2 |
| All four models trained and evaluated on the **test** set only | 3 |
| Comparison table with all five metrics, correct positive class | 3 |
| ROC and precision–recall curves for the best model, labelled | 2 |
| Threshold chosen by cost, with the three alternatives for comparison | 3 |
| Five insights: correct, specific, in business language | 3 |

Marks 16+: ready for Module 04. Below 12: redo 03.6 and 03.7 first.

## Hints

- Scale features for KNN and logistic regression (`StandardScaler` inside `make_pipeline`); trees do not need it. Module 04 explains why.
- `predict_proba(X)[:, 1]` gives P(churn); thresholds apply to that, not to `predict`.
- Keep the positive class consistent: churn = 1, survived = 1.
- Scikit-learn's confusion matrix is `[[TN, FP], [FN, TP]]`.

## Credits

Titanic data: Kaggle, *Titanic: Machine Learning from Disaster*, via the seaborn dataset repository. ShopWave data: Online Retail II (UCI, CC BY 4.0).
