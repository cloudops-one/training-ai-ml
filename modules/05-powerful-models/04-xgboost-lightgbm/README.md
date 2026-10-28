# 05.4 · XGBoost, LightGBM and CatBoost in practice

⏱ about 1.5 hours · Prerequisites: 05.3 (gradient boosting), 04.1 (validation sets), 04.3 (missing values) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The head of data's message: *"Every Kaggle winner on tabular data uses XGBoost or LightGBM. Why are we using logistic regression?"*

It is a fair question. Since about 2015, gradient-boosted trees from three libraries, **XGBoost**, **LightGBM** and **CatBoost**, have won the large majority of machine-learning competitions on business-style tables (rows of customers, transactions, claims), and they run in production at almost every large company. You need to know how to use them well.

You also need to know when they are *not* worth it. This lesson tests both on ShopWave data and gives you an honest answer for each.

## 2. The idea

XGBoost, LightGBM and CatBoost are all **gradient boosting** exactly as you built by hand in 05.3: start from a constant, fit small trees to the gradient one at a time, add them with a learning rate. What they add is **engineering**:

| | what it adds |
|---|---|
| **Speed** | Bin each feature into ~256 buckets once (histograms), so finding the best split is fast; use all CPU cores; LightGBM grows trees leaf-by-leaf, CatBoost uses symmetric trees. |
| **Regularisation** | Penalties on leaf values (L1/L2), minimum examples per leaf, row and column subsampling for every tree. |
| **Missing values** | Learned default direction at every split, no imputation needed. |
| **Early stopping** | Watch a validation set and stop adding trees when it stops improving. |
| **Categoricals** | LightGBM and especially CatBoost can use text categories directly. |

So the big ideas are not new. The libraries make boosting fast, robust and convenient enough to be the default for tabular data.

## 3. Definition

The settings you will actually touch (XGBoost name / LightGBM name):

| setting | meaning | typical start |
|---|---|---|
| `n_estimators` | maximum number of trees | 1000–5000 **with early stopping** |
| `learning_rate` | shrinkage $\nu$ per tree (05.3) | 0.01–0.1 |
| `max_depth` / `num_leaves` | size of each tree; controls how many features can interact | depth 3–6 / 15–63 leaves |
| `min_child_weight` / `min_child_samples` | minimum data per leaf | 1–50 |
| `subsample` (+ `subsample_freq=1` in LightGBM) | fraction of rows each tree sees | 0.7–0.9 |
| `colsample_bytree` | fraction of features each tree sees | 0.7–0.9 |
| `reg_lambda`, `reg_alpha` | L2 / L1 penalty on leaf values | 0–10 |
| `early_stopping_rounds` | stop after this many rounds with no validation improvement | 50 |

**Early stopping**: train on part of the training data, monitor a **validation** set (never the test set), and keep the number of trees at which validation score was best.

## 4. Formula

XGBoost improves on plain gradient boosting by adding a penalty to every tree it builds:

$$\text{objective} = \sum_i \text{loss}(y_i, \hat{y}_i) \;+\; \sum_{\text{trees}} \Big(\gamma \, T + \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^2\Big)$$

- $T$: number of leaves in a tree; $\gamma$ (`gamma`): a cost per leaf, so a split must gain at least $\gamma$ to be worth making.
- $w_j$: the value in leaf $j$; $\lambda$ (`reg_lambda`): shrinks leaf values towards zero, like Ridge regression (04.10) for trees.

The first part is the same loss you minimise everywhere. The second part says *prefer small trees with modest leaf values*. You rarely compute this by hand, but it explains why XGBoost overfits less than the textbook algorithm.

## 5. By hand

The decision you make most often with these libraries is **how many trees**. Here is the validation record from the notebook for XGBoost with learning rate 0.05 on the churn data, `early_stopping_rounds=50`:

| trees | 10 | 20 | 30 | **34** | 50 | 70 | 84 |
|---|---|---|---|---|---|---|---|
| validation AUC | 0.693 | 0.694 | 0.694 | **0.696** | 0.693 | 0.691 | 0.687 |

Walk through it as the library does. After round 34 the best score so far is 0.696. Rounds 35, 36, … 84 never beat it. After 50 rounds without improvement (34 + 50 = 84), training stops, and the model keeps only its first **34** trees. *Best round + patience = where it stops.* That is all early stopping is.

Notice also how **flat** the curve is: 0.693 after 10 trees, 0.696 at best. On this data the boosters find most of what there is to find almost immediately, which foreshadows the results below.

## 6. In code

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=2000, learning_rate=0.05, max_depth=3,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=50, eval_metric="auc",
)
model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
model.best_iteration           # trees actually used
```

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=8, verbose=-1)
model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric="auc",
          callbacks=[lgb.early_stopping(50, verbose=False)])
```

Both follow the scikit-learn API (`fit`, `predict_proba`), so they work in pipelines, cross-validation and grid search.

## 7. On real data

**ShopWave churn** (5-fold CV on 2,487 customers, then 830 test customers):

| model | CV AUC | test AUC |
|---|---|---|
| **logistic regression** | **0.740** | **0.741** |
| random forest | 0.736 | 0.731 |
| XGBoost | 0.723 | 0.730 |
| LightGBM | 0.721 | 0.727 |

The boosting libraries **lose** to logistic regression here. With early stopping, XGBoost stops at 34 trees and reaches test AUC 0.737, closer but still not ahead. In the exercises, XGBoost with depth-1 trees (stumps) scores CV AUC 0.741 and depth 10 only 0.701. Everything points the same way: the churn signal in these five features is **additive and nearly linear** (more orders → less churn; more days away → more churn), and 2,487 noisy examples leave little for trees to find.

**Missing values.** With 20% of `spend` blanked out, logistic regression refuses to train (`Input X contains NaN`); XGBoost trains as is and scores test AUC 0.737.

**Fraud orders** (20,000 orders built like lesson 00.1; fraud = cliffs and "and" rules):

| model | test AUC | fit time |
|---|---|---|
| logistic regression | 0.815 | instant |
| random forest | 0.885 | 0.7 s |
| XGBoost | 0.873 | 0.1 s |
| LightGBM | 0.874 | 0.2 s |

Here every tree ensemble is far ahead. Logistic regression can only draw a gentle slope ("risk rises with total"); the truth is a **cliff** ("an account under 10 days old"), and trees draw cliffs naturally. (Times are from one laptop; yours will differ.)

The rule to take away: **always include a logistic regression in the comparison.** If boosting cannot beat it, your features are the bottleneck, not your model, and Module 04's feature engineering is the next step, not a bigger model.

**CatBoost** is the third library. Its strength is categorical features (pass `country` as text; it encodes it internally without leaking the target). It is optional in this course: `pip install catboost` if you want to try it.

## 8. Where it breaks

- **Small or nearly-linear data.** As on churn: more flexible ≠ better. Simple models win when there is little structure or little data.
- **Tuning a test set into the model.** Early stopping or tuning on the test set gives a flattering, unrepeatable score. Use a validation split or cross-validation.
- **No extrapolation.** Like all trees, predictions are flat beyond the training range. Forecasting a trend (Module 07) needs care.
- **Many interacting knobs.** Learning rate, number of trees and tree size trade off against each other. Tune them together (05.6), starting from the "typical start" column above.
- **Explainability.** Hundreds of trees cannot be read directly. Use SHAP (05.8).
- **Version drift.** These libraries change parameter names and defaults between versions; pin versions in production.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Early stopping for LightGBM on churn. At which round does it stop, and what is the test AUC?
2. XGBoost `max_depth` = 1, 2, 3, 6, 10 on churn by CV AUC. What does the result say about interactions in the churn data?
3. The same depth comparison on fraud. Do stumps fall behind?
4. Train logistic regression and XGBoost on 500, 2,000, 5,000 and 15,000 fraud orders and plot test AUC against training size.

**Quiz**

1. Why do gradient-boosting libraries dominate competitions on tabular data?
2. Which two settings trade off directly against each other, and how do you set them in practice?
3. Early stopping watches which data set, and why not the test set?
4. How do XGBoost and LightGBM handle a missing value at a split?
5. Why does XGBoost beat logistic regression on fraud but not on churn?

## 10. Recap

- XGBoost, LightGBM and CatBoost are gradient boosting (05.3) plus speed, regularisation, missing-value handling and early stopping.
- Set `n_estimators` high, `learning_rate` small, and let **early stopping on a validation set** choose the number of trees.
- On ShopWave churn they **do not** beat logistic regression (0.72–0.73 vs 0.74): the signal is additive and nearly linear.
- On threshold-shaped fraud data they beat it by a wide margin (0.87–0.89 vs 0.82).
- Always benchmark against a simple model; if it wins, improve the features, not the model.

**Next:** [05.5 Support vector machines](../05-svm/) *(optional)*, or go straight to [05.6 Hyperparameter tuning](../06-hyperparameter-tuning/).
