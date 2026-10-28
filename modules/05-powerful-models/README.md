# Module 05 · Powerful Models & Explainability

**ShopWave asks:** *"Make the churn model as accurate as possible. Then explain it to the CEO, in plain English, before the retention budget is signed off."*

**Time:** about 16 hours. **Prerequisites:** Module 03 (classification, AUC, thresholds) and Module 04 (cross-validation, pipelines, overfitting).

## Where you are

You have an honest churn model: a logistic-regression pipeline with a cross-validated AUC of about **0.74**. Retention likes it. Now two new requests arrive in the same week.

The head of data asks: *"Everyone on Kaggle uses XGBoost. Why aren't we? Squeeze out every point of accuracy."* And the CEO asks: *"If I am spending £50,000 on vouchers because a model told me to, I want to know **why** it picks the customers it picks."*

These pull in opposite directions. The most accurate models on business tables are **ensembles**, hundreds of trees voting or correcting each other, and no human can read 300 trees. This module gives you both halves: how ensembles work (by hand, on five rows), how to tune and use them properly, how to cope when the thing you are predicting is rare, and how to explain any model's decisions with SHAP.

It also teaches you something many courses skip: **the fancy model does not always win.** On ShopWave's churn table, which has few features and mostly straight-line relationships, logistic regression is very hard to beat. You will see exactly when ensembles earn their reputation, and when they do not.

## What you will be able to do afterwards

- Explain why many weak, different models beat one strong model, with the probability calculation to back it up.
- Draw a bootstrap sample by hand and explain bagging, random forests and out-of-bag scores.
- Run gradient boosting by hand for three rounds on five rows, and match scikit-learn to the penny.
- Use XGBoost and LightGBM with sensible settings, early stopping and missing values.
- Explain support vector machines, margins and kernels *(optional)*.
- Tune hyperparameters with grid, random and Bayesian search without cheating on the test set.
- Handle rare targets (fraud, churn in a loyal base) with class weights, resampling and the right metrics.
- Explain a model globally and for a single customer with feature importance, permutation importance and SHAP.
- Know what AutoML tools such as PyCaret automate, and what they hide *(optional)*.
- Choose a final model on accuracy, explainability and cost, and save it for production.

## Lessons

| # | Lesson | Time |
|---|---|---|
| 05.1 | [Ensembles: why many weak models beat one strong one](01-ensembles/) | 1 h |
| 05.2 | [Bagging and Random Forest](02-random-forest/) | 1.5 h |
| 05.3 | [Boosting: AdaBoost and gradient boosting by hand](03-boosting-by-hand/) | 2 h |
| 05.4 | [XGBoost, LightGBM and CatBoost in practice](04-xgboost-lightgbm/) | 1.5 h |
| 05.5 | [Support vector machines](05-svm/) *(optional)* | 1.5 h |
| 05.6 | [Hyperparameter tuning: grid, random, Bayesian (Optuna)](06-hyperparameter-tuning/) | 1.5 h |
| 05.7 | [Imbalanced data: class weights, resampling, SMOTE, the right metrics](07-imbalanced-data/) | 1.5 h |
| 05.8 | [Explaining models: feature importance, permutation importance, SHAP](08-explaining-models/) | 2 h |
| 05.9 | [AutoML in industry: PyCaret and what it automates](09-automl-pycaret/) *(optional)* | 1 h |
| 05.10 | [Choosing and saving the final model](10-choosing-and-saving/) | 0.5 h |
| 05.P | [Mini-project: a tuned, explained churn model](project/) | 2 h |

## The data in this module

- **By hand:** `data/shopwave_visits_spend_5.csv` (gradient boosting in 05.3) and small tables typed into the notebooks.
- **Real:** the ShopWave churn table from Module 03: 3,317 customers, features from before 2011-09-01, `churned` = no purchase September–November 2011 (43.6%). Same split as before: 2,487 training customers, 830 test customers. **The number to beat is the logistic-regression AUC of 0.741.**
- **Contrast:** a 20,000-order fraud table generated like the one in lesson 00.1, where the answer depends on combinations of thresholds. This is the kind of problem tree ensembles were built for.

Data loads through `load_retail()` in `data/download.py`, which caches the Excel file as Parquet so every notebook starts in about a second.

## The maths in this module

Three ideas, each computed by hand once:

- **Majority-vote probability**: if each voter is right with probability $p$, the chance that most of $n$ independent voters are right is a binomial sum. Five voters at 60% → 68%.
- **Residuals as targets**: gradient boosting fits the next tree to what the current model still gets wrong, then adds a fraction of it. You will see the error shrink from 6,400 to 534 in three rounds.
- **Margin**: an SVM picks the boundary with the widest empty corridor between classes. Width $= 2 / \lVert w \rVert$.
