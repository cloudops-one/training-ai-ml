# Module 04 · Making Models Trustworthy

**ShopWave asks:** *"The churn model scored 90% in the notebook and 60% in its first month live. What went wrong?"*

**Time:** about 14 hours. **Prerequisites:** Modules 02 and 03 (fitting regression and classification models, error metrics, the confusion matrix).

## Where you are

A colleague built a churn model before you joined. In their notebook it was almost perfect. Retention trusted it, sent vouchers to the customers it flagged, and a month later the results were barely better than guessing. Nobody can explain the gap.

This is the most common way machine learning fails in real companies, and it is almost never because the algorithm was wrong. It is because the model was **measured dishonestly**: it was scored on data it had already seen, or it was given information it would never have in real life, or the data was prepared in a way that quietly passed answers from the test set into training.

Modules 02 and 03 taught you to *build* models. This module teaches you to build models whose scores you can **believe**: how to split data, how to spot and prevent leakage, how to prepare messy columns (missing values, outliers, categories, different scales) without cheating, how to get a stable score with cross-validation, and how to package everything into a pipeline so the same steps run identically in the notebook and in production. You finish by rebuilding the churn model so that the number you report is the number retention will see.

## What you will be able to do afterwards

- Split data into train, validation and test sets, and choose between random, stratified and time-based splits.
- Recognise the three kinds of data leakage and catch them before they reach a manager's inbox.
- Handle missing values and outliers, and decide between dropping, filling, capping and transforming.
- Turn categories into numbers with one-hot, ordinal and target encoding, and know which models each one suits.
- Scale features, and know which models need it and which do not.
- Engineer useful features from dates, ratios and customer behaviour (RFM).
- Use k-fold cross-validation to get a score you can trust, with an error bar.
- Wrap preprocessing and model in one scikit-learn `Pipeline` so leakage becomes impossible by construction.
- Diagnose overfitting and underfitting with learning curves, and control it with regularisation (Ridge and Lasso).

## Lessons

| # | Lesson | Time |
|---|---|---|
| 04.1 | [Train, validation, test: why you must hide data from yourself](01-train-validation-test/) | 1 h |
| 04.2 | [Data leakage: the mistake that makes every model look great](02-data-leakage/) | 1 h |
| 04.3 | [Missing values and outliers done properly](03-missing-and-outliers/) | 1.5 h |
| 04.4 | [Encoding categories: one-hot, ordinal, target encoding](04-encoding-categories/) | 1.5 h |
| 04.5 | [Scaling features: who needs it and who does not](05-scaling/) | 1 h |
| 04.6 | [Feature engineering: dates, ratios, bins, RFM](06-feature-engineering/) | 1.5 h |
| 04.7 | [Cross-validation: a fairer score](07-cross-validation/) | 1.5 h |
| 04.8 | [Pipelines and ColumnTransformer: leak-proof preprocessing](08-pipelines/) | 1.5 h |
| 04.9 | [Overfitting, underfitting and the bias–variance trade-off](09-bias-variance/) | 1.5 h |
| 04.10 | [Regularisation: Ridge and Lasso](10-regularisation/) | 1 h |
| 04.P | [Mini-project: rebuild churn as a leak-free pipeline](project/) | 1 h |

## The data in this module

- **By hand:** tiny tables of 5 to 10 values written into each lesson, plus `data/shopwave_customers_small.csv` (20 customers).
- **For real:** the ShopWave churn table from Modules 01 and 03: one row per customer, features from orders before 1 September 2011, `churned` = 1 if the customer placed no order in September–November 2011. 3,317 customers, 43.6% churned. Lessons in this module build it with a small `customer_table(cutoff)` function so the cut-off date can be moved, which is exactly what an honest time-based test needs.

## The maths in this module

Very little that is new. The **standard error** of an accuracy, $\sqrt{p(1-p)/n}$, tells you how much a test score can wobble by luck. The **z-score** $(x-\mu)/\sigma$ appears twice, for outliers and for scaling. **Quartiles** come back from 01.8 to define outlier fences. Everything else in this module is about *discipline*: which rows each calculation is allowed to see.
