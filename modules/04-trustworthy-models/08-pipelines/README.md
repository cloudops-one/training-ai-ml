# 04.8 · Pipelines and ColumnTransformer: leak-proof preprocessing

⏱ about 1.5 hours · Prerequisites: 04.3–04.5 (missing values, encoding, scaling), 04.7 (cross-validation) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Your churn notebook now has a dozen preprocessing steps scattered across cells: fill missing spend, log the skewed columns, scale, one-hot encode the country. Three things go wrong:

1. **Cross-validation is quietly dishonest.** You scaled and selected features on the whole training set, then cross-validated. Each held-out fold has already influenced its own preprocessing.
2. **Production breaks.** The engineering team gets your model file and asks: *"What do we do with the raw customer row before calling `predict`?"* The answer is "run cells 4, 7, 9 and 12 in that order with the same medians and means", which nobody can reproduce.
3. **A new country arrives** (a first customer from Kenya) and the one-hot encoder crashes.

You need preprocessing and model packaged as **one object** that is fitted together, cross-validated together and shipped together.

## 2. The idea

A coffee machine does not hand you beans, a grinder, a kettle and instructions. You press one button and every step runs, in order, with the settings it was calibrated with.

A **pipeline** is that machine for ML. You chain the steps once:

```
raw customer rows
      │
      ▼
┌─────────────────────── ColumnTransformer ───────────────────────┐
│  numeric columns ─► impute median ─► log(1+x) ─► scale          │
│  country         ─► impute mode   ─► one-hot (rare → "other")   │
└─────────────────────────────────────────────────────────────────┘
      │  10 clean numeric columns
      ▼
LogisticRegression ─► probability of churn
```

Then `pipe.fit(X_train, y_train)` fits every step on the training rows, in order, and `pipe.predict(new_rows)` replays exactly those fitted steps. Inside `cross_val_score`, the whole pipeline is refitted in every fold, so **no held-out row ever influences preprocessing**.

## 3. Definition

- **Transformer**: a step with `fit` (learn something from data, e.g. a median) and `transform` (apply it). Imputers, scalers, encoders and feature selectors are transformers.
- **Estimator**: the model at the end, with `fit` and `predict`.
- **Pipeline**: a sequence of transformers ending in an estimator, behaving as one estimator.
- **ColumnTransformer**: applies different transformers to different columns and joins the results side by side.
- **Fit inside the fold**: the rule that every step which learns from data is fitted only on the training part of each fold.

## 4. Formula

Standard scaling, the step this lesson uses to show the leak:

$$z = \frac{x - \mu}{\sigma}$$

The question is *which rows* $\mu$ (mean) and $\sigma$ (standard deviation) are computed from. **Honest**: training rows only. **Leaky**: training and test rows together.

## 5. By hand

Training values $[10, 20, 30]$; one test value $100$ (a customer far outside anything seen in training). Population standard deviation (divide by $n$).

**Fit on training only.**

$\mu = (10 + 20 + 30)/3 = 20$

$\sigma = \sqrt{\big((-10)^2 + 0^2 + 10^2\big)/3} = \sqrt{200/3} = \sqrt{66.7} = 8.165$

Test $z = (100 - 20)/8.165 = $ **9.80**

**Fit on all four values.**

$\mu = (10 + 20 + 30 + 100)/4 = 40$

$\sigma = \sqrt{\big((-30)^2 + (-20)^2 + (-10)^2 + 60^2\big)/4} = \sqrt{5000/4} = \sqrt{1250} = 35.36$

Test $z = (100 - 40)/35.36 = $ **1.70**

Honestly scaled, the test customer is a shocking 9.8 standard deviations out, the kind of input that should make you distrust the prediction. Leakily scaled, they look ordinary. The test row changed the ruler it was measured with.

## 6. In code

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression

numeric = make_pipeline(SimpleImputer(strategy="median"),
                        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                        StandardScaler())
categorical = make_pipeline(SimpleImputer(strategy="most_frequent"),
                            OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20))
prep = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, ["country"])])
pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))])

cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")   # honest CV
pipe.fit(X_train, y_train)
pipe.predict_proba(raw_new_rows)                                   # raw in, probability out
```

`pipe.set_output(transform="pandas")` makes each step return a DataFrame with named columns, so you can inspect exactly what the model receives.

## 7. On real data

The notebook demonstrates two leaks and then builds the ShopWave pipeline.

| Experiment | Leaky | Inside a pipeline |
|---|---|---|
| Scale the 2,487 training customers, then 5-fold CV (accuracy) | 0.6856 | 0.6856 |
| Pick 20 of 5,000 **pure-noise** columns, then 5-fold CV (accuracy) | **0.795** | 0.550 |

The first leak changes nothing measurable: a mean and standard deviation over 2,000 rows barely move when one fold is removed. The second invents an 80%-accurate model out of random numbers, because selecting features looked at the labels of every row, including the ones later used for testing. You cannot always tell in advance which leaks will matter, so the habit is: **everything that learns goes in the pipeline**.

The full churn pipeline (6 numeric columns logged and scaled, `country` one-hot encoded with 30 rare countries lumped together) scores **CV AUC 0.738 ± 0.027** and **test AUC 0.746**. It accepts raw rows with a missing `spend` and a never-seen country ("Kenya") without crashing: a one-order new customer gets 72% churn probability, a 12-order active customer 3%.

## 8. Where it breaks

- **Steps outside the pipeline still leak.** A pipeline only protects what is inside it. Feature engineering that uses labels or future data (04.2, 04.6 exercise 4) must be fixed at the source.
- **Row-removing steps do not fit.** Pipelines transform columns; they cannot drop outlier rows from `y` at the same time. Remove rows before splitting, using rules that do not peek at the test set.
- **Unknown categories crash by default.** `OneHotEncoder()` raises an error on a new category unless you set `handle_unknown="ignore"` or `"infrequent_if_exist"`.
- **Sparse output and pandas output do not mix.** With `set_output(transform="pandas")`, give the encoder `sparse_output=False`.
- **Custom functions need care.** A `FunctionTransformer` with a lambda cannot be saved with `joblib` reliably; use a named function or a NumPy function like `np.log1p`.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: training `[4, 6, 8, 10]`, test `2`. Compute the test z-score with the scaler fitted on training only and on all five values.
2. Swap the model for KNN with *k* = 101, then remove scaling. What changes, and why?
3. Blank out 30% of `spend` and compare imputing before CV with imputing inside the pipeline.
4. List the country columns created by the encoder and how many customers fell into the infrequent bucket.

**Quiz**

1. What does "fit inside the fold" mean?
2. What does a `ColumnTransformer` do that a plain `Pipeline` cannot?
3. Why did scaling before CV make no measurable difference here?
4. Why did selecting features before CV make such a big difference?
5. What stops the pipeline from crashing on a customer from a new country?

## 10. Recap

- A **pipeline** chains preprocessing and model into one object: fit once, cross-validate honestly, deploy as one file.
- A **ColumnTransformer** sends numeric and categorical columns through different steps.
- **Everything that learns from data goes inside.** Some leaks are tiny (scaling); some are enormous (feature selection: 55% → 80% on pure noise).
- Use `handle_unknown` and imputers so raw, messy production rows still work.
- `set_output(transform="pandas")` lets you see exactly what the model receives.

**Next:** [04.9 Bias and variance](../09-bias-variance/) explains *why* some models underfit and others overfit, and how to tell which problem you have.
