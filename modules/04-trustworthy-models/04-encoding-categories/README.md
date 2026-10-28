# 04.4 · Encoding categories: one-hot, ordinal, target encoding

⏱ about 1.5 hours · Prerequisites: 02.5 (multiple regression), 04.2 (leakage) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Retention believes that customers in some countries are more loyal than others, and marketing wants to know whether a customer's **favourite product** says anything about whether they will come back. Both are reasonable ideas. Both are text: `"France"`, `"85123A"`.

Every model you have met so far does arithmetic: multiply features by weights, measure distances, compare against thresholds. None of that works on the word "France". You have to turn categories into numbers. The obvious way (France = 1, Germany = 2, …) is quietly wrong for most models, and the clever way for columns with hundreds of values (replace each product with its churn rate) is a leakage trap. This lesson covers the three encodings you will use for the rest of your career, and when each one is safe.

## 2. The idea

Imagine giving a calculator to someone who has never heard of countries and saying "UK is 2, France is 1". They will happily conclude that the UK is "twice France" and that Germany (0) is "the least country". Numbers carry **order** and **distance**, whether you meant them to or not.

So there are two kinds of category:

- **Ordered** (shirt sizes S < M < L, delivery speed standard < express < next-day): numbers 0, 1, 2 are honest.
- **Unordered** (country, colour, payment method): any numbering invents an order. Give each value its own yes/no column instead.

And when a column has hundreds of values (products, postcodes), you can replace each value with a *summary of the target* for that value, such as "churn rate of customers whose favourite product is this". Powerful, and dangerous, because it uses the answer to build the feature.

## 3. Definition

- **Categorical feature**: a column whose values are labels, not quantities. **Nominal**: no order. **Ordinal**: has an order.
- **Cardinality**: the number of distinct values. Country has 36; favourite product has 951.
- **Label encoding**: each category gets an integer, usually alphabetically. (`OrdinalEncoder` with default settings.)
- **Ordinal encoding**: integers in an order *you* specify. (`OrdinalEncoder(categories=[["S", "M", "L"]])`.)
- **One-hot encoding**: one 0/1 column per category. (`OneHotEncoder`, `pd.get_dummies`.) One column can be dropped (`drop="first"`) because it is implied by the others; this matters for plain linear regression.
- **Target encoding** (also *mean encoding*): each category is replaced by the mean of the target for that category, usually **smoothed** towards the overall mean. (`TargetEncoder`.)
- **Unseen category**: a value that appears at prediction time but not in training.

## 4. Formula

**One-hot.** For category set $\{c_1, \dots, c_K\}$, row $i$ gets $K$ columns:

$$x_{i,k} = \begin{cases} 1 & \text{if row } i \text{ has category } c_k \\ 0 & \text{otherwise} \end{cases}$$

**Target encoding with smoothing.** For category $c$ with $n_c$ training rows, mean target $\bar{y}_c$, overall mean $\bar{y}$ and smoothing strength $m$:

$$\text{enc}(c) = \frac{n_c\,\bar{y}_c + m\,\bar{y}}{n_c + m}$$

- Large $n_c$: the category's own rate dominates.
- Small $n_c$: the encoding is pulled towards $\bar{y}$.
- $m = 0$: no smoothing, the raw category mean.

## 5. By hand

**Three encodings.** Six customers:

| customer | country | plan | label (alphabetical) | plan ordinal | DE | FR | UK |
|---|---|---|---|---|---|---|---|
| A | UK | S | 2 | 0 | 0 | 0 | 1 |
| B | DE | L | 0 | 2 | 1 | 0 | 0 |
| C | FR | M | 1 | 1 | 0 | 1 | 0 |
| D | UK | M | 2 | 1 | 0 | 0 | 1 |
| E | FR | S | 1 | 0 | 0 | 1 | 0 |
| F | DE | L | 0 | 2 | 1 | 0 | 0 |

Plan as 0/1/2 is fine: L really is bigger than S. Country as 0/1/2 claims "UK − FR = FR − DE", which means nothing.

**What label encoding does to a linear model.** On the 20-customer table, a regression of next-month spend on visits, items, recency and country gives:

- **one-hot** (DE as the reference): FR **+£15.08**, UK **+£7.32**.
- **label-encoded** (DE = 0, FR = 1, UK = 2): one coefficient, £3.58 per step, so FR **+£3.58**, UK **+£7.16**.

The label-encoded model is forced to put UK above FR and exactly twice as far from DE. The data says French customers spend the most. Training MAE: £7.46 label-encoded vs £6.35 one-hot.

**Target encoding.** Ten customers; overall churn rate $\bar{y} = 5/10 = 0.5$; smoothing $m = 2$.

| country | churned (n) | raw mean | smoothed $\frac{n\bar{y}_c + 2 \times 0.5}{n + 2}$ |
|---|---|---|---|
| UK | 2 of 4 | 0.500 | (2 + 1) / 6 = **0.500** |
| DE | 0 of 2 | 0.000 | (0 + 1) / 4 = **0.250** |
| FR | 2 of 3 | 0.667 | (2 + 1) / 5 = **0.600** |
| NL | 1 of 1 | 1.000 | (1 + 1) / 3 = **0.667** |

NL's raw encoding is 1.0, computed from a single customer: that customer's feature *is* their own label. This is target leakage (04.2) built by the encoder itself. Smoothing softens it; cross-fitting (section 6) removes it.

## 6. In code

Open [lesson.ipynb](lesson.ipynb).

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

# unordered, low cardinality → one-hot; survive new values; group rare ones
oh = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)
oh.fit(X_train[["country"]])

# ordered → ordinal with your order
ordn = OrdinalEncoder(categories=[["S", "M", "L"]])

# unordered, high cardinality → target encoding, cross-fitted on train
te = TargetEncoder(random_state=0)
fav_train = te.fit_transform(X_train[["fav_product"]], y_train)   # fit_transform, NOT fit + transform
fav_test = te.transform(X_test[["fav_product"]])
```

`TargetEncoder.fit_transform` splits the training rows into folds and encodes each fold with means from the *other* folds, so no row sees its own label. `transform` on test rows uses the means from all training rows.

## 7. On real data

- **Unseen countries.** With a stratified 75/25 split, Brazil, Singapore and the United Arab Emirates appear only in the test set. `OneHotEncoder(handle_unknown="ignore")` encodes them as all zeros; the default setting would crash.
- **Does country help predict churn?** A logistic regression on `days_since_last` + `orders` scores 66.7% on test. Adding label-encoded country: 66.7%. One-hot with rare countries grouped (France, Germany, UK, other): 66.9%. With 90% of customers in the UK, country carries almost no signal. Knowing that is useful too.
- **Favourite product** (951 values) with **naive** target encoding: training accuracy jumps from 67.8% to **78.6%**, test accuracy *falls* from 66.7% to **62.0%**. The model learned to trust a feature that contained its own answer.
- With **`TargetEncoder.fit_transform`** (cross-fitted): training 67.8%, test **66.6%**. The leak is gone. The honest result is that favourite product adds nothing here, and the naive version made the model worse while making it look better.

## 8. Where it breaks

- **Label-encoding unordered categories for linear models, KNN or neural networks.** They read the codes as amounts. Trees are different: they split on thresholds and can isolate any group of codes, so label encoding is acceptable (and compact) for tree models.
- **One-hot on high-cardinality columns** creates hundreds of mostly-zero columns, slows training and invites overfitting. Group rare values (`min_frequency`) or use target encoding.
- **Forgetting unseen categories.** New countries, new products and typos arrive in production. Decide in advance: ignore, group as "infrequent", or map to −1.
- **Target encoding without cross-fitting** leaks, as shown above. With `TargetEncoder`, use `fit_transform` on training rows; `fit` followed by `transform` on the same rows brings the leak back.
- **Encoding before splitting** lets test rows decide the categories and, for target encoding, the means. Fit encoders on train only (04.8 automates this).
- **Dropping a one-hot column** (`drop="first"`) matters for unregularised linear regression (otherwise the columns always sum to 1 and duplicate the intercept). Trees and regularised models do not need it.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Add a customer from Spain on an "XL" plan to the six-row table. What does each encoder do? Make it not crash.
2. Recompute the ten-row target encoding with $m = 0, 2, 10$.
3. Use `TargetEncoder` the wrong way (`fit` then `transform` on train) and compare with `fit_transform`.
4. Is label encoding harmful for a decision tree? Compare it with one-hot.

**Quiz**

1. Which encoding would you use for `plan_size` (S, M, L) and which for `country`? Why?
2. Why does label-encoding `country` mislead a linear regression?
3. What does `handle_unknown="ignore"` do in `OneHotEncoder`?
4. Why does naive target encoding leak, and what two techniques fix it?
5. A category has 3 customers, of whom 1 churned. The overall churn rate is 0.5 and $m = 2$. What is its smoothed target encoding?

Answers are at the end of the lesson notebook.

## 10. Recap

- Models need numbers, and numbers imply order and distance. Only give a category an integer if that order is real.
- **Ordinal** for ordered categories, **one-hot** for unordered ones with few values, **target encoding** for unordered ones with many values.
- Plan for categories you have never seen: `handle_unknown`, `min_frequency`.
- Target encoding uses the answer to build a feature. Smooth it, cross-fit it (`TargetEncoder.fit_transform`), and fit it on training rows only.
- Trees tolerate label encoding; linear models, KNN and neural networks do not.

**Next:** [04.5 Scaling features](../05-scaling/): making sure a column measured in pounds does not shout down a column measured in orders.
