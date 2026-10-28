# 04.3 · Missing values and outliers done properly

⏱ about 1.5 hours · Prerequisites: 01.5 (missing values in Pandas), 01.8 (quartiles, standard deviation), 04.1, 04.2 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

Retention wants a new feature in the churn model: **the average number of days between a customer's orders**. It makes sense: someone who orders every two weeks and has been silent for two months is a worry; someone who orders twice a year and has been silent for two months is not.

You compute it, and 41.5% of customers get `NaN`. Scikit-learn models refuse to train on `NaN`. Meanwhile the `spend` column has a customer at £176,356 next to a median of £553, and your logistic regression is struggling with it.

In Module 01 you cleaned data for *reports*: drop the bad rows, fill a gap, move on. For a *model*, every choice changes what the model learns, and every fill value or outlier fence is something learned from data, which means it can leak (04.2). This lesson is about making those choices deliberately.

## 2. The idea

**Missing values.** Before filling a hole, ask *why it is there*. A form field left blank because the server crashed tells you nothing. A field left blank because the customer chose not to answer, or because the thing did not exist (no second order, so no gap between orders), tells you something. Throwing that information away is like a doctor ignoring that a patient refused a blood test.

**Outliers.** An outlier is a value far from the others. It is a *question*, not a verdict: is it an error (a price typed as £5,000 instead of £50.00), or a real extreme (a wholesaler)? Errors get fixed or removed. Real extremes stay, and you choose a model or a transformation that is not bullied by them.

## 3. Definition

**Kinds of missingness** (names from statistics, meanings in plain words):

- **MCAR, missing completely at random**: the gap has nothing to do with anything. A server glitch dropped random rows.
- **MAR, missing at random**: the gap is explained by *other columns you can see*. Guest checkouts (no customer ID) are far more common in the UK than in Germany.
- **MNAR, missing not at random**: the gap is linked to the *missing value itself* or to the target. No `avg_gap_days` because the customer only ever ordered once, and one-time customers churn more.

**Imputation** = filling a missing value with an estimate:

- **mean** or **median** of the column (numbers), **mode** = most frequent value (categories), or a **constant** (0, −1, "Unknown").
- **Missing indicator** = an extra 0/1 column saying "this value was filled in", so the model can still see that it was missing.

**Outlier rules:**

- **IQR rule**: outside $[Q_1 - 1.5\,IQR,\; Q_3 + 1.5\,IQR]$, where $IQR = Q_3 - Q_1$.
- **z-score rule**: $|z| > 3$, where $z = (x - \bar{x}) / s$.

**Treatments:** **drop** the row, **cap** (clip) values at a limit, or **transform** the column (usually $\log(1+x)$) so extremes are pulled in.

## 4. Formula

$$\text{mean} = \frac{1}{n}\sum x_i \qquad \text{median} = \text{middle value of the sorted known values}$$

$$IQR = Q_3 - Q_1 \qquad \text{fences} = Q_1 - 1.5\,IQR \;\text{ and }\; Q_3 + 1.5\,IQR$$

$$z_i = \frac{x_i - \bar{x}}{s} \qquad s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n-1}}$$

$$\text{log transform: } x' = \log(1 + x) \quad (\text{the } 1+ \text{ keeps } x = 0 \text{ finite})$$

Every one of these numbers ($\bar{x}$, median, $Q_1$, $Q_3$, $s$, a cap) is **learned from data**. So each is computed on training rows only and then applied to test rows.

## 5. By hand

**Imputation.** Eight customers' average gap between orders (days): 12, 20, ?, 30, ?, 45, 90, ?.

- Known values: 12, 20, 30, 45, 90. Mean = 197 / 5 = **39.4**. Median = **30** (the middle of five sorted values).
- Mean fill: 12, 20, **39.4**, 30, **39.4**, 45, 90, **39.4**. The single 90 pulled the fill value up.
- Median fill: 12, 20, **30**, 30, **30**, 45, 90, **30**.
- Indicator column: 0, 0, 1, 0, 1, 0, 0, 1.
- Country for the same customers: UK, UK, DE, ?, UK, FR, ?, UK. Mode = **UK** (4 of 6 known).

Train-only version: if rows 1–5 are the training set, the known training values are 12, 20, 30, so the mean the imputer learns is 62/3 = **20.67**, and every gap, in train *and* test, gets 20.67.

**Outliers and masking.** Eight quarterly spends: 120, 150, 180, 200, 240, 260, 300, **4,800**.

*IQR rule, by the paper method* (median of each half): lower half 120, 150, 180, 200 → $Q_1 = 165$; upper half 240, 260, 300, 4,800 → $Q_3 = 280$; $IQR = 115$; upper fence $= 280 + 172.5 = 452.5$. **4,800 is an outlier.** (NumPy interpolates between values and gets $Q_1 = 172.5$, $Q_3 = 270$, fence 416.25. Same verdict; the methods differ slightly on small samples, as you saw in 01.8.)

*z-score rule:* mean $= 6{,}250 / 8 = 781.25$. Squared deviations sum to about 18.48 million; divide by 7 and take the root: $s \approx 1{,}624.9$. Then

$$z_{4800} = \frac{4800 - 781.25}{1624.9} \approx 2.47$$

**Not an outlier by the z rule**, because 4,800 inflated the very mean and standard deviation it is being measured against. This is called **masking**. Against the other seven customers alone (mean 207, s 63), it would be 72 standard deviations away. On skewed data, prefer the IQR rule.

## 6. In code

Open [lesson.ipynb](lesson.ipynb).

```python
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="median", add_indicator=True)
imp.fit(X_train)                     # learns one median per column, from TRAIN rows
X_train_f = imp.transform(X_train)   # fills gaps and appends indicator columns
X_test_f = imp.transform(X_test)     # same medians, applied to test

# outliers: learn the cap on train, apply everywhere
cap = X_train["spend"].quantile(0.99)
X_train["spend"] = X_train["spend"].clip(upper=cap)
X_test["spend"] = X_test["spend"].clip(upper=cap)

# or transform instead
X_train["spend"] = np.log1p(X_train["spend"])
```

`strategy` can be `"mean"`, `"median"`, `"most_frequent"` or `"constant"` (with `fill_value=`).

## 7. On real data

- **Missing customer IDs.** 24.9% of raw transactions have no `Customer ID`. The rate is 0% in Germany, 0.8% in France, 8.7% in Ireland and **27%** in the UK: missing *at random* given country. Dropping them is right for a churn model (we cannot follow a guest), but would distort a revenue-by-country report.
- **Missing descriptions** (0.27%) are all price-£0 rows with no customer: warehouse adjustments, not sales. The standard cleaning removes them.
- **`avg_gap_days`** is missing for 41.5% of customers, the ones with a single order. Those customers churn at **62.5%**, versus **30.2%** for customers with a gap. Missingness *is* the signal.
- **Imputation in a churn model** (logistic regression on `days_since_last` + `avg_gap_days`): mean fill 64.3%, median fill 64.9%, median + indicator **65.7%**.
- **Spend outliers.** The IQR fence is £3,000 and flags **293 customers (8.8%)**; the z > 3 rule flags only 27 (masking again). The top five spenders are wholesalers in four countries, and all five came back. These are the last people to delete.
- **Treatments** in a logistic regression with `days_since_last` + `spend`: raw 64.2%, capped at the 99th percentile 64.6%, **log(1 + spend) 67.3%**. The problem was the skew of the whole column, not the top 1%.

## 8. Where it breaks

- **Dropping rows with any missing value** (`dropna()`) can silently throw away half your data, and a biased half (here: every one-time customer, the ones most likely to churn).
- **Mean imputation on skewed data** fills gaps with a value pulled towards the extremes. Prefer the median.
- **Imputing without an indicator** erases the information that the value was missing. When missingness is MNAR, that information is often the best feature you have.
- **Deleting outliers from training but not at prediction time.** The £176,000 customer will still be scored live. A model that never saw anyone like them will extrapolate badly. Transform instead of delete, or use trees, which do not care about scale.
- **Fitting any of this on all the data.** The median, the quartiles, the cap: train rows only (04.2). Lesson 04.8 puts all of it inside a `Pipeline` so it cannot go wrong.
- **Trees vs linear models.** A tree only asks "above or below?", so a sentinel value like −1 for missing works, and log transforms change nothing. Linear models, KNN and neural nets are sensitive to both. Prepare the data for the model you are using.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: the mean an imputer learns when only five of the eight rows are training rows. Check in code.
2. Investigate the rows with a missing `Description`. MCAR, MAR or MNAR? What should you do with them?
3. Recompute the wholesaler's z-score against the other seven customers only.
4. Fill `avg_gap_days` with −1, 0 and 999 for a decision tree. Does the constant matter?

**Quiz**

1. Explain MCAR, MAR and MNAR with one ShopWave example each.
2. Why must `SimpleImputer` be fitted on training rows only?
3. What does a missing-indicator column add that imputation alone does not?
4. Why can the z-score rule fail to flag an extreme value?
5. The IQR rule flags 293 high-spending customers. Should you delete them? What would you do instead?

Answers are at the end of the lesson notebook.

## 10. Recap

- Ask **why** a value is missing before filling it. MNAR missingness is information; keep it with an indicator column.
- Median for skewed numbers, mode for categories, a constant for trees. Fit on train, apply to test.
- An outlier is a question: error or real extreme? Fix errors; keep real extremes.
- The IQR rule is robust; the z-score rule can be masked by the very outlier it is looking for.
- For skewed columns like spend, a **log transform** usually beats capping or deleting. Trees do not need either.

**Next:** [04.4 Encoding categories](../04-encoding-categories/): turning `country` and other text columns into numbers a model can use.
