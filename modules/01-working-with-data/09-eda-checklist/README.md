# 01.9 · Exploratory data analysis: a repeatable checklist

⏱ about 2 hours · Prerequisites: 01.5–01.8 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

A new dataset lands: *"Here's the export from the loyalty-card system. Can we predict which members will upgrade to premium?"* Four thousand rows, thirty columns, no documentation. Where do you start?

Most beginners start with a model. It scores 99%. Then someone notices one column is `upgraded_on_date`, which is only filled in for members who upgraded. The model learned to read the answer off the input. Three days wasted.

**Exploratory data analysis (EDA)** is the routine that catches that in ten minutes. It is not "make some charts". It is a fixed list of questions you ask every dataset, in order, before you trust it with a model.

**Why this matters for ML:** every modelling disaster in this course's "Where it breaks" sections (leakage, imbalance, outliers, duplicated features, skewed targets) is visible in EDA if you look. EDA is where you decide what the target is, which columns are features, which are poison, and what cleaning and transformation the model will need.

## 2. The idea

A pilot's pre-flight checklist does not change because the pilot is experienced. Neither should yours. Eight questions, always in the same order, each with a one-line answer written down:

1. **Shape**: how many rows, how many columns, what is one row?
2. **Types**: what is each column, numeric, categorical, date, text, ID?
3. **Missing**: how much, where, and is it random?
4. **Duplicates**: exact and near.
5. **Target**: what is its distribution? skewed? imbalanced? how was it produced?
6. **Each feature alone**: distribution, range, outliers, odd values.
7. **Features vs target**: which ones relate, and how?
8. **Features vs each other and leakage**: duplicated information, and anything that *could not be known at prediction time*.

The output is a short written summary. Not the charts; the sentences.

## 3. Definition

- **EDA**: a systematic first pass over a dataset using tables and charts to understand its structure, quality and relationships before modelling.
- **Unit of analysis**: what one row represents (a transaction? a customer? a day?). Models predict at this level; if your rows are transactions and your question is about customers, you must aggregate first (01.6).
- **Target leakage**: a feature that contains information about the target that would not be available when the prediction is actually made. Classic forms: a column filled in *after* the outcome; a column derived from the target; a timestamp from the future.
- **Class imbalance**: for a categorical target, one class is much rarer than the other (2% churners, 98% not). Accuracy becomes meaningless (Module 03).
- **Cardinality**: the number of distinct values in a categorical column. `Country` has 38; `StockCode` has 3,665; `Invoice` has 18,000 (that is an ID, not a feature).

## 4. Formula

No new maths. Two rules of thumb to write on a sticky note:

> **Missing %** per column (01.5). **Skew**: if mean ≫ median, the column is right-skewed; consider `log1p()` before modelling.
>
> **Imbalance ratio** = minority count ÷ majority count. Below about 1:4, you need the techniques in 05.7.

## 5. By hand

A ten-row loyalty extract (simplified):

| member | age | visits | country | joined | upgraded_on | premium |
|---|---|---|---|---|---|---|
| 1 | 34 | 12 | UK | 2019-03 | 2021-05 | 1 |
| 2 | 51 | 3 | UK | 2020-11 | | 0 |
| 3 | 29 | 40 | DE | 2018-07 | 2019-01 | 1 |
| 4 | — | 7 | UK | 2021-01 | | 0 |
| 5 | 45 | 0 | FR | 2021-06 | | 0 |
| 6 | 38 | 15 | UK | 2019-09 | | 0 |
| 7 | 62 | 22 | UK | 2017-02 | 2018-03 | 1 |
| 8 | 23 | 1 | UK | 2021-08 | | 0 |
| 9 | 41 | 9 | DE | 2020-02 | | 0 |
| 10 | 33 | 350 | UK | 2020-05 | | 0 |

Run the checklist:

1. **Shape**: 10 rows × 7 columns. One row = one member. Good: the question is about members.
2. **Types**: `member` is an ID (not a feature). `age`, `visits` numeric. `country` categorical. `joined`, `upgraded_on` dates. `premium` is the target (0/1).
3. **Missing**: `age` 1/10 (10%); `upgraded_on` 7/10, **and it is missing exactly when premium = 0**. That is not random.
4. **Duplicates**: none.
5. **Target**: 3 of 10 premium (30%). Mildly imbalanced; fine for now.
6. **Features alone**: `visits` has a 350 among values of 0–40. Outlier: a bot, a shared account, or a data error. `age` 23–62, plausible.
7. **Features vs target**: premium members have visits 12, 40, 22 (mean 25); non-premium 3, 7, 0, 15, 1, 9, 350 (median 7). Visits relate to the target. Premium members joined earlier (2017–2019) than most non-premium. Tenure may matter.
8. **Leakage**: `upgraded_on` is filled *only after* a member upgrades. At prediction time (deciding who to target) it is always empty. **Drop it.** A model given this column scores 100% and learns nothing.

Written summary: *"One row per member; target 30% positive; `upgraded_on` is leakage, drop; `visits` strongly related to target but has an outlier (member 10) to check; `age` 10% missing, impute median; consider tenure = today − joined as a feature."* Five sentences, and you have a modelling plan.

## 6. In code

A skeleton you will reuse. Each step is one or two Pandas calls you already know:

```python
df.shape; df.head(); df.sample(5)                     # 1 shape
df.dtypes; df.nunique()                              # 2 types + cardinality
df.isna().mean().sort_values(ascending=False)        # 3 missing
df.duplicated().sum()                                # 4 duplicates
df[target].describe(); df[target].value_counts(normalize=True)   # 5 target
df.describe().T; sns.histplot / sns.boxplot          # 6 each feature
df.groupby(target).mean(); sns.scatterplot(x=f, y=target)       # 7 feature vs target
df.corr(numeric_only=True); sns.heatmap               # 8 redundancy
```

The notebook wraps step 1–4 into a reusable `overview(df)` function.

## 7. On real data

You run the full checklist on the ShopWave **customer table** with a real target: `spend_next_quarter` (what each customer spent in Sep–Nov 2011, computed from the data) using features built only from Dec 2010–Aug 2011. You will find a heavily skewed target with 44% zeros (customers who did not come back), three near-duplicate features, and one deliberately planted leaky column that you must catch.

## 8. Where it breaks

- **Charts without sentences.** Thirty histograms nobody reads. Each step ends with one written finding or it did not happen.
- **EDA on the wrong unit.** Exploring transactions when you predict per customer. Aggregate first.
- **Peeking at the test set.** In Module 04 you will hold data back. EDA decisions (which features, which transformations) must be made on training data only, or you leak.
- **Deleting outliers you do not understand.** Member 10 with 350 visits may be your best customer or a bot. Find out.
- **Treating missingness as noise.** `upgraded_on` being missing *is* the target. Missingness can be the most informative feature in the table.
- **Trusting column names.** `revenue` might be net or gross, in pounds or pence, with or without VAT. Check ranges against something you know.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Run `overview()` on the raw transactions (not the customer table). Write five sentences of findings. Which of the eight steps do not apply to transactions, and why?
2. The customer table has a 0/1 column `came_back`. What is the imbalance ratio? Which single feature separates the two classes best (compare medians)?
3. Add the feature `avg_days_between_orders`. What is its missing rate, and *why* is it missing for those customers? Is that missingness informative?
4. List every pair of features with correlation above 0.6. For each pair, say which one you would keep.

**Quiz**

1. What is the unit of analysis, and why must it match the question?
2. Give two signs that a column is leaking the target.
3. The target is 3% positive. Which later technique do you note down now?
4. Mean spend £2,000, median £650. What transformation might you plan?
5. Why must EDA conclusions be drawn on training data only?

## 10. Recap

- Eight questions, same order, every time: shape, types, missing, duplicates, target, each feature, features vs target, features vs each other + leakage.
- Output = written sentences, not charts.
- Leakage = anything not knowable at prediction time. Columns filled after the outcome are the classic case.
- Note skew (plan a log), imbalance (plan Module 05), redundancy (plan to drop), outliers (plan to investigate).
- The customer table with a time-based target you built here is the template for every supervised problem in this course.

**Next:** [01.P Mini-project: the ShopWave sales report](../project/), then on to [Module 02](../../02-regression/).
