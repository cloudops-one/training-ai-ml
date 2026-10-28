# 04.2 · Data leakage: the mistake that makes every model look great

⏱ about 1 hour · Prerequisites: 04.1 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

You open the old churn notebook. It does use a train/test split, so lesson 04.1's warning does not explain the disaster. The test accuracy really is **97.6%**, on customers the model never saw. And yet in production it got about 63%.

How can a model ace an honest exam and fail in real life? Because the exam was not honest after all. Somewhere, information that would **not exist at prediction time** slipped into the features. This is called **data leakage**, and it is the single most expensive mistake in applied machine learning: it produces models that look brilliant, get approved, and then quietly fail.

## 2. The idea

Imagine a student who scores 98% on a history exam. Impressive, until you discover the exam paper had the answers printed faintly on the back. The student did not learn history; they learned to read the back of the page. Take away the answers and they score like everyone else.

A leaky model is that student. It has found a column, or a preparation step, or a duplicated row that carries the answer, and it uses that instead of learning the real pattern. The test cannot catch it, because the test paper has the same faint answers on the back.

## 3. Definition

**Data leakage** is when information that would not be available at the time of prediction is used to train or evaluate the model. Three kinds cover almost every case:

1. **Target leakage**: a feature is computed from information that comes *after* (or *from*) the target. Example: "days since last order" measured on the export date, after the churn window.
2. **Train–test contamination**: a step that learns from data (scaling, filling missing values, choosing features, encoding categories) is fitted on *all* rows, including the test rows, before the split.
3. **Duplicates or groups across the split**: the same entity (an identical row, or several rows of the same customer) appears in both train and test, so the model is tested on things it has memorised.

## 4. Formula

Leakage is a timing problem, so the "formula" is a rule about timestamps. For every feature $x$ and the target $y$:

$$t(\text{feature } x \text{ is known}) \;<\; t(\text{prediction is made}) \;\le\; t(\text{target } y \text{ happens})$$

If you cannot say *when* a feature becomes known, you cannot rule out leakage.

And for every data-dependent step $s$ (scaler, imputer, selector, encoder):

$$s \text{ is fitted on } \text{rows}_{\text{train}} \text{ only, then applied to } \text{rows}_{\text{test}}$$

## 5. By hand

**Target leakage.** ShopWave's churn table uses a cut-off of 1 September 2011: features from before it, target = "no order between 1 September and 1 December". The export ran on 9 December, **99 days** after the cut-off. The leaky column is `recency_days` = days from a customer's last order to the export date.

| customer | last order before cut-off | next order after cut-off | `days_since_last` (at 1 Sep) | `recency_days` (at 9 Dec) | churned |
|---|---|---|---|---|---|
| A | 10 Aug | 15 Oct | 22 | 55 | 0 |
| B | 10 Aug | none | 22 | 121 | 1 |
| C | 1 Jul | 20 Nov | 62 | 19 | 0 |
| D | 1 Jul | none | 62 | 161 | 1 |

A and B look identical on the honest feature (`days_since_last` = 22). On `recency_days` they are worlds apart, because A's October order *after* the cut-off reset it. Any customer who came back in the window has `recency_days` < 99; any who did not has ≥ 99. One rule, "`recency_days` ≥ 99 → churned", gets nearly every customer right.

Now go live: it is 1 September next year and you export the data that morning. On that day the export date *is* the cut-off, so `recency_days` = `days_since_last` for everyone. The column no longer knows the future.

**Contamination.** Ten customers' spends: 100, 120, 150, 180, 200, 220, 250, 300, 400, **5,000**. Suppose the 5,000 customer is in the test set. The mean of all ten is 692; the mean of the nine training values is 213.3. Scaling with the mean of all ten means the training data has been shaped by a customer the model is not supposed to know exists.

## 6. In code

Open [lesson.ipynb](lesson.ipynb). The fixes are all about order and timing:

```python
# Target leakage: only features known at the cut-off
features = ["orders", "spend", "products", "tenure_days", "days_since_last"]   # not recency_days

# Contamination: split FIRST, then fit every preparation step on train only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_train)          # learns mean/std from train rows
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)             # applies train's mean/std to test

# Groups: keep all rows of a customer on the same side
from sklearn.model_selection import GroupShuffleSplit
train_idx, test_idx = next(GroupShuffleSplit(test_size=0.25).split(X, y, groups=customer_id))
```

## 7. On real data

The notebook rebuilds each leak on ShopWave data:

| leak | leaky score | honest score |
|---|---|---|
| `recency_days` measured at export | **97.6%** test accuracy, importance 0.98 | 63.1% when the same model is used live; 65.1% for a model trained without it |
| choosing 20 of 2,000 noise columns using all labels | **75%** on coin-flip labels | 45% (chance level) when selected on train only |
| scaling before vs after the split (KNN) | 63.6% | 64.5% (small, but still the wrong habit) |
| 30% of customers exported twice (unlimited tree) | **71.6%** (35% of test rows have a twin in train) | 61.1% after de-duplicating |

Note the scaling row: not every leak inflates the score dramatically, and this one even scores slightly lower. You cannot detect leakage by "the number looked fine". You prevent it by doing the steps in the right order every time.

## 8. Where it breaks

Leakage hides in ordinary-looking places. A checklist to run on every project:

- [ ] **For each feature, when is it known?** Write the date next to it. Anything computed from a table exported after the cut-off is suspect ("total spend in 2011", "tickets ever filed").
- [ ] **Is any single feature almost perfect?** A feature with most of the importance, or a correlation above about 0.8 with the target, is guilty until proven innocent.
- [ ] **Is anything fitted before the split?** Scalers, imputers, encoders, feature selection, outlier fences, PCA. All of them go after the split, or inside a `Pipeline` (04.8).
- [ ] **Can the same entity appear on both sides?** Duplicated rows, several rows per customer, several photos of the same product. De-duplicate or split by group.
- [ ] **Does the score beat the baseline by a believable amount?** Churn models on transaction data usually land 5–15 points above the baseline. A 40-point jump is a reason to investigate, not to celebrate.
- [ ] **Does it hold up on a later time period?** The final check is always a time-based test (04.1).

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. Decide for six candidate features whether they may be used at the 1 September cut-off. Then add "total spend in 2011" to the model and watch what happens.
2. Find the leak in `recency_days` using only feature importances or a correlation table.
3. Show that the noise-selection leak grows with the number of columns.
4. Compare how much a shallow tree, an unlimited tree and 1-nearest-neighbour gain from duplicated rows.

**Quiz**

1. What is target leakage, in one sentence?
2. Why did the `recency_days` model drop from 97.6% to 63.1% in production?
3. What is train–test contamination? Give an example.
4. Why do duplicated rows make a test score too optimistic?
5. A new model scores 96% on a problem where everyone else gets 70%. What do you check first?

Answers are at the end of the lesson notebook.

## 10. Recap

- Leakage = using information that will not exist at prediction time. It makes models look great and fail live.
- **Target leakage:** a feature measured after the target. Ask "when is this known?" for every column.
- **Contamination:** anything that learns from data is fitted on training rows only, after the split.
- **Duplicates / groups:** the same entity must not be on both sides of the split.
- A score that is too good is a bug report, not a result.

**Next:** [04.3 Missing values and outliers](../03-missing-and-outliers/): preparing messy columns, with every step fitted on training rows only.
