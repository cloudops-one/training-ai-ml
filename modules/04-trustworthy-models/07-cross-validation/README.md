# 04.7 · Cross-validation: a fairer score

⏱ about 1.5 hours · Prerequisites: 04.1 (train/validation/test), 03.2 (KNN) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

In 04.6 you logged the skewed columns and the churn model's AUC went from 0.741 to 0.747. Your manager asks: *"Is that real? Should we change the production model?"*

You have one number from one test split of 830 customers. Had the random split put a slightly different 830 customers in the test set, would the gain still be there? Would it be bigger? Would it reverse? With a single split you cannot tell. You also used that same test set in Module 03 to choose KNN's *k*, which means it has quietly become part of training.

You need a score that does not depend on the luck of one split, and a way to compare options **without** touching the test set.

## 2. The idea

A teacher who sets one exam question might pick one you happened to revise or one you happened to skip. A fairer teacher sets five questions on different topics and averages. The average tells you how good you are; the spread tells you how consistent.

**Cross-validation** does the same with data. Cut the training rows into *k* equal groups (folds). Train on all but one fold, test on the one left out. Repeat *k* times so every fold is the test fold once. Average the *k* scores.

```
5-fold cross-validation on the training set (the test set stays locked away)

fold 1:  [TEST] [train] [train] [train] [train]   → score 1
fold 2:  [train] [TEST] [train] [train] [train]   → score 2
fold 3:  [train] [train] [TEST] [train] [train]   → score 3
fold 4:  [train] [train] [train] [TEST] [train]   → score 4
fold 5:  [train] [train] [train] [train] [TEST]   → score 5
                                                   ─────────
                                                   mean ± std
```

## 3. Definition

**k-fold cross-validation** splits the training data into *k* folds, trains *k* models (each on *k* − 1 folds) and evaluates each on its held-out fold. The **CV score** is the mean of the *k* scores; the **standard deviation** measures how much the score depends on which rows were held out.

- **Stratified k-fold**: each fold keeps the same class proportions as the whole set. Use it for classification.
- **Leave-one-out (LOO)**: *k* = number of rows. Each model is tested on one row. Expensive; sometimes used on tiny data.
- **Time-series split**: folds respect time order; every model trains on the past and tests on the following period.

Cross-validation is for **choosing** (a model, a *k*, a feature set). The test set is for **reporting**, once, at the end.

## 4. Formula

With fold scores $s_1, \dots, s_k$:

$$\text{CV score} = \bar{s} = \frac{1}{k}\sum_{i=1}^{k} s_i
\qquad\qquad
\text{spread} = \sqrt{\frac{1}{k}\sum_{i=1}^{k} (s_i - \bar{s})^2}$$

Report them together: **0.80 ± 0.24**. Each model trains on a fraction $\frac{k-1}{k}$ of the training rows (80% for *k* = 5).

## 5. By hand

Ten customers. The model is **1-nearest-neighbour on `days_since_last`**: a test customer gets the label of the closest training customer.

| row | days_since_last | churned |
|---|---|---|
| 1 | 5 | 0 |
| 2 | 80 | 1 |
| 3 | 12 | 0 |
| 4 | 60 | 1 |
| 5 | 28 | 0 |
| 6 | 95 | 1 |
| 7 | 42 | 1 |
| 8 | 18 | 0 |
| 9 | 72 | 0 |
| 10 | 52 | 1 |

**Step 1: the folds.** Five folds of two consecutive rows: {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}.

**Step 2: fold 1.** Test rows 1 and 2; train on rows 3–10.

- Row 1 (5 days): closest training value is 12 (row 3, distance 7) → predict 0. True 0 ✓.
- Row 2 (80 days): closest are 72 (row 9, distance 8) and 95 (distance 15) → row 9 → predict 0. True 1 ✗.
- Accuracy 1/2 = **0.5**.

**Step 3: fold 2.** Test rows 3 (12) and 4 (60); train on 1, 2, 5–10.

- 12 → nearest 18 (distance 6; 5 is 7 away) → 0 ✓.
- 60 → nearest 52 (distance 8; 72 is 12 away) → 1 ✓.
- Accuracy **1.0**.

**Step 4: fold 3.** Test 28 and 95. 28 → 18 (10 away; 42 is 14) → 0 ✓. 95 → 80 → 1 ✓. **1.0**.

**Step 5: fold 4.** Test 42 and 18. 42 → 52 (10 away; 28 is 14) → 1 ✓. 18 → 12 (6 away) → 0 ✓. **1.0**.

**Step 6: fold 5.** Test 72 and 52. 72 → 80 (8 away; 60 is 12) → 1, true 0 ✗. 52 → 60 (8 away; 42 is 10) → 1 ✓. **0.5**.

**Step 7: average and spread.** Scores 0.5, 1, 1, 1, 0.5.

$\bar{s} = 4 / 5 = 0.8$

Deviations: −0.3, 0.2, 0.2, 0.2, −0.3. Squares: 0.09, 0.04, 0.04, 0.04, 0.09. Sum 0.30. Divide by 5: 0.06. Square root: **0.245**.

**Result: 0.80 ± 0.24.**

Now the point. A single split might have been fold 1 (you would report 50%) or fold 3 (you would report 100%). Both are true facts about one split, and both are wrong as a description of the model.

## 6. In code

The loop you just did is one line:

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
scores = cross_val_score(KNeighborsClassifier(n_neighbors=1), X10, y10, cv=KFold(5))
scores.mean(), scores.std()            # 0.8, 0.245
```

For real classification work use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Passing `cv=5` to a classifier does stratification automatically, but not shuffling. Change the metric with `scoring="roc_auc"`, `"f1"`, `"neg_mean_absolute_error"` and so on.

## 7. On real data

On the 2,487 ShopWave training customers (the 830 test customers stay locked away):

- **One split wobbles.** Twenty different random single splits of the training set give the same logistic regression an accuracy anywhere from **0.619 to 0.711**.
- **CV is steadier.** 5-fold stratified CV: folds 0.651, 0.701, 0.662, 0.692, 0.720 → **0.685 ± 0.026**.
- **Choosing *k* honestly.** CV accuracy for KNN rises from 0.597 (*k* = 1) to **0.682 at *k* = 101**, then flattens (0.680 at 201). The single final test score is 0.669, inside the CV error bar.
- **The 04.6 gain was luck.** Five-fold CV AUC is **0.740 for raw features and 0.740 for logged ones**; logged wins two folds and loses three. The +0.006 seen on one split does not survive.

That last result is the reason this lesson exists.

## 8. Where it breaks

- **Preprocessing outside the folds leaks.** If you scale, impute or select features on all training rows *before* cross-validating, the held-out fold has influenced training. Lesson 04.8 fixes this with pipelines and shows a case where the leak inflates the score from 0.55 to 0.80.
- **Time-ordered data needs `TimeSeriesSplit`.** Shuffled folds let the model learn from the future.
- **Groups need `GroupKFold`.** If one customer has several rows, all of them must sit in the same fold or the model sees the answer.
- **More folds is not always better.** Means barely change from 3 to 20 folds; the spread grows (each test fold is smaller) and the cost grows linearly.
- **CV does not replace the test set.** After you have chosen using CV, the CV score is slightly optimistic. Report the test score, once.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: use 2 folds (rows 1–5 and 6–10) with 1-NN. Compute both fold accuracies and the mean.
2. Run leave-one-out on the ten rows. Which rows are misclassified, and why?
3. Compare 3, 5, 10 and 20 folds on the real data: mean, standard deviation, time.
4. Choose a decision tree's `max_depth` by CV, then score it once on the test set.

**Quiz**

1. In k-fold CV, how many times is each row used for testing?
2. With 5-fold CV, how many models are trained, and on what fraction of the training data?
3. Why report the standard deviation as well as the mean?
4. What is wrong with trying several *k* values on the test set and keeping the best?
5. Which splitter would you use for (a) churn classification, (b) monthly sales?

## 10. Recap

- One train/test split gives one noisy number. On ShopWave it ranged from 0.62 to 0.71 for the same model.
- **k-fold CV** trains *k* models, tests each on a different fold and reports **mean ± std**.
- Use **stratified** folds for classification and **time-series** splits for data in time order.
- Use CV to **choose**; use the test set once to **report**.
- Before believing a small improvement, cross-validate and compare fold by fold. The 04.6 "gain" did not survive.

**Next:** [04.8 Pipelines](../08-pipelines/) makes sure preprocessing happens *inside* each fold, so cross-validation stays honest.
