# 04.1 · Train, validation, test: why you must hide data from yourself

⏱ about 1 hour · Prerequisites: Module 03 · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

The previous data person left a churn model and a note: *"Decision tree, 100% accuracy."* The note is true. They trained the tree on all 3,317 customers and then measured it on the same 3,317 customers.

Retention does not care how well the model remembers customers it has already seen. They care what happens on **1 September next year**, when the model scores customers whose future nobody knows yet. The only way to estimate that is to keep some customers *hidden* from the model, and to treat those as the exam.

In Module 03 you used `train_test_split` with a promise that Module 04 would explain it. This is that lesson: how to split, how big the pieces should be, how much to trust the score you get, and why ShopWave in particular needs a split based on **time**.

## 2. The idea

Think of a student preparing for an exam.

- **Training set** = the textbook exercises. The student studies them as much as they like.
- **Validation set** = mock exams. The student sits them to decide *how* to study (more practice? different method?). Using them repeatedly is fine, but each use leaks a little information into the student's choices.
- **Test set** = the real exam. Sat **once**. If the student saw the real exam questions beforehand, the grade means nothing.

A model is the student. A data scientist who tunes a model by checking the test score again and again has, in effect, shown the student the real exam.

## 3. Definition

- **Training set**: the rows the model's parameters are fitted on.
- **Validation set**: rows used to compare models and choose hyperparameters (such as `max_depth`). Not used for fitting.
- **Test set** (or **holdout set**): rows used *once*, at the very end, to report the final score.
- **Random split**: rows are assigned to train or test at random.
- **Stratified split**: a random split that keeps the class proportions the same in every piece (`stratify=y`).
- **Time-based split**: everything in training comes from *before* a date, everything in test from *after* it.
- **Generalisation**: how well a model does on data it has not seen. The test score is our estimate of it.

Common proportions: 80/20 or 75/25 for train/test when you also use cross-validation (lesson 04.7), or **60/20/20** for train/validation/test when you tune with a single validation set.

## 4. Formula

A test accuracy is an estimate based on a sample of customers, so it has random error. If the true accuracy is $p$ and the test set has $n$ rows, the **standard error** is

$$SE = \sqrt{\frac{p\,(1-p)}{n}}$$

- $p$: the accuracy (a proportion between 0 and 1)
- $n$: the number of test rows
- About 95% of the time the measured accuracy lands within $\pm 2\,SE$ of the true one.

The $\sqrt{n}$ underneath is the important part: **to halve the wobble you need four times as many test rows.**

For a stratified split, each class $c$ with $n_c$ rows contributes $n_c \times \text{test\_fraction}$ rows (rounded) to the test set.

## 5. By hand

**A test set can be unlucky.** The 20-customer table has 9 churners and 11 stayers. Take a random test set of 5. The chance that it contains *no* churner is

$$P = \frac{\binom{11}{5}}{\binom{20}{5}} = \frac{462}{15{,}504} = 0.030$$

About 3%. In that case recall is undefined: there is no churner to catch. With `stratify`, the test set always gets $9 \times 5/20 = 2.25 \to 2$ churners and 3 stayers.

**How big is the wobble?** On the real table a depth-4 tree scores about 66%.

| test rows $n$ | $SE = \sqrt{0.66 \times 0.34 / n}$ | 95% range |
|---|---|---|
| 830 | $\sqrt{0.2244/830} = 0.0164$ | ±3.3 points |
| 100 | $\sqrt{0.2244/100} = 0.0474$ | ±9.5 points |

So with 830 test customers, a model at 66% can honestly score anywhere from about 63% to 69%. If a colleague says "my model is 1 point better than yours" on a test set this size, the honest reply is "maybe".

## 6. In code

Open [lesson.ipynb](lesson.ipynb). The key calls:

```python
from sklearn.model_selection import train_test_split

# train / test, keeping the churn rate equal in both
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# train / validation / test = 60 / 20 / 20: split twice
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42)
```

`random_state` fixes the shuffle so you (and your readers) get the same split every run. Change it and the score changes, which is exactly what section 4 predicts.

## 7. On real data

The notebook shows four things on ShopWave's 3,317 customers:

1. **Homework vs exam.** An unlimited decision tree scores **100%** on the customers it learned from and **59.9%** on 830 hidden customers, barely above the 56.4% you get by predicting "stays" for everyone.
2. **Luck.** The same depth-4 tree, split 20 different ways, scores between **63.5% and 69.0%** with 830 test customers and between **61% and 73%** with only 100, matching the ±3.3 and ±9.5 points above.
3. **Tuning on validation.** Trying `max_depth` 1 to 10 on a 60/20/20 split: training accuracy climbs from 66% to 84%, validation accuracy *falls* from 67% to 63%. The validation set picks depth 1, which scores **65.8%** on the test set, looked at once.
4. **Time.** A tree trained on the **June** snapshot (features before 1 June, target June–August) scores **66.8%** on the **September** snapshot, about the same as a random split inside September (65.1%). Good news: this model survives three months. And the churn rate itself moved from **50.4%** to **43.6%** between snapshots. A random split could never have shown you that.

## 8. Where it breaks

- **Random splits on time-ordered data.** If the same customer or the same week appears on both sides, the test score is too optimistic. For forecasting (Module 07) a random split is simply wrong.
- **Tiny test sets.** Below a few hundred rows the score is mostly noise. Use cross-validation (04.7) instead of one split.
- **Re-using the test set.** Every time you look at the test score and change something, the test set becomes a little more like a validation set. Look once.
- **Groups.** If one customer has many rows (one per order, say), split by *customer*, not by row, or the model meets the same customer in train and test. Lesson 04.2 shows this.
- **Splitting after preparing.** Scaling, filling missing values or selecting features *before* the split lets the test set influence training. Lesson 04.2 shows why, lesson 04.8 shows the cure.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: a model scores 70% on 300 test customers. Compute the standard error and a 95% range. Check in NumPy.
2. For test sizes 50, 200 and 800, run 30 random splits each and compare the spread of scores with the formula.
3. Tune `n_neighbors` for KNN on a 60/20/20 split and report the test score once.
4. Train on a March snapshot and test on September. Is six months too long a gap?

**Quiz**

1. Why is a model's score on its own training rows not a measure of how good it is?
2. What is the difference between a validation set and a test set?
3. What does `stratify=y` do, and when does it matter most?
4. A model scores 80% on 400 test rows. Roughly how far could the true accuracy be from 80%?
5. Which kind of split best rehearses how ShopWave will use a churn model, and why?

Answers are at the end of the lesson notebook.

## 10. Recap

- Score a model only on rows it never saw. Its score on training rows measures memory.
- **Train** to fit, **validation** to choose, **test** once to report. 60/20/20 is a good default when tuning with one validation set.
- Stratify classification splits; it is free and protects small or rare classes.
- A test score wobbles by about $\pm 2\sqrt{p(1-p)/n}$. Small differences on small test sets mean nothing.
- When the model will be used on the future, test it on the future: a **time-based split**.

**Next:** [04.2 Data leakage](../02-data-leakage/): how answers sneak into training even when you split correctly.
