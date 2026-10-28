# 05.2 · Bagging and Random Forest

⏱ about 1.5 hours · Prerequisites: 05.1, 03.3 (decision trees), 04.7 (cross-validation), 04.9 (overfitting) · Notebook: [lesson.ipynb](lesson.ipynb)

---

## 1. The problem

In 03.3 you saw that a decision tree grown to full depth memorises the training customers: 100% training accuracy, poor test accuracy. Limit the depth and it underfits. A single tree is **unstable**: change a few training customers and the first split can change, and everything below it with it.

Lesson 05.1 showed that averaging *different* models cancels their mistakes. Here is the trick: you can get many different trees out of **one** dataset, by giving each tree a slightly different version of the data and a slightly different set of questions it is allowed to ask. Average a few hundred of them and the wobble disappears. That is a **random forest**, and it is one of the most reliable models in all of machine learning.

## 2. The idea

Imagine asking 300 analysts to each build a churn rule. To stop them all copying each other, you give each one:

1. **A different sample of customers**: each analyst gets a bag of 2,487 customer cards drawn *with replacement* from the deck, so some customers appear twice, some not at all.
2. **A restricted menu at every decision**: whenever an analyst wants to split a group, they may only look at 2 of the 5 features, chosen at random for that split.

Each analyst's rule is a bit worse than one built with full information. But their mistakes are now *different*, and the average of 300 different decent rules beats any one of them.

## 3. Definition

- **Bootstrap sample**: $n$ rows drawn **with replacement** from $n$ rows. Some rows repeat; on average about 37% are left out.
- **Out-of-bag (OOB) rows**: the rows a given tree did *not* see. They are a free test set for that tree.
- **Bagging**: train one model per bootstrap sample, then average their predictions (or vote).
- **Random forest**: bagging with decision trees, **plus** at every split each tree considers only a random subset of features (`max_features`, by default $\sqrt{\text{number of features}}$ for classification).
- **OOB score**: each training row is scored only by the trees that did not see it, and those predictions are compared with the truth. It approximates cross-validation without any extra training.
- **Feature importance** (impurity-based): how much, on average across all trees and splits, each feature reduced Gini impurity. Quick and rough; 05.8 shows better methods.

Key settings in scikit-learn's `RandomForestClassifier`:

| setting | meaning | typical |
|---|---|---|
| `n_estimators` | number of trees | 200–500; more never overfits, just costs time |
| `max_features` | features considered per split | `"sqrt"` |
| `max_depth`, `min_samples_leaf` | how big each tree may grow | trees are grown deep; `min_samples_leaf` 1–20 |
| `oob_score=True` | compute the OOB score | turn on, it is free |
| `n_jobs=-1` | use all CPU cores | trees are independent, so they parallelise perfectly |

## 4. Formula

**Why about 37% are out of bag.** In one draw, a particular row is *not* picked with probability $1 - \frac{1}{n}$. In $n$ independent draws:

$$P(\text{row never picked}) = \left(1 - \frac{1}{n}\right)^{n} \;\longrightarrow\; e^{-1} \approx 0.368 \quad\text{as } n \text{ grows}$$

So each bootstrap sample contains about **63%** of the distinct rows, and about **37%** are out of bag. For $n = 10$ the exact value is $0.9^{10} = 0.349$; for $n = 2{,}487$ it is $0.368$.

**The forest's prediction** for a customer $x$ with $T$ trees:

$$\hat{p}(x) = \frac{1}{T}\sum_{t=1}^{T} \hat{p}_t(x)$$

The average of the trees' churn probabilities. Averaging $T$ predictions with variance $\sigma^2$ and pairwise correlation $\rho$ gives variance $\rho\,\sigma^2 + \frac{1-\rho}{T}\sigma^2$. More trees shrink the second term; random feature subsets shrink $\rho$. The first term is why you cannot average your way to perfection with correlated trees.

## 5. By hand

**A bootstrap sample.** Ten customers, numbered 0–9. Draw ten times with replacement (the notebook uses a fixed seed, so you get these same draws):

```
draws:  7  9  2  2  7  8  5  1  8  5
```

| customer | times drawn | in bag? |
|---|---|---|
| 0 | 0 | **out of bag** |
| 1 | 1 | in |
| 2 | 2 | in (twice) |
| 3 | 0 | **out of bag** |
| 4 | 0 | **out of bag** |
| 5 | 2 | in (twice) |
| 6 | 0 | **out of bag** |
| 7 | 2 | in (twice) |
| 8 | 2 | in (twice) |
| 9 | 1 | in |

Six distinct customers made it into the bag, four are out of bag. This tree is trained on the bag and can be honestly scored on customers 0, 3, 4 and 6. The formula predicted $0.349 \times 10 \approx 3.5$ out-of-bag rows; we got 4.

**A random feature subset.** The churn table has 5 features. With `max_features="sqrt"`, each split looks at $\lfloor\sqrt{5}\rfloor = 2$ of them, chosen at random, for example `{spend, tenure_days}` at the root of one tree and `{orders, days_since_last}` at the root of another. The strongest feature therefore cannot dominate every tree, which makes the trees more different from each other.

## 6. In code

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,        # 300 trees
    min_samples_leaf=10,     # each leaf holds at least 10 customers
    oob_score=True,          # free out-of-bag estimate
    random_state=42, n_jobs=-1,
)
rf.fit(X_train, y_train)
rf.oob_score_                 # OOB accuracy
rf.feature_importances_       # one number per feature, summing to 1
```

No scaling is needed: trees split on thresholds, so units do not matter.

## 7. On real data

The notebook compares, on the ShopWave churn table (CV AUC on 2,487 training customers, then test AUC on 830):

| model | CV AUC | test AUC |
|---|---|---|
| one tree, unlimited depth | 0.593 | 0.593 |
| one tree, depth 3 | 0.731 | 0.724 |
| random forest, default settings | 0.713 | 0.702 |
| random forest, `min_samples_leaf=10` | 0.736 | 0.731 |
| *benchmark: logistic regression* | *0.740* | *0.741* |

Read it carefully:

- The forest turns a terrible overfit tree (0.59) into a strong model (0.74). Averaging is doing exactly what 05.1 promised.
- Letting each tree grow until every leaf is pure (`min_samples_leaf=1`, the default) is noticeably worse than requiring 10 customers per leaf. With noisy labels like churn, slightly smoother trees average better.
- The forest **ties but does not beat** logistic regression. With five features whose effects are mostly "more orders → less churn, more days away → more churn", a straight line already captures most of the signal. Forests win when the signal lives in thresholds and interactions; 05.4 shows such a case.

The **number of trees** curve (test AUC): 1 tree 0.678 → 5 trees 0.711 → 25 trees 0.723 → 100 trees 0.730 → 300 trees 0.731. It rises fast, then flattens. More trees never hurt accuracy; they only cost time.

**Feature importance** (impurity-based): spend 0.28, days since last order 0.21, orders 0.20, products 0.18, tenure 0.12. Money and recency lead, as in Module 03.

## 8. Where it breaks

- **Impurity importance is biased** towards continuous features with many possible split points (like `spend`) and is computed on training data. Use permutation importance or SHAP for decisions (05.8).
- **No extrapolation.** Like every tree model, a forest predicts within the range it saw. A customer with ten times the largest training spend gets treated like the largest one.
- **Big and slow to predict** compared with a linear model: 300 deep trees can be hundreds of megabytes.
- **Not interpretable as a whole.** You cannot print 300 trees for the CEO.
- **OOB needs bootstrap.** If you turn off `bootstrap`, there are no out-of-bag rows.

## 9. Check yourself

**Exercises** ([exercises.ipynb](exercises.ipynb))

1. By hand: for $n = 4$, compute the probability that a given row is out of bag. Then draw 10,000 bootstrap samples of size 4 in code and check.
2. Compare `oob_score_` with 5-fold CV accuracy for the forest. How close are they?
3. Try `max_features` = 1, 2, 3, 5 (5 means "no feature randomness", i.e. plain bagging). Which gives the best CV AUC? What does that tell you about diversity?
4. Try `min_samples_leaf` = 1, 5, 10, 25, 50. Plot CV AUC. Where is the sweet spot?

**Quiz**

1. What is "with replacement", and why does it matter for bagging?
2. About what fraction of rows is out of bag for each tree, and why?
3. What two sources of randomness make a random forest's trees different?
4. Can adding more trees make a random forest overfit?
5. Why did the forest only tie logistic regression on the churn table?

## 10. Recap

- **Bootstrap** = draw $n$ rows with replacement; about 63% of rows are in each bag, 37% out of bag.
- **Bagging** averages models trained on different bootstrap samples; **random forest** adds a random feature subset at every split, making trees more different.
- The **OOB score** is a free validation estimate.
- On ShopWave churn, a forest lifts one overfit tree from AUC 0.59 to 0.73–0.74, but only ties logistic regression, because the signal is mostly straight-line.
- More trees = more stable, never more overfit; control overfitting with leaf size and depth instead.

**Next:** [05.3 Boosting by hand](../03-boosting-by-hand/): instead of averaging independent trees, build them one after another, each fixing the last one's mistakes.
